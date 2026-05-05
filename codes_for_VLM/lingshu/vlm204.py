import os
import glob
import math
import re
import random
import logging
import warnings
from typing import List, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.segmentation import deeplabv3_resnet101
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    get_linear_schedule_with_warmup,
)


warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True



def get_external_segmentation_model() -> nn.Module:
    model = deeplabv3_resnet101(weights=None)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    return model

def get_external_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def post_process_mask(mask: np.ndarray, kernel_size: int = 5, min_area: int = 100) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    processed = np.zeros(mask.shape, dtype=np.uint8)
    if num_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        if stats[largest_idx, cv2.CC_STAT_AREA] > min_area:
            processed[labels == largest_idx] = 255
    return processed

def delineate_roi_on_image(pil_image: Image.Image, seg_model: nn.Module, seg_transform: A.Compose, device: str) -> Image.Image:
    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    augmented = seg_transform(image=img_rgb)
    tensor = augmented['image'].to(device).unsqueeze(0)
    seg_model.eval()
    with torch.no_grad():
        output = seg_model(tensor)['out']
    mask = torch.sigmoid(output).squeeze().cpu().numpy()
    binary_mask = (mask > 0.5).astype(np.uint8)
    clean_mask = post_process_mask(binary_mask)
    resized_mask = cv2.resize(clean_mask, (pil_image.width, pil_image.height), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(img_bgr, contours, -1, (0, 255, 255), 2)
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))



class JaccardLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 4 and y_pred.size(1) == 1:
            y_pred = y_pred[:, 0, :, :]
        if y_true.dim() == 4 and y_true.size(1) == 1:
            y_true = y_true[:, 0, :, :]
        y_pred_probs = torch.sigmoid(y_pred)
        y_pred_flat = y_pred_probs.view(y_pred.shape[0], -1)
        y_true_flat = y_true.view(y_true.shape[0], -1).float()
        intersection = (y_pred_flat * y_true_flat).sum(1)
        total = (y_pred_flat + y_true_flat).sum(1)
        union = total - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - iou
        return loss.mean() if self.reduction == "mean" else loss



class VLM_QADelineatedDataset(Dataset):
    def __init__(self, image_paths: List[str], metadata_df: pd.DataFrame, seg_model: nn.Module, seg_transform: A.Compose, device: str):
        self.image_paths: List[str] = []
        self.questions: List[str] = []
        self.answers: List[str] = []
        self.has_tumors: List[bool] = []
        self.seg_model = seg_model
        self.seg_transform = seg_transform
        self.device = device
        self.gt_mask_transform = transforms.Compose([
            transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        mdx = metadata_df.set_index("Patient")
        for img_path in tqdm(image_paths, desc="Processing Delineated Dataset"):
            mask_path = img_path.replace(".tif", "_mask.tif")
            if not os.path.exists(mask_path): continue
            gt_mask = np.array(Image.open(mask_path))
            has_tumor = np.any(gt_mask > 0)
            q = "Is there a tumor visible in this MRI? If so, what is its histologic grade: one or two?"
            if has_tumor:
                pid_folder = os.path.basename(os.path.dirname(img_path))
                pid_key = "_".join(pid_folder.split("_")[0:3])
                if pid_key in mdx.index:
                    row = mdx.loc[[pid_key]].iloc[0]
                    grade = row.get("neoplasm_histologic_grade")
                    if pd.notna(grade) and int(grade) in [1, 2]:
                        self.image_paths.append(img_path)
                        self.questions.append(q)
                        self.answers.append(f"A tumor is visible. The grade of the tumor is {'two' if int(grade) == 2 else 'one'}.")
                        self.has_tumors.append(True)
            else:
                self.image_paths.append(img_path)
                self.questions.append(q)
                self.answers.append("No tumor is visible in this MRI scan.")
                self.has_tumors.append(False)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        raw_pil = Image.open(img_path).convert("RGB")
        delineated_pil = delineate_roi_on_image(raw_pil, self.seg_model, self.seg_transform, self.device)
        mask_path = img_path.replace(".tif", "_mask.tif")
        gt_mask_pil = Image.open(mask_path).convert("L")
        gt_mask_tensor = (self.gt_mask_transform(gt_mask_pil) > 0).float()
        return delineated_pil, gt_mask_tensor, self.questions[idx], self.answers[idx], self.has_tumors[idx]

def vlm_collate_fn(batch):
    images, masks, questions, answers, has_tumors = zip(*batch)
    return list(images), torch.stack(masks), list(questions), list(answers), torch.tensor(has_tumors)



def build_training_batch(images, masks, questions, answers, has_tumors, processor):
    full_texts, prompts = [], []
    for q, a in zip(questions, answers):
        msg_full = [{"role": "user", "content": [{"type": "image", "image": None}, {"type": "text", "text": q}]},
                    {"role": "assistant", "content": [{"type": "text", "text": a}]}]
        full_texts.append(processor.apply_chat_template(msg_full, tokenize=False, add_generation_prompt=False) + processor.tokenizer.eos_token)
        msg_q = [{"role": "user", "content": [{"type": "image", "image": None}, {"type": "text", "text": q}]}]
        prompts.append(processor.apply_chat_template(msg_q, tokenize=False, add_generation_prompt=True))
    inputs = processor(text=full_texts, images=images, padding=True, return_tensors="pt")
    prompt_inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")
    labels = inputs.input_ids.clone()
    for i in range(labels.size(0)):
        labels[i, :prompt_inputs.attention_mask[i].sum().item()] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    inputs["seg_masks_gt"] = masks
    inputs["has_tumor"] = has_tumors
    return inputs



class LingshuSegmentationWrapper(nn.Module):
    def __init__(self, vlm_model: nn.Module, seg_out_size=(336, 336)):
        super().__init__()
        self.vlm = vlm_model
        self.seg_out_size = seg_out_size
        self.base_vlm = self.vlm.base_model if hasattr(self.vlm, "base_model") else self.vlm
        self.visual = self.base_vlm.model.visual
        self.deeplab = smp.DeepLabV3(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        self.encoder_out_channels = self.deeplab.encoder.out_channels[-1]
        
      
        self.feature_adapter = nn.Conv2d(3584, self.encoder_out_channels, kernel_size=1)

    def _get_visual_grid(self, pixel_values, grid_thw, batch_size):
        vis_out = self.visual(pixel_values, grid_thw=grid_thw)
        tokens = vis_out.last_hidden_state if hasattr(vis_out, "last_hidden_state") else vis_out
        num_tokens, actual_channels = tokens.shape[0], tokens.shape[-1]
        tokens_per_sample = num_tokens // batch_size
        orig_h, orig_w = grid_thw[0, 1].item(), grid_thw[0, 2].item()
        pooling_factor = int(math.sqrt((orig_h * orig_w) / tokens_per_sample))
        h_f, w_f = orig_h // pooling_factor, orig_w // pooling_factor
        feat_map = tokens.view(batch_size, tokens_per_sample, actual_channels).transpose(1, 2).contiguous().view(batch_size, actual_channels, h_f, w_f)
        return feat_map

    def forward(self, **batch):
        seg_masks_gt = batch.pop("seg_masks_gt", None)
        has_tumor = batch.pop("has_tumor", None)
        pixel_values, grid_thw, b_size = batch["pixel_values"], batch.get("image_grid_thw"), batch["input_ids"].size(0)
        with autocast(dtype=torch.bfloat16):
            vis_grid = self._get_visual_grid(pixel_values, grid_thw, b_size)
            decoder_out = self.deeplab.decoder([self.feature_adapter(vis_grid)])
            seg_logits_raw = self.deeplab.segmentation_head(decoder_out)
            seg_logits = F.interpolate(seg_logits_raw, size=self.seg_out_size, mode="bilinear", align_corners=False).squeeze(1)
        vlm_out = self.vlm(**batch, return_dict=True)
        return {"vqa_loss": vlm_out.loss, "vqa_logits": vlm_out.logits, "seg_logits": seg_logits}



def run_evaluation(model, processor, loader, device, desc="Eval"):
    model.eval()
    vlm_correct, total_samples, total_iou, total_vqa_loss = 0, 0, 0, 0
    with torch.no_grad():
        for images, masks, questions, answers, has_tumors in tqdm(loader, desc=desc):
            masks = masks.to(device)
            B = len(answers)
            prompts = [processor.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": None}, {"type": "text", "text": q}]}], tokenize=False, add_generation_prompt=True) for q in questions]
            gen_in = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
            gen_ids = model.vlm.generate(**gen_in, max_new_tokens=30)
            preds = processor.batch_decode([g[len(i):] for g, i in zip(gen_ids, gen_in.input_ids)], skip_special_tokens=True)
            for p, a in zip(preds, answers):
                p_l, a_l = p.lower(), a.lower()
                if "no tumor" in a_l:
                    if "no tumor" in p_l: vlm_correct += 1
                else:
                    want_two = "two" in a_l
                    h1, h2 = ("one" in p_l or "1" in p_l), ("two" in p_l or "2" in p_l)
                    if (want_two and h2 and not h1) or (not want_two and h1 and not h2): vlm_correct += 1
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in build_training_batch(images, masks.cpu(), questions, answers, has_tumors, processor).items()}
            with autocast(dtype=torch.bfloat16):
                out = model(**batch_gpu)
            if out["vqa_loss"] is not None: total_vqa_loss += out["vqa_loss"].item()
            pred_m = (torch.sigmoid(out["seg_logits"]) > 0.5).float()
            gt_m = masks.squeeze(1)
            inter = (pred_m * gt_m).sum(dim=(1,2))
            iou = inter / (pred_m.sum(dim=(1,2)) + gt_m.sum(dim=(1,2)) - inter + 1e-6)
            total_iou += iou[has_tumors].mean().item() if has_tumors.any() else 0
            total_samples += B
    avg_acc, avg_iou = (vlm_correct / total_samples) * 100, total_iou / len(loader)
    print(f"\n--- {desc} ---\nQA Accuracy: {avg_acc:.2f}% | Tumor IoU: {avg_iou:.4f}")
    return avg_acc + (avg_iou * 100)



if __name__ == "__main__":
    cfg = {
        "model_path": "/home/ealam/vlm/models/Lingshu-7B/",
        "dataset_path": "/home/ealam/vlm/mri_dataset/kaggle_3m/",
        "csv_path": "/home/ealam/vlm/mri_dataset/kaggle_3m/data.csv",
        "seg_model_path": "/home/ealam/vlm/best_model_segmentation_v2.pth",
        "save_path": "./lingshu_vlm204_delineated",
        "lr": 1e-4, "batch_size": 4, "epochs": 25, "early_stopping_patience": 5,
        "seg_weight": 5.0, "tumor_seg_weight": 4.0, "grad_clip": 1.0, "device": "cuda:1" 
    }
    print("Step 0: Loading Delineator Oracle...")
    hint_model = get_external_segmentation_model()
    hint_model.load_state_dict(torch.load(cfg["seg_model_path"], map_location=cfg["device"]), strict=False)
    hint_model.to(cfg["device"]).eval()
    hint_transform = get_external_transforms()
    processor = AutoProcessor.from_pretrained(cfg["model_path"], trust_remote_code=True)
    base_model = AutoModelForVision2Seq.from_pretrained(cfg["model_path"], torch_dtype=torch.bfloat16, device_map={"": cfg["device"]}, trust_remote_code=True)
    lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0.05, task_type="CAUSAL_LM")
    model = LingshuSegmentationWrapper(get_peft_model(base_model, lora_cfg)).to(cfg["device"])
    df = pd.read_csv(cfg["csv_path"])
    all_p = [p.replace("_mask.tif", ".tif") for p in glob.glob(os.path.join(cfg["dataset_path"], "*", "*_mask.tif"))]
    tr_v_p, test_p = train_test_split(all_p, test_size=0.20, random_state=42)
    tr_p, val_p = train_test_split(tr_v_p, test_size=0.20, random_state=42)
    train_loader = DataLoader(VLM_QADelineatedDataset(tr_p, df, hint_model, hint_transform, cfg["device"]), batch_size=cfg["batch_size"], shuffle=True, collate_fn=vlm_collate_fn)
    val_loader = DataLoader(VLM_QADelineatedDataset(val_p, df, hint_model, hint_transform, cfg["device"]), batch_size=cfg["batch_size"], collate_fn=vlm_collate_fn)
    test_loader = DataLoader(VLM_QADelineatedDataset(test_p, df, hint_model, hint_transform, cfg["device"]), batch_size=cfg["batch_size"], collate_fn=vlm_collate_fn)
    optimizer = AdamW(model.parameters(), lr=cfg["lr"])
    best_metric, patience_counter = 0.0, 0
    seg_loss_iou, seg_loss_bce = JaccardLoss(reduction="none").to(cfg["device"]), nn.BCEWithLogitsLoss(reduction="none").to(cfg["device"])
    print("Beginning Multitask Training with Visual Guidance...")
    for epoch in range(cfg["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for images, masks, questions, answers, has_tumors in pbar:
            batch_gpu = {k: v.to(cfg["device"]) if isinstance(v, torch.Tensor) else v for k, v in build_training_batch(images, masks, questions, answers, has_tumors, processor).items()}
            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16):
                out = model(**batch_gpu)
                gt_m, logits = batch_gpu["seg_masks_gt"].squeeze(1), out["seg_logits"]
                p_seg = (0.5 * seg_loss_bce(logits, gt_m).mean(dim=(1, 2))) + (0.5 * seg_loss_iou(logits, gt_m))
                weights = torch.ones_like(p_seg); weights[batch_gpu["has_tumor"]] = cfg["tumor_seg_weight"]
                combined_loss = out["vqa_loss"] + (cfg["seg_weight"] * (p_seg * weights).mean())
            if torch.isfinite(combined_loss):
                combined_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()
                pbar.set_postfix({"vqa": out["vqa_loss"].item()})
        cur_m = run_evaluation(model, processor, val_loader, cfg["device"], "Validation")
        if cur_m > best_metric:
            best_metric, patience_counter = cur_m, 0
            model.vlm.save_pretrained(os.path.join(cfg["save_path"], "vlm_adapter"))
            torch.save(model.state_dict(), os.path.join(cfg["save_path"], "multitask_full.pth"))
            print(f"New Best: {best_metric:.2f}")
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stopping_patience"]: break
    run_evaluation(model, processor, test_loader, cfg["device"], "Final Test Set")
