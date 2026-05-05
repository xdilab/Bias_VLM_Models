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

def extract_bbox_from_oracle(pil_image: Image.Image, seg_model: nn.Module, seg_transform: A.Compose, device: str) -> str:
   
    img_rgb = np.array(pil_image.convert("RGB"))
    augmented = seg_transform(image=img_rgb)
    tensor = augmented['image'].to(device).unsqueeze(0)

    seg_model.eval()
    with torch.no_grad():
        output = seg_model(tensor)['out']

    mask = torch.sigmoid(output).squeeze().cpu().numpy()
    binary_mask = (mask > 0.5).astype(np.uint8)
    
    
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    coords = np.argwhere(binary_mask)
    if coords.size == 0:
        return "none"
    
   
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    
  
    ny0, nx0 = int((y0 / 256) * 1000), int((x0 / 256) * 1000)
    ny1, nx1 = int((y1 / 256) * 1000), int((x1 / 256) * 1000)
    
    return f"[{ny0}, {nx0}, {ny1}, {nx1}]"



class JaccardLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 4: y_pred = y_pred.squeeze(1)
        if y_true.dim() == 4: y_true = y_true.squeeze(1)
        
        y_pred_probs = torch.sigmoid(y_pred)
        y_pred_flat = y_pred_probs.view(y_pred.shape[0], -1)
        y_true_flat = y_true.view(y_true.shape[0], -1).float()
        
        intersection = (y_pred_flat * y_true_flat).sum(1)
        union = y_pred_flat.sum(1) + y_true_flat.sum(1) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - iou
        return loss.mean() if self.reduction == "mean" else loss



class VLM_QABboxDataset(Dataset):
    def __init__(self, image_paths: List[str], metadata_df: pd.DataFrame, seg_model: nn.Module, seg_transform: A.Compose, device: str):
        self.image_paths = []
        self.questions = []
        self.answers = []
        self.has_tumors = []
        self.seg_model = seg_model
        self.seg_transform = seg_transform
        self.device = device
        
        self.gt_mask_transform = transforms.Compose([
            transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        mdx = metadata_df.set_index("Patient")
        for img_path in tqdm(image_paths, desc="Processing BBox Dataset"):
            mask_path = img_path.replace(".tif", "_mask.tif")
            if not os.path.exists(mask_path): continue

            gt_mask = np.array(Image.open(mask_path))
            has_tumor = np.any(gt_mask > 0)
            
            q_base = "Is there a tumor visible in this MRI? If so, what is its histologic grade: one or two?"

            if has_tumor:
                pid_folder = os.path.basename(os.path.dirname(img_path))
                pid_key = "_".join(pid_folder.split("_")[0:3])
                if pid_key in mdx.index:
                    row = mdx.loc[[pid_key]].iloc[0]
                    grade = row.get("neoplasm_histologic_grade")
                    if pd.notna(grade) and int(grade) in [1, 2]:
                        self.image_paths.append(img_path)
                        self.questions.append(q_base)
                        self.answers.append(f"A tumor is visible. The grade of the tumor is {'two' if int(grade) == 2 else 'one'}.")
                        self.has_tumors.append(True)
            else:
                self.image_paths.append(img_path)
                self.questions.append(q_base)
                self.answers.append("No tumor is visible in this MRI scan.")
                self.has_tumors.append(False)

    def __len__(self) -> int: return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        raw_pil = Image.open(img_path).convert("RGB")
        
        bbox_str = extract_bbox_from_oracle(raw_pil, self.seg_model, self.seg_transform, self.device)
        
        if bbox_str != "none":
            hint = f"An external system indicates a region of interest at {bbox_str}. "
        else:
            hint = "No specific region of interest detected by the external system. "
            
        final_question = hint + self.questions[idx]
        
        mask_path = img_path.replace(".tif", "_mask.tif")
        gt_mask_pil = Image.open(mask_path).convert("L")
        gt_mask_tensor = (self.gt_mask_transform(gt_mask_pil) > 0).float()
        
        return raw_pil, gt_mask_tensor, final_question, self.answers[idx], self.has_tumors[idx]

def vlm_collate_fn(batch):
    images, masks, questions, answers, has_tumors = zip(*batch)
    return list(images), torch.stack(masks), list(questions), list(answers), torch.tensor(has_tumors)



class LingshuMultitaskWrapper(nn.Module):
    def __init__(self, vlm: nn.Module, seg_out_size=(336, 336)):
        super().__init__()
        self.vlm = vlm
        self.seg_out_size = seg_out_size
        base = self.vlm.base_model if hasattr(self.vlm, "base_model") else self.vlm
        self.visual = base.model.visual
        
        self.deeplab = smp.DeepLabV3(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        self.encoder_out_channels = self.deeplab.encoder.out_channels[-1]
        
        self.adapter = nn.Conv2d(3584, self.encoder_out_channels, kernel_size=1)

    def _get_grid(self, pixel_values, grid_thw, batch_size):
        vis_out = self.visual(pixel_values, grid_thw=grid_thw)
        tokens = vis_out.last_hidden_state if hasattr(vis_out, "last_hidden_state") else vis_out
        S, C = tokens.shape[0] // batch_size, tokens.shape[-1]
        
        orig_h, orig_w = grid_thw[0, 1].item(), grid_thw[0, 2].item()
        pooling_factor = int(math.sqrt((orig_h * orig_w) / S))
        h_f, w_f = orig_h // pooling_factor, orig_w // pooling_factor
        
        return tokens.view(batch_size, S, C).transpose(1, 2).contiguous().view(batch_size, actual_channels if 'actual_channels' in locals() else C, h_f, w_f)

    def forward(self, **batch):
        seg_masks_gt = batch.pop("seg_masks_gt", None)
        has_tumor = batch.pop("has_tumor", None)
        pixel_values, grid_thw, b_size = batch["pixel_values"], batch.get("image_grid_thw"), batch["input_ids"].size(0)

        with autocast(dtype=torch.bfloat16):
            grid = self._get_grid(pixel_values, grid_thw, b_size)
            decoder_out = self.deeplab.decoder([self.adapter(grid)])
            seg_logits_raw = self.deeplab.segmentation_head(decoder_out)
            seg_logits = F.interpolate(seg_logits_raw, size=self.seg_out_size, mode="bilinear", align_corners=False).squeeze(1)

        out = self.vlm(**batch, return_dict=True)
        return {"vqa_loss": out.loss, "vqa_logits": out.logits, "seg_logits": seg_logits}



def run_evaluation(model, processor, loader, device, desc="Eval"):
    model.eval()
    vlm_correct, total_samples, total_iou, count = 0, 0, 0, 0
    with torch.no_grad():
        for imgs, masks, questions, answers, has_t in tqdm(loader, desc=desc):
            masks = masks.to(device)
            B = len(answers)
            prompts = [processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}], tokenize=False, add_generation_prompt=True) for q in questions]
            
            gen_in = processor(text=prompts, images=imgs, return_tensors="pt", padding=True).to(device)
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
            
   
            msg_full = [processor.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":q}]}, {"role":"assistant","content":[{"type":"text","text":a}]}], tokenize=False) for q,a in zip(questions, answers)]
            toks = processor(text=msg_full, images=imgs, padding=True, return_tensors="pt").to(device)
            eval_labels = toks.input_ids.clone()
            eval_labels[eval_labels == processor.tokenizer.pad_token_id] = -100
            
            with autocast(dtype=torch.bfloat16):
                out = model(pixel_values=toks.pixel_values, image_grid_thw=toks.image_grid_thw, input_ids=toks.input_ids, attention_mask=toks.attention_mask, labels=eval_labels)
            
            pred_m = (torch.sigmoid(out["seg_logits"]) > 0.5).float()
            gt_m = masks.squeeze(1)
            inter = (pred_m * gt_m).sum(dim=(1,2))
            iou_v = inter / (pred_m.sum(dim=(1,2)) + gt_m.sum(dim=(1,2)) - inter + 1e-6)
            
            if has_t.any():
                total_iou += iou_v[has_t].mean().item()
                count += 1
            total_samples += B

    avg_acc = (vlm_correct / total_samples) * 100
    avg_iou = total_iou / max(1, count)
    print(f"\n--- {desc} ---\nQA Accuracy: {avg_acc:.2f}% | Tumor IoU: {avg_iou:.4f}")
    return avg_acc + (avg_iou * 100)

if __name__ == "__main__":
    cfg = {
        "model_path": "/home/ealam/vlm/models/Lingshu-7B/",
        "dataset_path": "/home/ealam/vlm/mri_dataset/kaggle_3m/",
        "csv_path": "/home/ealam/vlm/mri_dataset/kaggle_3m/data.csv",
        "seg_model_path": "/home/ealam/vlm/best_model_segmentation_v2.pth",
        "save_path": "./lingshu_vlm204_1_bbox",
        "lr": 1e-4, "batch_size": 2, "epochs": 25, "early_stopping_patience": 5,
        "seg_weight": 5.0, "tumor_seg_weight": 4.0, "grad_clip": 1.0, "device": "cuda:2" 
    }

   
    hint_oracle = get_external_segmentation_model()
    hint_oracle.load_state_dict(torch.load(cfg["seg_model_path"], map_location=cfg["device"]), strict=False)
    hint_oracle.to(cfg["device"]).eval()
    hint_transform = get_external_transforms()

  
    processor = AutoProcessor.from_pretrained(cfg["model_path"], trust_remote_code=True)
    base_model = AutoModelForVision2Seq.from_pretrained(cfg["model_path"], torch_dtype=torch.bfloat16, device_map={"": cfg["device"]}, trust_remote_code=True)
    lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0.05, task_type="CAUSAL_LM")
    model = LingshuMultitaskWrapper(get_peft_model(base_model, lora_cfg)).to(cfg["device"])

    df = pd.read_csv(cfg["csv_path"])
    all_p = [p.replace("_mask.tif", ".tif") for p in glob.glob(os.path.join(cfg["dataset_path"], "*", "*_mask.tif"))]
    tr_v_p, test_p = train_test_split(all_p, test_size=0.20, random_state=42)
    tr_p, val_p = train_test_split(tr_v_p, test_size=0.20, random_state=42)

    train_loader = DataLoader(VLM_QABboxDataset(tr_p, df, hint_oracle, hint_transform, cfg["device"]), batch_size=cfg["batch_size"], shuffle=True, collate_fn=vlm_collate_fn)
    val_loader = DataLoader(VLM_QABboxDataset(val_p, df, hint_oracle, hint_transform, cfg["device"]), batch_size=cfg["batch_size"], collate_fn=vlm_collate_fn)
    test_loader = DataLoader(VLM_QABboxDataset(test_p, df, hint_oracle, hint_transform, cfg["device"]), batch_size=cfg["batch_size"], collate_fn=vlm_collate_fn)

    optimizer = AdamW(model.parameters(), lr=cfg["lr"])
    best_metric, patience_counter = 0.0, 0
    seg_iou_fn, seg_bce_fn = JaccardLoss(reduction="none").to(cfg["device"]), nn.BCEWithLogitsLoss(reduction="none").to(cfg["device"])

    print("Beginning VLM 204.1 Training (BBox Prompts + Raw Images)...")
    for epoch in range(cfg["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, masks, qs, ans, has_t in pbar:
         
            msg_full = [processor.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":q}]}, {"role":"assistant","content":[{"type":"text","text":a}]}], tokenize=False) for q,a in zip(qs, ans)]
            toks = processor(text=msg_full, images=imgs, padding=True, return_tensors="pt").to(cfg["device"])
            
          
            labels = toks.input_ids.clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            
            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16):
                
                out = model(pixel_values=toks.pixel_values, image_grid_thw=toks.image_grid_thw, input_ids=toks.input_ids, attention_mask=toks.attention_mask, labels=labels)
                gt_m = masks.to(cfg["device"]).squeeze(1)
                p_iou, p_bce = seg_iou_fn(out["seg_logits"], gt_m), seg_bce_fn(out["seg_logits"], gt_m).mean(dim=(1, 2))
                w = torch.ones_like(p_iou); w[has_t] = cfg["tumor_seg_weight"]
                combined_loss = out["vqa_loss"] + cfg["seg_weight"] * ((0.5*p_iou + 0.5*p_bce) * w).mean()

            if torch.isfinite(combined_loss):
                combined_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()
                pbar.set_postfix({"vqa": out["vqa_loss"].item()})

        cur_m = run_evaluation(model, processor, val_loader, cfg["device"], "Validation")
        if cur_m > best_metric:
            best_metric, patience_counter = cur_m, 0
            model.vlm.save_pretrained(os.path.join(cfg["save_path"], "vlm_adapter"))
            torch.save(model.state_dict(), os.path.join(cfg["save_path"], "multitask_model.pth"))
            print(f"New Best: {best_metric:.2f}")
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stopping_patience"]: break

    run_evaluation(model, processor, test_loader, cfg["device"], "Final Test")
