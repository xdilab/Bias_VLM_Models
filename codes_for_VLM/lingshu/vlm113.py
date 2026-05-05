import os
import glob
import math
import re
import random
import logging
import warnings
from typing import List, Tuple, Dict

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

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss



class VLM_QASegDataset(Dataset):
    def __init__(self, image_paths: List[str], metadata_df: pd.DataFrame, is_train: bool = True):
        self.image_paths = []
        self.mask_paths = []
        self.questions = []
        self.answers = []
        self.has_tumors = []

        self.mask_transform = transforms.Compose([
            transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        mdx = metadata_df.set_index("Patient")

        for img_path in tqdm(image_paths, desc="Loading Dataset"):
            mask_path = img_path.replace(".tif", "_mask.tif")
            if not os.path.exists(mask_path):
                continue

            mask_arr = np.array(Image.open(mask_path))
            has_tumor = np.any(mask_arr > 0)

            q = "Is there a tumor visible in this MRI? If so, what is its histologic grade: one or two?"

            if has_tumor:
                pid_folder = os.path.basename(os.path.dirname(img_path))
                pid_key = "_".join(pid_folder.split("_")[0:3])
                if pid_key not in mdx.index: continue
                row = mdx.loc[[pid_key]].iloc[0]
                grade = row.get("neoplasm_histologic_grade")
                if pd.isna(grade) or int(grade) not in [1, 2]: continue
                a = f"A tumor is visible. The grade of the tumor is {'two' if int(grade) == 2 else 'one'}."
            else:
                a = "No tumor is visible in this MRI scan."

            self.image_paths.append(img_path)
            self.mask_paths.append(mask_path)
            self.questions.append(q)
            self.answers.append(a)
            self.has_tumors.append(bool(has_tumor))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask_pil = Image.open(self.mask_paths[idx]).convert("L")
        mask_tensor = (self.mask_transform(mask_pil) > 0).float()
        return image, mask_tensor, self.questions[idx], self.answers[idx], self.has_tumors[idx]

def vlm_seg_collate_fn(batch):
    images, masks, questions, answers, has_tumors = zip(*batch)
    return list(images), torch.stack(masks), list(questions), list(answers), torch.tensor(has_tumors)



def build_training_batch(images, masks, questions, answers, has_tumors, processor):
    full_texts, prompts = [], []
    for q, a in zip(questions, answers):
        msg_full = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
            {"role": "assistant", "content": [{"type": "text", "text": a}]}
        ]
        text_full = processor.apply_chat_template(msg_full, tokenize=False, add_generation_prompt=False)
        full_texts.append(text_full + processor.tokenizer.eos_token)

        msg_q = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        text_q = processor.apply_chat_template(msg_q, tokenize=False, add_generation_prompt=True)
        prompts.append(text_q)

    inputs = processor(text=full_texts, images=images, padding=True, return_tensors="pt")
    prompt_inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt")

    labels = inputs.input_ids.clone()
    for i in range(labels.size(0)):
        p_len = prompt_inputs.attention_mask[i].sum().item()
        labels[i, :p_len] = -100

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
        
        self.vis_hidden = self.base_vlm.config.vision_config.hidden_size

        self.deeplab = smp.DeepLabV3(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )

        self.encoder_out_channels = self.deeplab.encoder.out_channels[-1]
        self.feature_adapter = None 

    def _get_visual_grid(self, pixel_values, grid_thw, batch_size):
        
        vis_out = self.visual(pixel_values, grid_thw=grid_thw)
        tokens = vis_out.last_hidden_state if hasattr(vis_out, "last_hidden_state") else vis_out

        num_tokens = tokens.shape[0]
        tokens_per_sample = num_tokens // batch_size
        actual_channels = tokens.shape[-1]

       
        if self.feature_adapter is None:
            self.feature_adapter = nn.Conv2d(actual_channels, self.encoder_out_channels, kernel_size=1).to(tokens.device).to(tokens.dtype)
            self.deeplab.to(tokens.device).to(tokens.dtype) 

      
        orig_h, orig_w = grid_thw[0, 1].item(), grid_thw[0, 2].item()
        pooling_factor = int(math.sqrt((orig_h * orig_w) / tokens_per_sample))
        h_feats, w_feats = orig_h // pooling_factor, orig_w // pooling_factor

 
        tokens_reshaped = tokens.view(batch_size, tokens_per_sample, actual_channels)
        feat_map = tokens_reshaped.transpose(1, 2).contiguous().view(batch_size, actual_channels, h_feats, w_feats)

        return feat_map

    def forward(self, **batch):
        seg_masks_gt = batch.pop("seg_masks_gt", None)
        has_tumor = batch.pop("has_tumor", None)
        
        pixel_values = batch["pixel_values"]
        grid_thw = batch.get("image_grid_thw")
        batch_size = batch["input_ids"].size(0)

        with autocast(dtype=torch.bfloat16):
            vis_grid = self._get_visual_grid(pixel_values, grid_thw, batch_size)
            adapted_features = self.feature_adapter(vis_grid)
            decoder_out = self.deeplab.decoder([adapted_features])
            seg_logits_raw = self.deeplab.segmentation_head(decoder_out)
            
            seg_logits = F.interpolate(
                seg_logits_raw, size=self.seg_out_size, mode="bilinear", align_corners=False
            ).squeeze(1)

        vlm_out = self.vlm(**batch, return_dict=True)

        return {
            "vqa_loss": vlm_out.loss,
            "vqa_logits": vlm_out.logits,
            "seg_logits": seg_logits
        }



def run_evaluation(model, processor, loader, device, desc="Eval"):
    model.eval()
    vlm_correct, total_samples, total_iou, total_vqa_loss = 0, 0, 0, 0
    with torch.no_grad():
        for images, masks, questions, answers, has_tumors in tqdm(loader, desc=desc):
            masks = masks.to(device)
            B = len(answers)
            prompts = []
            for q in questions:
                m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
                prompts.append(processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True))
            
            gen_in = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
            gen_ids = model.vlm.generate(**gen_in, max_new_tokens=25)
            gen_ids = [g[len(i):] for g, i in zip(gen_ids, gen_in.input_ids)]
            preds = processor.batch_decode(gen_ids, skip_special_tokens=True)

            for p, a in zip(preds, answers):
                p_low, a_low = p.lower(), a.lower()
                if "no tumor" in a_low:
                    if "no tumor" in p_low: vlm_correct += 1
                else:
                    want_two = "two" in a_low
                    has_one = "one" in p_low or "1" in p_low
                    has_two = "two" in p_low or "2" in p_low
                    if (want_two and has_two and not has_one) or (not want_two and has_one and not has_two):
                        vlm_correct += 1
            
            batch_cpu = build_training_batch(images, masks.cpu(), questions, answers, has_tumors, processor)
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_cpu.items()}
            
            with autocast(dtype=torch.bfloat16):
                out = model(**batch_gpu)
            
            if out["vqa_loss"] is not None: total_vqa_loss += out["vqa_loss"].item()
            
            pred_masks = (torch.sigmoid(out["seg_logits"]) > 0.5).float()
            intersection = (pred_masks * masks.squeeze(1)).sum(dim=(1, 2))
            union = pred_masks.sum(dim=(1, 2)) + masks.squeeze(1).sum(dim=(1, 2)) - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            total_iou += iou.mean().item()
            total_samples += B

    avg_acc = (vlm_correct / total_samples) * 100
    avg_iou = total_iou / len(loader) if len(loader) > 0 else 0
    avg_loss = total_vqa_loss / len(loader) if len(loader) > 0 else 0
    print(f"\n--- {desc} ---\nQA Accuracy: {avg_acc:.2f}% | IoU: {avg_iou:.4f} | VQA Loss: {avg_loss:.4f}")
    return avg_acc + (avg_iou * 100)



if __name__ == "__main__":
    cfg = {
        "model_path": "/home/ealam/vlm/models/Lingshu-7B/",
        "dataset_path": "/home/ealam/vlm/mri_dataset/kaggle_3m/",
        "csv_path": "/home/ealam/vlm/mri_dataset/kaggle_3m/data.csv",
        "save_path": "./lingshu_vlm113_multitask",
        "lr": 1e-4, "batch_size": 2, "epochs": 25, "early_stopping_patience": 5,
        "seg_weight": 3.0, "tumor_seg_weight": 1.0, "grad_clip": 1.0, "device": "cuda:0" 
    }

    print(f"Starting VLM113 Multitask (Lingshu) on {cfg['device']}")
    processor = AutoProcessor.from_pretrained(cfg["model_path"], trust_remote_code=True)
    base_model = AutoModelForVision2Seq.from_pretrained(
        cfg["model_path"], torch_dtype=torch.bfloat16, device_map={"": cfg["device"]}, trust_remote_code=True
    )

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM"
    )
    model = LingshuSegmentationWrapper(get_peft_model(base_model, lora_cfg)).to(cfg["device"])

    df = pd.read_csv(cfg["csv_path"])
    all_paths = [p.replace("_mask.tif", ".tif") for p in glob.glob(os.path.join(cfg["dataset_path"], "*", "*_mask.tif"))]
    tr_val_p, test_p = train_test_split(all_paths, test_size=0.20, random_state=42)
    tr_p, val_p = train_test_split(tr_val_p, test_size=0.20, random_state=42)

    train_loader = DataLoader(VLM_QASegDataset(tr_p, df), batch_size=cfg["batch_size"], shuffle=True, collate_fn=vlm_seg_collate_fn)
    val_loader = DataLoader(VLM_QASegDataset(val_p, df), batch_size=cfg["batch_size"], collate_fn=vlm_seg_collate_fn)
    test_loader = DataLoader(VLM_QASegDataset(test_p, df), batch_size=cfg["batch_size"], collate_fn=vlm_seg_collate_fn)

    optimizer = AdamW(model.parameters(), lr=cfg["lr"])
    
   
    best_metric, patience_counter = 0.0, 0
    seg_loss_fn_iou = JaccardLoss(reduction="none").to(cfg["device"])
    seg_loss_fn_bce = nn.BCEWithLogitsLoss(reduction="none").to(cfg["device"])

    print("Beginning Training...")
    for epoch in range(cfg["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for images, masks, questions, answers, has_tumors in pbar:
            batch_cpu = build_training_batch(images, masks, questions, answers, has_tumors, processor)
            batch_gpu = {k: v.to(cfg["device"]) if isinstance(v, torch.Tensor) else v for k, v in batch_cpu.items()}
            
            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16):
                outputs = model(**batch_gpu)
                gt_masks, logits = batch_gpu["seg_masks_gt"].squeeze(1), outputs["seg_logits"]
                loss_bce = seg_loss_fn_bce(logits, gt_masks).mean(dim=(1, 2))
                loss_iou = seg_loss_fn_iou(logits, gt_masks)
                per_sample_seg_loss = (0.5 * loss_bce) + (0.5 * loss_iou)
                weights = torch.ones_like(per_sample_seg_loss)
                weights[batch_gpu["has_tumor"]] = cfg["tumor_seg_weight"]
                weighted_seg_loss = (per_sample_seg_loss * weights).mean()
                combined_loss = outputs["vqa_loss"] + (cfg["seg_weight"] * weighted_seg_loss)

            if torch.isfinite(combined_loss):
                combined_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()
                pbar.set_postfix({"vqa": outputs["vqa_loss"].item(), "seg": weighted_seg_loss.item()})

        current_metric = run_evaluation(model, processor, val_loader, cfg["device"], "Validation")
        if current_metric > best_metric:
            best_metric, patience_counter = current_metric, 0
            os.makedirs(cfg["save_path"], exist_ok=True)
            model.vlm.save_pretrained(os.path.join(cfg["save_path"], "vlm_adapter"))
            torch.save(model.state_dict(), os.path.join(cfg["save_path"], "multitask_full.pth"))
            print(f"Saved Checkpoint: {best_metric:.2f}")
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stopping_patience"]:
                print("Early stopping triggered."); break

    run_evaluation(model, processor, test_loader, cfg["device"], "Final Test")
