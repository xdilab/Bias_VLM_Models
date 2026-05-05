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
from sklearn.metrics import r2_score, accuracy_score
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


DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
dtype_to_use = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print(f"vlm113-Multitask-Phy Initialized: Using {DEVICE} with {dtype_to_use} precision.")



class JaccardLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred_probs = torch.sigmoid(y_pred)
        y_pred_flat = y_pred_probs.view(y_pred.shape[0], -1)
        y_true_flat = y_true.view(y_true.shape[0], -1).float()

        intersection = (y_pred_flat * y_true_flat).sum(1)
        union = (y_pred_flat + y_true_flat).sum(1) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou.mean()

class PhysicsEvaluator:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.true_grades, self.pred_grades = [], []
        self.true_areas, self.pred_areas = [], []
        self.true_masses, self.pred_masses = [], []
        self.true_diams, self.pred_diams = [], []

    def parse_text(self, text: str):
        grade, area, mass, diam = 0, 0.0, 0.0, 0.0
        g_match = re.search(r"grade:\s*(\d)", text, re.IGNORECASE)
        if g_match: grade = int(g_match.group(1))
        a_match = re.search(r"area:\s*([\d\.]+)", text, re.IGNORECASE)
        if a_match:
            try: area = float(a_match.group(1)) 
            except: pass
        m_match = re.search(r"mass:\s*([\d\.]+)", text, re.IGNORECASE)
        if m_match:
            try: mass = float(m_match.group(1)) 
            except: pass
        d_match = re.search(r"diameter:\s*([\d\.]+)", text, re.IGNORECASE)
        if d_match:
            try: diam = float(d_match.group(1)) 
            except: pass
        return grade, area, mass, diam

    def update(self, true_texts: List[str], pred_texts: List[str]):
        for t_txt, p_txt in zip(true_texts, pred_texts):
            t_grade, t_area, t_mass, t_diam = self.parse_text(t_txt)
            p_grade, p_area, p_mass, p_diam = self.parse_text(p_txt)
            self.true_grades.append(t_grade); self.pred_grades.append(p_grade)
            self.true_areas.append(t_area); self.pred_areas.append(p_area)
            self.true_masses.append(t_mass); self.pred_masses.append(p_mass)
            self.true_diams.append(t_diam); self.pred_diams.append(p_diam)

    def compute_metrics(self):
        metrics = {"Grade_Acc": accuracy_score(self.true_grades, self.pred_grades) if self.true_grades else 0.0}
        if len(self.true_areas) > 1:
            metrics["Area_R2"] = r2_score(self.true_areas, self.pred_areas)
            metrics["Mass_R2"] = r2_score(self.true_masses, self.pred_masses)
            metrics["Diam_R2"] = r2_score(self.true_diams, self.pred_diams)
        else:
            for k in ["Area_R2", "Mass_R2", "Diam_R2"]: metrics[k] = 0.0
        return metrics



class VLM_PhysicsSegDataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, base_img_dir: str):
        self.image_paths, self.mask_paths, self.questions, self.answers, self.has_tumors = [], [], [], [], []
        self.mask_transform = transforms.Compose([
            transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Loading Multitask Data"):
            raw_path = str(row['image_path'])
            if "kaggle_3m/" in raw_path:
                img_path = os.path.join(base_img_dir, raw_path[raw_path.find("kaggle_3m/"):])
            else:
                img_path = raw_path

            mask_path = img_path.replace(".tif", "_mask.tif")
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                continue

            has_tumor = row['has_tumor']
            q = ("Analyze this MRI. Provide the histologic grade (1 or 2), "
                 "tumor area (mm^2), estimated mass (g), and max diameter (mm).")

            if has_tumor:
                a = (f"A tumor is visible. Grade: {int(float(row['grade']))}. "
                     f"Area: {row['tumor_area_mm2']:.2f} mm^2. Mass: {row['tumor_mass_g']:.4f} g. "
                     f"Diameter: {row['tumor_diameter_mm']:.2f} mm.")
            else:
                a = "No tumor is visible in this MRI scan. Grade: 0. Area: 0.0 mm^2. Mass: 0.0 g. Diameter: 0.0 mm."

            self.image_paths.append(img_path)
            self.mask_paths.append(mask_path)
            self.questions.append(q)
            self.answers.append(a)
            self.has_tumors.append(bool(has_tumor))

    def __len__(self) -> int: return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask_pil = Image.open(self.mask_paths[idx]).convert("L")
        mask_tensor = (self.mask_transform(mask_pil) > 0).float()
        return image, mask_tensor, self.questions[idx], self.answers[idx], self.has_tumors[idx]

def vlm_seg_collate_fn(batch):
    imgs, masks, qs, ans, tumors = zip(*batch)
    return list(imgs), torch.stack(masks), list(qs), list(ans), torch.tensor(tumors)



class LingshuMultitaskWrapper(nn.Module):
    def __init__(self, vlm_model: nn.Module, seg_out_size=(336, 336)):
        super().__init__()
        self.vlm = vlm_model
        self.seg_out_size = seg_out_size
        self.base_vlm = self.vlm.base_model if hasattr(self.vlm, "base_model") else self.vlm
        self.visual = self.base_vlm.model.visual
        
        self.deeplab = smp.DeepLabV3(
            encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1
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
        return tokens_reshaped.transpose(1, 2).contiguous().view(batch_size, actual_channels, h_feats, w_feats)

    def forward(self, **batch):
        pixel_values = batch["pixel_values"]
        grid_thw = batch.get("image_grid_thw")
        batch_size = batch["input_ids"].size(0)

       
        vis_grid = self._get_visual_grid(pixel_values, grid_thw, batch_size)
        adapted_features = self.feature_adapter(vis_grid)
        decoder_out = self.deeplab.decoder([adapted_features])
        seg_logits_raw = self.deeplab.segmentation_head(decoder_out)
        seg_logits = F.interpolate(seg_logits_raw, size=self.seg_out_size, mode="bilinear").squeeze(1)

        
        vlm_out = self.vlm(**batch, return_dict=True)

        return {"vqa_loss": vlm_out.loss, "vqa_logits": vlm_out.logits, "seg_logits": seg_logits}



def run_evaluation(model, processor, loader, device, desc="Eval"):
    model.eval()
    physics_eval = PhysicsEvaluator()
    total_iou, total_samples = 0, 0
    debug_shown = False
    
    with torch.no_grad():
        for imgs, masks, qs, ans, tumors in tqdm(loader, desc=desc):
            masks_gt = masks.to(device).squeeze(1)
        
            prompts = [processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}], 
                       tokenize=False, add_generation_prompt=True) for q in qs]
            gen_in = processor(text=prompts, images=imgs, return_tensors="pt", padding=True).to(device)
            with autocast(dtype=torch.bfloat16):
                gen_ids = model.vlm.generate(**gen_in, max_new_tokens=100)
            preds = processor.batch_decode([g[len(i):] for g, i in zip(gen_ids, gen_in.input_ids)], skip_special_tokens=True)
            
         
            batch_prep = processor(text=prompts, images=imgs, padding=True, return_tensors="pt").to(device)
            outputs = model(**batch_prep)
            
            pred_masks = (torch.sigmoid(outputs["seg_logits"]) > 0.5).float()
            intersection = (pred_masks * masks_gt).sum(dim=(1, 2))
            union = pred_masks.sum(dim=(1, 2)) + masks_gt.sum(dim=(1, 2)) - intersection
            total_iou += ((intersection + 1e-6) / (union + 1e-6)).mean().item()
            
            physics_eval.update(ans, preds)
            total_samples += len(ans)

            if not debug_shown:
                print(f"\n[SAMPLE] PRED: {preds[0]} | TRUE: {ans[0]}")
                debug_shown = True

    m = physics_eval.compute_metrics()
    avg_iou = total_iou / len(loader) if len(loader) > 0 else 0.0
    
    print(f"\n--- {desc} Results ---")
    print(f"  Grade Acc: {m['Grade_Acc']*100:.2f}% | IoU: {avg_iou:.4f}")
    print(f"  Area R2: {m['Area_R2']:.4f} | Mass R2: {m['Mass_R2']:.4f} | Diam R2: {m['Diam_R2']:.4f}")
    
   
    physics_avg_r2 = (m['Area_R2'] + m['Mass_R2'] + m['Diam_R2']) / 3.0
    return m['Grade_Acc'] + physics_avg_r2 + avg_iou



if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = {
        "model_path": "/home/ealam/vlm/models/Lingshu-7B/",
        "csv_path": "/home/ealam/vlm/mri_dataset/lgg_physics_metadata_v2.csv",
        "base_img_dir": "/home/ealam/vlm/mri_dataset/",
        "save_path": os.path.join(script_dir, "lingshu_v113_multitask"),
        "lr": 1e-4, "batch_size": 2, "epochs": 25, "grad_accum": 4, "seg_weight": 3.0,
        "patience": 5, "seed": 42
    }

    random.seed(config["seed"]); torch.manual_seed(config["seed"]); np.random.seed(config["seed"])

    print("Step 1: Initializing Multitask Lingshu...")
    processor = AutoProcessor.from_pretrained(config["model_path"], trust_remote_code=True)
    base = AutoModelForVision2Seq.from_pretrained(config["model_path"], torch_dtype=torch.bfloat16, device_map={"": DEVICE})
    
    lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = LingshuMultitaskWrapper(get_peft_model(base, lora_cfg)).to(DEVICE)

    print("Step 2: Preparing Three-Way Data Split...")
    df = pd.read_csv(config["csv_path"])
    train_val_df, test_df = train_test_split(df, test_size=0.20, random_state=config["seed"])
    train_df, val_df = train_test_split(train_val_df, test_size=0.20, random_state=config["seed"])

    train_loader = DataLoader(VLM_PhysicsSegDataset(train_df, config["base_img_dir"]), batch_size=config["batch_size"], shuffle=True, collate_fn=vlm_seg_collate_fn)
    val_loader = DataLoader(VLM_PhysicsSegDataset(val_df, config["base_img_dir"]), batch_size=config["batch_size"], collate_fn=vlm_seg_collate_fn)
    test_loader = DataLoader(VLM_PhysicsSegDataset(test_df, config["base_img_dir"]), batch_size=config["batch_size"], collate_fn=vlm_seg_collate_fn)

    optimizer = AdamW(model.parameters(), lr=config["lr"])
    seg_loss_iou = JaccardLoss().to(DEVICE)
    seg_loss_bce = nn.BCEWithLogitsLoss().to(DEVICE)

    best_score, patience_counter = -float("inf"), 0

    print("Step 3: Joint Training (VQA + Seg)...")
    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        for step, (imgs, masks, qs, ans, tumors) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            # Build training batch with assistant masking
            texts = []
            for q, a in zip(qs, ans):
                msg = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
                    {"role": "assistant", "content": [{"type": "text", "text": a}]}
                ]
                texts.append(processor.apply_chat_template(msg, tokenize=False) + processor.tokenizer.eos_token)
            
            batch = processor(text=texts, images=imgs, padding=True, return_tensors="pt").to(DEVICE)
            
            labels = batch.input_ids.clone()
            marker = "<|im_start|>assistant\n"
            for i, text in enumerate(texts):
                if marker in text:
                    prefix_len = len(processor.tokenizer.encode(text.split(marker)[0] + marker, add_special_tokens=False))
                    labels[i, :prefix_len] = -100
            batch["labels"] = labels
            
            with autocast(dtype=torch.bfloat16):
                outputs = model(**batch)
                gt_masks = masks.to(DEVICE).squeeze(1)
                s_loss = (0.5 * seg_loss_bce(outputs["seg_logits"], gt_masks)) + (0.5 * seg_loss_iou(outputs["seg_logits"], gt_masks))
                loss = outputs["vqa_loss"] + (config["seg_weight"] * s_loss)
            
            (loss / config["grad_accum"]).backward()
            if (step + 1) % config["grad_accum"] == 0:
                optimizer.step(); optimizer.zero_grad()

   
        score = run_evaluation(model, processor, val_loader, DEVICE, f"Validation Ep {epoch+1}")
        
        if score > best_score:
            best_score = score
            patience_counter = 0
          
            model.vlm.save_pretrained(config["save_path"])
            torch.save(model.state_dict(), os.path.join(config["save_path"], "multitask_weights.pth"))
            print(f"Metrics Improved: Best Score {best_score:.4f} - Checkpoint Saved.")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print("Early stopping triggered.")
                break

    
    if os.path.exists(config["save_path"]):
        print("\nStep 4: Final Benchmarking (Reloading Best Weights)...")
        
        del model
        torch.cuda.empty_cache()
        
      
        base_test = AutoModelForVision2Seq.from_pretrained(config["model_path"], torch_dtype=torch.bfloat16, device_map={"": DEVICE})
        model = LingshuMultitaskWrapper(PeftModel.from_pretrained(base_test, config["save_path"])).to(DEVICE)
        weights_path = os.path.join(config["save_path"], "multitask_weights.pth")
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        
        run_evaluation(model, processor, test_loader, DEVICE, "Final Test Evaluation")

    print("Multitask Training process complete.")
