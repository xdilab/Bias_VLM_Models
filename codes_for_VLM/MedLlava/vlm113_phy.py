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
    LlavaForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from torch.cuda.amp import GradScaler, autocast


warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Local Environment Initialized: Using {DEVICE}")



class SegLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(SegLoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        y_pred_probs = torch.sigmoid(y_pred)
        y_pred_flat = y_pred_probs.view(-1)
        y_true_flat = y_true.view(-1)
        intersection = (y_pred_flat * y_true_flat).sum()
        total = (y_pred_flat + y_true_flat).sum()
        union = total - intersection
        iou_loss = 1 - (intersection + self.smooth) / (union + self.smooth)
        return 0.5 * iou_loss + 0.5 * self.bce(y_pred, y_true)

class PhysicsEvaluator:
    def __init__(self):
        self.reset()
    def reset(self):
        self.true_grades, self.pred_grades = [], []
        self.true_areas, self.pred_areas = [], []
        self.true_masses, self.pred_masses = [], []
        self.true_diams, self.pred_diams = [], []
        self.total_samples = 0

    def parse_text(self, text: str):
        grade, area, mass, diam = 0, 0.0, 0.0, 0.0
        text = text.lower().replace("one", "1").replace("two", "2")
        g_match = re.search(r"grade:\s*(\d)", text)
        if g_match: grade = int(g_match.group(1))
        a_match = re.search(r"area:\s*([\d\.]+)", text)
        if a_match:
            try: area = float(a_match.group(1)) 
            except: pass
        m_match = re.search(r"mass:\s*([\d\.]+)", text)
        if m_match:
            try: mass = float(m_match.group(1)) 
            except: pass
        d_match = re.search(r"diameter:\s*([\d\.]+)", text)
        if d_match:
            try: diam = float(d_match.group(1)) 
            except: pass
        return grade, area, mass, diam

    def update(self, true_texts: List[str], pred_texts: List[str]):
        for t_txt, p_txt in zip(true_texts, pred_texts):
            self.total_samples += 1
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



class VLM_PhysicsMultiTaskDataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, base_img_dir: str):
        self.image_paths, self.mask_paths, self.questions, self.answers = [], [], [], []
        self.img_transform = transforms.Compose([transforms.Resize((336, 336))])
        self.mask_transform = transforms.Compose([
            transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        for _, row in metadata_df.iterrows():
            raw_path = str(row['image_path'])
            img_path = raw_path if os.path.exists(raw_path) else os.path.join(base_img_dir, "kaggle_3m", raw_path.split("kaggle_3m/")[-1])
            mask_path = img_path.replace(".tif", "_mask.tif")
            if not os.path.exists(img_path) or not os.path.exists(mask_path): continue

            q = ("Analyze this MRI scan. Identify the tumor region and determine its "
                 "histologic grade (1 or 2), tumor area (mm^2), estimated mass (g), and max diameter (mm).")
            if row['has_tumor']:
                a = (f"A tumor is visible. Grade: {int(float(row['grade']))}. "
                     f"Area: {row['tumor_area_mm2']:.2f} mm^2. Mass: {row['tumor_mass_g']:.4f} g. "
                     f"Diameter: {row['tumor_diameter_mm']:.2f} mm.")
            else:
                a = "No tumor is visible. Grade: 0. Area: 0.0 mm^2. Mass: 0.0 g. Diameter: 0.0 mm."

            self.image_paths.append(img_path); self.mask_paths.append(mask_path)
            self.questions.append(q); self.answers.append(a)

    def __len__(self) -> int: return len(self.image_paths)
    def __getitem__(self, idx: int):
        img = self.img_transform(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = (self.mask_transform(Image.open(self.mask_paths[idx]).convert("L")) > 0).float()
        return img, mask, self.questions[idx], self.answers[idx]

def vlm_collate_fn(batch):
    imgs, masks, qs, ans = zip(*batch)
    return list(imgs), torch.stack(masks), list(qs), list(ans)



class LlavaPhysicsMultiTask(nn.Module):
    def __init__(self, llava_model):
        super().__init__()
        self.llava = llava_model
        if isinstance(self.llava, PeftModel):
            self.vision_tower = self.llava.base_model.model.model.vision_tower
        else:
            self.vision_tower = self.llava.model.vision_tower

        self.seg_head = smp.DeepLabV3Plus(
            encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1,
        )
        
        ch = self.seg_head.encoder.out_channels
        self.proj = nn.ModuleDict({
            "p2": nn.Sequential(nn.Conv2d(1024, ch[2], 1), nn.BatchNorm2d(ch[2]), nn.ReLU(inplace=True)),
            "p3": nn.Sequential(nn.Conv2d(1024, ch[3], 1), nn.BatchNorm2d(ch[3]), nn.ReLU(inplace=True)),
            "p4": nn.Sequential(nn.Conv2d(1024, ch[4], 1), nn.BatchNorm2d(ch[4]), nn.ReLU(inplace=True)),
            "p5": nn.Sequential(nn.Conv2d(1024, ch[5], 1), nn.BatchNorm2d(ch[5]), nn.ReLU(inplace=True)),
        })

    def forward(self, input_ids, pixel_values, attention_mask, labels=None, **kwargs):
        vision_out = self.vision_tower(pixel_values, output_hidden_states=True)
        feat = vision_out.hidden_states[-1][:, 1:, :] 
        
        B, N, C = feat.shape
        grid = int(math.sqrt(N))
        feat_2d = feat.reshape(B, grid, grid, C).permute(0, 3, 1, 2).contiguous() 

       
        x5 = F.interpolate(feat_2d, size=(14, 14), mode="bilinear", align_corners=False) 
        x2 = F.interpolate(feat_2d, size=(56, 56), mode="bilinear", align_corners=False) 
        x3 = F.interpolate(feat_2d, size=(42, 42), mode="bilinear", align_corners=False)
        x4 = F.interpolate(feat_2d, size=(28, 28), mode="bilinear", align_corners=False)

        features = [
            None, None, 
            self.proj["p2"](x2), 
            self.proj["p3"](x3), 
            self.proj["p4"](x4), 
            self.proj["p5"](x5) 
        ]
        
        seg_feat = self.seg_head.decoder(features)
        seg_out = self.seg_head.segmentation_head(seg_feat)
        seg_logits = F.interpolate(seg_out, size=(336, 336), mode="bilinear", align_corners=False)

        vqa_out = self.llava(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=labels, return_dict=True)

        return {"vqa_loss": vqa_out.loss, "vqa_logits": vqa_out.logits, "seg_logits": seg_logits.squeeze(1)}



def align_image_tokens(input_ids, attention_mask, labels, img_tok_idx, expected=576):
    new_input_ids, new_attn, new_lbs = input_ids.clone(), attention_mask.clone(), labels.clone() if labels is not None else None
    for i in range(input_ids.shape[0]):
        if (input_ids[i] == img_tok_idx).sum().item() == expected - 1:
            idx = (input_ids[i] == img_tok_idx).nonzero(as_tuple=True)[0][-1]
            new_input_ids[i] = torch.cat([input_ids[i, :idx+1], torch.tensor([img_tok_idx], device=input_ids.device), input_ids[i, idx+1:-1]])
            new_attn[i] = torch.cat([attention_mask[i, :-1], torch.tensor([1], device=DEVICE)])
            if new_lbs is not None:
                new_lbs[i] = torch.cat([labels[i, :idx+1], torch.tensor([-100], device=DEVICE), labels[i, idx+1:-1]])
    return new_input_ids, new_attn, new_lbs

def build_batch(images, masks, questions, answers, processor, img_tok_idx):
    prompts = [f"USER: <image>\n{q}\nASSISTANT:" for q in questions]
    full_texts = [f"USER: <image>\n{q}\nASSISTANT: {a}{processor.tokenizer.eos_token}" for q, a in zip(questions, answers)]
    toks_f = processor(text=full_texts, images=images, return_tensors="pt", padding=True).to(DEVICE)
    toks_p = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(DEVICE)
    lbs = toks_f.input_ids.clone()
    p_lens = torch.sum(toks_p.attention_mask, dim=1)
    for i in range(lbs.size(0)): lbs[i, :p_lens[i]] = -100
    lbs[lbs == processor.tokenizer.pad_token_id] = -100
    ids, attn, lbs = align_image_tokens(toks_f.input_ids, toks_f.attention_mask, lbs, img_tok_idx)
    return {"input_ids": ids, "attention_mask": attn, "labels": lbs, "pixel_values": toks_f.pixel_values.to(dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32), "seg_masks_gt": masks.to(DEVICE)}

def run_evaluation(model, processor, data_loader, description="Evaluating"):
    model.eval()
    physics_eval = PhysicsEvaluator()
    total_iou, total_loss_count = 0.0, 0
    llava_base = model.llava if not isinstance(model.llava, PeftModel) else model.llava.base_model
    img_tok_idx = getattr(llava_base.config, "image_token_index", 32000)

    with torch.no_grad():
        for imgs, masks, qs, ans in tqdm(data_loader, desc=description):
            batch = build_batch(imgs, masks, qs, ans, processor, img_tok_idx)
            with autocast(enabled=(DEVICE.type == "cuda")):
                out = model(**batch)
                gen_ids = model.llava.generate(input_ids=batch["input_ids"], pixel_values=batch["pixel_values"], attention_mask=batch["attention_mask"], max_new_tokens=100)
            
            decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)
            responses = [d.split("ASSISTANT:")[-1].strip() for d in decoded]
            physics_eval.update(ans, responses)

            pred_mask = (torch.sigmoid(out["seg_logits"]) > 0.5).float()
            gt_mask = batch["seg_masks_gt"].squeeze(1)
            inter = (pred_mask * gt_mask).sum(dim=(1, 2))
            union = pred_mask.sum(dim=(1, 2)) + gt_mask.sum(dim=(1, 2)) - inter
            total_iou += ((inter + 1e-6) / (union + 1e-6)).mean().item()
            total_loss_count += 1

    m = physics_eval.compute_metrics()
    avg_iou = total_iou / total_loss_count
    print(f"\n--- {description} Results ---")
    print(f"  QA Grade Acc: {m['Grade_Acc']*100:.2f}% | Seg IoU: {avg_iou:.4f}")
    print(f"  Physics R2 -> Area: {m['Area_R2']:.4f} | Mass: {m['Mass_R2']:.4f} | Diam: {m['Diam_R2']:.4f}")
    return m['Grade_Acc'] + (m['Area_R2'] + m['Mass_R2'] + m['Diam_R2']) / 3.0 + avg_iou



if __name__ == "__main__":
    config = {
        "model_path": "/home/ealam/vlm/Medllava/llava_med_local/",
        "csv_path": "/home/ealam/vlm/mri_dataset/lgg_physics_metadata_v2.csv",
        "base_img_dir": "/home/ealam/vlm/mri_dataset/",
        "save_path": "./vlm113_multi_task_physics",
        "llm_lr": 1e-5, "seg_lr": 5e-4,
        "epochs": 25, "batch_size": 2, "grad_accum": 4, "patience": 5, "seed": 42
    }

    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])

    print(f"Step 1: Initializing Model on {DEVICE}...")
    processor = AutoProcessor.from_pretrained(config["model_path"])
    
  
    base_llava = LlavaForConditionalGeneration.from_pretrained(
        config["model_path"], 
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32, 
        device_map="auto" if DEVICE.type == "cuda" else None
    )
    if DEVICE.type == "cpu":
        base_llava = base_llava.to(DEVICE)
    
    p_size = getattr(base_llava.config.vision_config, "patch_size", 14)
    processor.patch_size = p_size
    if hasattr(processor, "image_processor"): processor.image_processor.patch_size = p_size
    if not hasattr(processor, "vision_feature_select_strategy"): processor.vision_feature_select_strategy = "default"
    if processor.tokenizer.pad_token is None: processor.tokenizer.pad_token = processor.tokenizer.eos_token

    lora_cfg = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], lora_dropout=0.05, task_type="CAUSAL_LM")
    peft_llava = get_peft_model(base_llava, lora_cfg)
    
    model = LlavaPhysicsMultiTask(peft_llava).to(DEVICE)
    
  
    if DEVICE.type == "cuda":
        model.seg_head = model.seg_head.half()
        model.proj = model.proj.half()
    
    for param in model.vision_tower.parameters(): param.requires_grad = False

    llm_params, seg_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if "llava" in name: llm_params.append(param)
        elif "seg_head" in name or "proj" in name: seg_params.append(param)
    optimizer = AdamW([{"params": llm_params, "lr": config["llm_lr"]}, {"params": seg_params, "lr": config["seg_lr"]}])
    
    print("Step 2: Loading Data...")
    df = pd.read_csv(config["csv_path"])
    train_val_df, test_df = train_test_split(df, test_size=0.98, random_state=config["seed"])
    train_df, val_df = train_test_split(train_val_df, test_size=0.20, random_state=config["seed"])

    train_loader = DataLoader(VLM_PhysicsMultiTaskDataset(train_df, config["base_img_dir"]), batch_size=config["batch_size"], shuffle=True, collate_fn=vlm_collate_fn, drop_last=True)
    val_loader = DataLoader(VLM_PhysicsMultiTaskDataset(val_df, config["base_img_dir"]), batch_size=config["batch_size"], collate_fn=vlm_collate_fn)
    test_loader = DataLoader(VLM_PhysicsMultiTaskDataset(test_df, config["base_img_dir"]), batch_size=config["batch_size"], collate_fn=vlm_collate_fn)

    seg_loss_fn = SegLoss().to(DEVICE)
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))
    best_score, patience_counter = -float("inf"), 0
    img_tok_idx = getattr(base_llava.config, "image_token_index", 32000)

    print(f"Step 3: Training Loop ({len(train_df)} samples)...")
    for epoch in range(config["epochs"]):
        model.train()
        for param in model.vision_tower.parameters(): param.requires_grad = False
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, masks, qs, ans in pbar:
            batch = build_batch(imgs, masks, qs, ans, processor, img_tok_idx)
            optimizer.zero_grad()
            with autocast(enabled=(DEVICE.type == "cuda")):
                out = model(**batch)
                loss = out["vqa_loss"] + 2.0 * seg_loss_fn(out["seg_logits"], batch["seg_masks_gt"].squeeze(1))
            
            if DEVICE.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            pbar.set_postfix(loss=loss.item())

        score = run_evaluation(model, processor, val_loader, f"Val Ep {epoch+1}")
        if score > best_score:
            best_score = score; patience_counter = 0
            os.makedirs(config["save_path"], exist_ok=True)
            model.llava.save_pretrained(os.path.join(config["save_path"], "llava_lora"))
            torch.save(model.seg_head.state_dict(), os.path.join(config["save_path"], "seg_head.pth"))
            torch.save(model.proj.state_dict(), os.path.join(config["save_path"], "proj.pth"))
            print(f"Metrics Improved (Best: {best_score:.4f}). Saved Model.")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]: break

    print("\nStep 4: Final Benchmarking on Test Set...")
    if os.path.exists(config["save_path"]):
        final_base = LlavaForConditionalGeneration.from_pretrained(
            config["model_path"], 
            torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32, 
            device_map="auto" if DEVICE.type == "cuda" else None
        )
        final_model_peft = PeftModel.from_pretrained(final_base, os.path.join(config["save_path"], "llava_lora"))
        final_model = LlavaPhysicsMultiTask(final_model_peft).to(DEVICE)
        final_model.seg_head.load_state_dict(torch.load(os.path.join(config["save_path"], "seg_head.pth")))
        final_model.proj.load_state_dict(torch.load(os.path.join(config["save_path"], "proj.pth")))
        
        final_model.eval()
        if DEVICE.type == "cuda": final_model.half()
        run_evaluation(final_model, processor, test_loader, "Final Test Set")

    print("Process complete.")
