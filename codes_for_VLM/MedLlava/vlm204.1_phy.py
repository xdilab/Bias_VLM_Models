import os
import math
import re
import random
import logging
import warnings
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from PIL import Image
import cv2
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.segmentation import deeplabv3_resnet101
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    get_linear_schedule_with_warmup,
)


warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True


DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    dtype_to_use = torch.bfloat16
    use_scaler = False 
else:
    dtype_to_use = torch.float16
    use_scaler = True

print(f"Environment Initialized: Using {DEVICE} with {dtype_to_use} precision (Weights: float32).")



def get_external_segmentation_model() -> nn.Module:
    model = deeplabv3_resnet101(weights=None)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    return model

def get_external_seg_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def post_process_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    processed = np.zeros_like(mask)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        processed[labels == largest_label] = 255
    return processed

def get_predicted_bbox(pil_img: Image.Image, model: nn.Module, transform: A.Compose, device: str) -> Optional[Tuple[int, int, int, int]]:
    cv_img = np.array(pil_img.convert("RGB"))
    tensor = transform(image=cv_img)['image'].to(device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        out = model(tensor)['out']
    mask = (torch.sigmoid(out).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    mask = post_process_mask(mask)
    if mask.max() == 0: return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    return (int(x), int(y), int(w), int(h))



class CombinedSegLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_probs = torch.sigmoid(y_pred)
        y_flat = y_probs.view(y_pred.shape[0], -1)
        t_flat = y_true.view(y_true.shape[0], -1).float()
        inter = (y_flat * t_flat).sum(1)
        union = (y_flat + t_flat).sum(1) - inter
        iou = (inter + self.smooth) / (union + self.smooth)
        return 0.5 * (1.0 - iou.mean()) + 0.5 * self.bce(y_pred, y_true)

def convert_bn_to_gn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            ch = child.num_features
            gr = 32 if ch % 32 == 0 else (16 if ch % 16 == 0 else 8)
            setattr(module, name, nn.GroupNorm(gr, ch))
        else: convert_bn_to_gn(child)

class PhysicsEvaluator:
    def __init__(self): self.reset()
    def reset(self):
        self.true_grades, self.pred_grades = [], []
        self.true_areas, self.pred_areas = [], []
        self.true_masses, self.pred_masses = [], []
        self.true_diams, self.pred_diams = [], []
        
    def parse_text(self, text):
        grade, area, mass, diam = 0, 0.0, 0.0, 0.0
        text = text.lower().replace("one", "1").replace("two", "2")
        g_match = re.search(r"grade:\s*(\d)", text)
        if g_match: grade = int(g_match.group(1))
        def _f(p):
            m = re.search(p, text)
            try: return float(m.group(1)) if m else 0.0
            except: return 0.0
        area, mass, diam = _f(r"area:\s*([\d\.]+)"), _f(r"mass:\s*([\d\.]+)"), _f(r"diameter:\s*([\d\.]+)")
        return grade, area, mass, diam

    def update(self, t_texts, p_texts):
        for t, p in zip(t_texts, p_texts):
            tg, ta, tm, td = self.parse_text(t)
            pg, pa, pm, pd = self.parse_text(p)
            self.true_grades.append(tg); self.pred_grades.append(pg)
            self.true_areas.append(ta); self.pred_areas.append(pa)
            self.true_masses.append(tm); self.pred_masses.append(pm)
            self.true_diams.append(td); self.pred_diams.append(pd)

    def compute_metrics(self):
        m = {"Grade_Acc": accuracy_score(self.true_grades, self.pred_grades) if self.true_grades else 0.0}
        if len(self.true_areas) > 1:
            m["Area_R2"], m["Mass_R2"], m["Diam_R2"] = r2_score(self.true_areas, self.pred_areas), r2_score(self.true_masses, self.pred_masses), r2_score(self.true_diams, self.pred_diams)
        else:
            for k in ["Area_R2", "Mass_R2", "Diam_R2"]: m[k] = 0.0
        return m



class VLM_PhysicsTextualBBoxDataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, base_img_dir: str, ext_model: nn.Module, ext_transform: A.Compose, device: str):
        self.image_paths, self.mask_paths, self.questions, self.answers = [], [], [], []
        self.vlm_resize = transforms.Compose([transforms.Resize((336, 336))])
        self.mask_transform = transforms.Compose([
            transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Pre-calculating BBoxes"):
            raw_path = str(row['image_path'])
            img_path = raw_path if os.path.exists(raw_path) else os.path.join(base_img_dir, "kaggle_3m", raw_path.split("kaggle_3m/")[-1])
            if not os.path.exists(img_path): continue

            raw_img = Image.open(img_path).convert("RGB")
            bbox = get_predicted_bbox(raw_img, ext_model, ext_transform, device)
            
            if bbox:
                x, y, w, h = bbox
                q = (f"Analyze this MRI slice. A potential lesion was identified at bounding box "
                     f"(x:{x}, y:{y}, w:{w}, h:{h}) in 256x256 space. Is a tumor truly visible? "
                     f"If yes, provide the histologic grade (1 or 2), tumor area (mm^2), estimated mass (g), and max diameter (mm).")
            else:
                q = ("Analyze this MRI slice. Is a tumor visible? If yes, provide the histologic grade (1 or 2), "
                     "tumor area (mm^2), estimated mass (g), and max diameter (mm).")

            if row['has_tumor']:
                a = (f"Yes, a tumor is visible. Grade: {int(row['grade'])}. Area: {row['tumor_area_mm2']} mm^2. "
                     f"Mass: {row['tumor_mass_g']} g. Diameter: {row['tumor_diameter_mm']} mm.")
            else:
                a = "No tumor is visible in this MRI scan. Grade: 0. Area: 0.0 mm^2. Mass: 0.0 g. Diameter: 0.0 mm."

            self.image_paths.append(img_path)
            self.mask_paths.append(img_path.replace(".tif", "_mask.tif"))
            self.questions.append(q); self.answers.append(a)

    def __len__(self) -> int: return len(self.image_paths)
    def __getitem__(self, idx: int):
        vlm_img = self.vlm_resize(Image.open(self.image_paths[idx]).convert("RGB"))
        mask_path = self.mask_paths[idx]
        if os.path.exists(mask_path):
            mask = (self.mask_transform(Image.open(mask_path).convert("L")) > 0).float()
        else:
            mask = torch.zeros((1, 336, 336))
        return vlm_img, mask, self.questions[idx], self.answers[idx]

def multitask_collate_fn(batch):
    imgs, masks, qs, ans = zip(*batch)
    return list(imgs), torch.stack(masks), list(qs), list(ans)



class LlavaPhysicsMultiTask(nn.Module):
    def __init__(self, llava_model):
        super().__init__()
        self.llava = llava_model
        self.vision_tower = self.llava.base_model.model.model.vision_tower if isinstance(self.llava, PeftModel) else self.llava.model.vision_tower
        self.seg_head = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        convert_bn_to_gn(self.seg_head.decoder)
        convert_bn_to_gn(self.seg_head.segmentation_head)
        ch = self.seg_head.encoder.out_channels
        self.proj = nn.ModuleDict({
            "p2": nn.Sequential(nn.Conv2d(1024, ch[2], 1), nn.GroupNorm(ch[2], ch[2]), nn.ReLU(inplace=True)),
            "p3": nn.Sequential(nn.Conv2d(1024, ch[3], 1), nn.GroupNorm(ch[3], ch[3]), nn.ReLU(inplace=True)),
            "p4": nn.Sequential(nn.Conv2d(1024, ch[4], 1), nn.GroupNorm(ch[4], ch[4]), nn.ReLU(inplace=True)),
            "p5": nn.Sequential(nn.Conv2d(1024, ch[5], 1), nn.GroupNorm(ch[5], ch[5]), nn.ReLU(inplace=True)),
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
        features = [None, None, self.proj["p2"](x2), self.proj["p3"](x3), self.proj["p4"](x4), self.proj["p5"](x5)]
        seg_feat = self.seg_head.decoder(features)
        seg_out = self.seg_head.segmentation_head(seg_feat)
        seg_logits = F.interpolate(seg_out, size=(336, 336), mode="bilinear", align_corners=False)
        vqa_out = self.llava(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=labels, return_dict=True)
        return {"vqa_loss": vqa_out.loss, "vqa_logits": vqa_out.logits, "seg_logits": seg_logits.squeeze(1)}



def run_evaluation(model, processor, data_loader, description="Evaluating"):
    model.eval()
    physics_eval = PhysicsEvaluator()
    total_iou, count = 0.0, 0
    img_tok_idx = getattr(model.llava.config, "image_token_index", 32000)
    with torch.no_grad():
        for imgs, masks, qs, ans in tqdm(data_loader, desc=description):
            inf_batch = build_batch(imgs, masks, qs, ans, processor, img_tok_idx, inference_only=True)
            full_batch = build_batch(imgs, masks, qs, ans, processor, img_tok_idx, inference_only=False)
            with autocast(enabled=(DEVICE.type=="cuda"), dtype=dtype_to_use):
                out = model(**full_batch)
                gen_ids = model.llava.generate(input_ids=inf_batch["input_ids"], pixel_values=inf_batch["pixel_values"], attention_mask=inf_batch["attention_mask"], max_new_tokens=100)
            
            gen_ids = gen_ids[:, inf_batch["input_ids"].shape[1]:]
            physics_eval.update(ans, processor.batch_decode(gen_ids, skip_special_tokens=True))
            
            
            pred_mask = (torch.sigmoid(out["seg_logits"]) > 0.5).float()
            gt_mask = full_batch["seg_masks_gt"].squeeze(1).to(DEVICE)
            intersection = (pred_mask * gt_mask).sum(dim=(1, 2))
            union = pred_mask.sum(dim=(1, 2)) + gt_mask.sum(dim=(1, 2)) - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            total_iou += iou.mean().item()
            count += 1

    m = physics_eval.compute_metrics()
    avg_iou = total_iou / count
    print(f"\n--- {description} --- \n  Acc: {m['Grade_Acc']*100:.2f}% | Seg IoU: {avg_iou:.4f}")
    print(f"  Physics R2 -> Area: {m['Area_R2']:.4f} | Mass: {m['Mass_R2']:.4f} | Diam: {m['Diam_R2']:.4f}")
    return m['Grade_Acc'] + (m['Area_R2'] + m['Mass_R2'] + m['Diam_R2']) / 3.0 + avg_iou




def align_image_tokens(input_ids, attention_mask, labels, img_tok_idx, expected=576):
    new_ids, new_attn, new_lbs = input_ids.clone(), attention_mask.clone(), labels.clone() if labels is not None else None
    for i in range(input_ids.shape[0]):
        if (input_ids[i] == img_tok_idx).sum().item() == expected - 1:
            idx = (input_ids[i] == img_tok_idx).nonzero(as_tuple=True)[0][-1]
            new_ids[i] = torch.cat([input_ids[i, :idx+1], torch.tensor([img_tok_idx], device=input_ids.device), input_ids[i, idx+1:-1]])
            new_attn[i] = torch.cat([attention_mask[i, :-1], torch.tensor([1], device=DEVICE)])
            if new_lbs is not None: new_lbs[i] = torch.cat([labels[i, :idx+1], torch.tensor([-100], device=DEVICE), labels[i, idx+1:-1]])
    return new_ids, new_attn, new_lbs

def build_batch(images, masks, questions, answers, processor, img_tok_idx, inference_only=False):
    prompts = [f"USER: <image>\n{q}\nASSISTANT:" for q in questions]
    if inference_only:
        toks = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(DEVICE)
        ids, attn, _ = align_image_tokens(toks.input_ids, toks.attention_mask, None, img_tok_idx)
        return {"input_ids": ids, "attention_mask": attn, "pixel_values": toks.pixel_values.to(dtype=dtype_to_use), "seg_masks_gt": masks.to(DEVICE)}
    else:
        full_texts = [f"USER: <image>\n{q}\nASSISTANT: {a}{processor.tokenizer.eos_token}" for q, a in zip(questions, answers)]
        toks_f = processor(text=full_texts, images=images, return_tensors="pt", padding=True).to(DEVICE)
        toks_p = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(DEVICE)
        lbs = toks_f.input_ids.clone()
        p_lens = torch.sum(toks_p.attention_mask, dim=1)
        for i in range(lbs.size(0)): lbs[i, :p_lens[i]] = -100
        lbs[lbs == processor.tokenizer.pad_token_id] = -100
        ids, attn, lbs = align_image_tokens(toks_f.input_ids, toks_f.attention_mask, lbs, img_tok_idx)
        return {"input_ids": ids, "attention_mask": attn, "labels": lbs, "pixel_values": toks_f.pixel_values.to(dtype=dtype_to_use), "seg_masks_gt": masks.to(DEVICE)}

if __name__ == "__main__":
    config = {
        "model_path": "./llava_med_local",
        "csv_path": "/home/ealam/Downloads/LGG dataset Cameron/lgg_physics_metadata_v2.csv",
        "base_img_dir": "/home/ealam/Downloads/LGG dataset Cameron/",
        "ext_seg_path": "/home/ealam/myenv/best_model_segmentation_v2.pth",
        "save_path": "/home/ealam/Desktop/vlm204_1_textual_bbox",
        "llm_lr": 2e-5, "seg_lr": 1e-4, "epochs": 25, "batch_size": 2, "grad_accum": 4, "seed": 42
    }
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])

    ext_seg = get_external_segmentation_model()
    ext_seg.load_state_dict(torch.load(config["ext_seg_path"], map_location=DEVICE), strict=False)
    ext_seg.to(DEVICE).eval()
    ext_transform = get_external_seg_transforms()

    processor = AutoProcessor.from_pretrained(config["model_path"])
    base_llava = LlavaForConditionalGeneration.from_pretrained(config["model_path"], torch_dtype=dtype_to_use, device_map={"": DEVICE})
    processor.patch_size = getattr(base_llava.config.vision_config, "patch_size", 14)
    if processor.tokenizer.pad_token is None: processor.tokenizer.pad_token = processor.tokenizer.eos_token

    lora_cfg = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0.05, task_type="CAUSAL_LM")
    model = LlavaPhysicsMultiTask(get_peft_model(base_llava, lora_cfg)).to(DEVICE)
    
   
    
    for param in model.vision_tower.parameters(): param.requires_grad = False

    llm_p = []
    seg_p = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if "llava" in name: llm_p.append(param)
        elif "seg_head" in name or "proj" in name: seg_p.append(param)
        else: llm_p.append(param)

    optimizer = AdamW([{"params": llm_p, "lr": config["llm_lr"]}, {"params": seg_p, "lr": config["seg_lr"]}])

    df = pd.read_csv(config["csv_path"])
    tr_val, test_df = train_test_split(df, test_size=0.20, random_state=config["seed"])
    train_df, val_df = train_test_split(tr_val, test_size=0.20, random_state=config["seed"])

    train_ds = VLM_PhysicsTextualBBoxDataset(train_df, config["base_img_dir"], ext_seg, ext_transform, DEVICE)
    val_ds = VLM_PhysicsTextualBBoxDataset(val_df, config["base_img_dir"], ext_seg, ext_transform, DEVICE)
    test_ds = VLM_PhysicsTextualBBoxDataset(test_df, config["base_img_dir"], ext_seg, ext_transform, DEVICE)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=multitask_collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], collate_fn=multitask_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], collate_fn=multitask_collate_fn)

    best_score, counter, img_tok_idx = -float("inf"), 0, getattr(base_llava.config, "image_token_index", 32000)
    seg_loss_fn = CombinedSegLoss().to(DEVICE); scaler = GradScaler(enabled=use_scaler)

    for epoch in range(config["epochs"]):
        model.train(); optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for step, (imgs, masks, qs, ans) in enumerate(pbar):
            batch = build_batch(imgs, masks, qs, ans, processor, img_tok_idx, inference_only=False)
            with autocast(enabled=(DEVICE.type=="cuda"), dtype=dtype_to_use):
                out = model(**batch)
              
                loss = (out["vqa_loss"] + 5.0 * seg_loss_fn(out["seg_logits"], batch["seg_masks_gt"].squeeze(1))) / config["grad_accum"]
            
            if use_scaler:
                scaler.scale(loss).backward()
                if (step+1)%config["grad_accum"]==0: scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            else:
                loss.backward()
                if (step+1)%config["grad_accum"]==0: optimizer.step(); optimizer.zero_grad()
            pbar.set_postfix(loss=loss.item()*config["grad_accum"])

        score = run_evaluation(model, processor, val_loader, f"Val Ep {epoch+1}")
        if score > best_score:
            best_score = score; counter = 0
            os.makedirs(config["save_path"], exist_ok=True)
            model.llava.save_pretrained(os.path.join(config["save_path"], "llava_lora"))
            torch.save(model.seg_head.state_dict(), os.path.join(config["save_path"], "seg_head.pth"))
            torch.save(model.proj.state_dict(), os.path.join(config["save_path"], "proj.pth"))
        else:
            counter += 1
            if counter >= 5: break

    run_evaluation(model, processor, test_loader, "Final Test Set")
