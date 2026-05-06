import os
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
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
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
)

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    dtype_to_use = torch.bfloat16
    use_scaler = False 
else:
    dtype_to_use = torch.float16
    use_scaler = True

print(f"Environment Initialized: Using {DEVICE} with {dtype_to_use} precision.")

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

def post_process_mask(mask: np.ndarray, kernel_size: int = 5, min_area: int = 100) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
    processed_mask = np.zeros(mask.shape, dtype=np.uint8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        if stats[largest_label, cv2.CC_STAT_AREA] > min_area:
            processed_mask[labels == largest_label] = 255
    return processed_mask

def draw_yellow_lines(pil_image: Image.Image, seg_model: nn.Module, seg_transform: A.Compose, device: str) -> Image.Image:
    open_cv_image = np.array(pil_image.convert("RGB"))
    augmented = seg_transform(image=open_cv_image)
    image_tensor = augmented['image'].to(device).unsqueeze(0)

    seg_model.eval()
    with torch.no_grad():
        output = seg_model(image_tensor)['out']

    mask = torch.sigmoid(output).squeeze().cpu().numpy()
    binary_mask = (mask > 0.5).astype(np.uint8)
    cleaned_mask = post_process_mask(binary_mask)

    if cleaned_mask.shape[:2] != open_cv_image.shape[:2]:
        cleaned_mask = cv2.resize(cleaned_mask, (open_cv_image.shape[1], open_cv_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(open_cv_image, contours, -1, (0, 255, 255), 2) 

    return Image.fromarray(open_cv_image)

class CombinedSegLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred_probs = torch.sigmoid(y_pred)
        y_pred_flat = y_pred_probs.view(y_pred.shape[0], -1)
        y_true_flat = y_true.view(y_true.shape[0], -1).float()
        intersection = (y_pred_flat * y_true_flat).sum(1)
        union = (y_pred_flat + y_true_flat).sum(1) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 0.5 * (1.0 - iou.mean()) + 0.5 * self.bce(y_pred, y_true)

def convert_bn_to_gn(module, num_groups=32):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            groups = min(num_groups, num_channels)
            while num_channels % groups != 0: groups -= 1
            setattr(module, name, nn.GroupNorm(groups, num_channels))
        else:
            convert_bn_to_gn(child, num_groups)

class ClassificationEvaluator:
    def __init__(self): self.reset()
    def reset(self):
        self.true_grades, self.pred_grades = [], []
        
    def parse_grade(self, text):
        text = text.lower()
        g_match = re.search(r"grade:\s*(\d)", text)
        if g_match:
            return int(g_match.group(1))
        if "no tumor" in text or "grade: 0" in text:
            return 0
        return 0

    def update(self, t_texts, p_texts):
        for t, p in zip(t_texts, p_texts):
            self.true_grades.append(self.parse_grade(t))
            self.pred_grades.append(self.parse_grade(p))

    def compute_metrics(self):
        return {
            "Accuracy": accuracy_score(self.true_grades, self.pred_grades),
            "F1_Weighted": f1_score(self.true_grades, self.pred_grades, average='weighted')
        }

class VLM_DelineatedDataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, base_img_dir: str, ext_model: nn.Module, ext_transform: A.Compose, device: str):
        self.image_paths, self.mask_paths, self.questions, self.answers = [], [], [], []
        self.ext_model = ext_model
        self.ext_transform = ext_transform
        self.device = device
        self.vlm_resize = transforms.Compose([transforms.Resize((336, 336))])
        self.mask_transform = transforms.Compose([
            transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Indexing Dataset"):
            raw_path = str(row['image_path'])
            img_path = raw_path if os.path.exists(raw_path) else os.path.join(base_img_dir, "kaggle_3m", raw_path.split("kaggle_3m/")[-1])
            mask_path = img_path.replace(".tif", "_mask.tif")
            if not os.path.exists(img_path): continue

            q = ("Analyze this MRI slice where the tumor is delineated by a yellow contour. "
                 "Identify if a tumor is visible and state the histologic grade (0, 1, or 2).")

            if row['has_tumor']:
                grade = int(row.get('grade', 1))
                a = f"A tumor is visible in the delineated area. Grade: {grade}."
            else:
                a = "No tumor is visible. Grade: 0."

            self.image_paths.append(img_path); self.mask_paths.append(mask_path)
            self.questions.append(q); self.answers.append(a)

    def __len__(self) -> int: return len(self.image_paths)
    def __getitem__(self, idx: int):
        raw_img = Image.open(self.image_paths[idx]).convert("RGB")
        delineated_img = draw_yellow_lines(raw_img, self.ext_model, self.ext_transform, self.device)
        vlm_img = self.vlm_resize(delineated_img)
        
        mask_path = self.mask_paths[idx]
        if os.path.exists(mask_path):
            mask = (self.mask_transform(Image.open(mask_path).convert("L")) > 0).float()
        else:
            mask = torch.zeros((1, 336, 336), dtype=torch.float32)
        return vlm_img, mask, self.questions[idx], self.answers[idx]

def multitask_collate_fn(batch):
    imgs, masks, qs, ans = zip(*batch)
    return list(imgs), torch.stack(masks), list(qs), list(ans)

class LlavaMultiTaskClassification(nn.Module):
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
        x4 = F.interpolate(feat_2d, size=(28, 28), mode="bilinear", align_corners=False)
        x3 = F.interpolate(feat_2d, size=(42, 42), mode="bilinear", align_corners=False)
        x2 = F.interpolate(feat_2d, size=(56, 56), mode="bilinear", align_corners=False) 
        
        features = [None, None, self.proj["p2"](x2), self.proj["p3"](x3), self.proj["p4"](x4), self.proj["p5"](x5)]
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

def run_evaluation(model, processor, data_loader, description="Evaluating"):
    model.eval()
    clf_eval = ClassificationEvaluator()
    total_iou, count = 0.0, 0
    llava_base = model.llava if not isinstance(model.llava, PeftModel) else model.llava.base_model
    img_tok_idx = getattr(llava_base.config, "image_token_index", 32000)

    with torch.no_grad():
        for imgs, masks, qs, ans in tqdm(data_loader, desc=description):
            inf_batch = build_batch(imgs, masks, qs, ans, processor, img_tok_idx, inference_only=True)
            full_batch = build_batch(imgs, masks, qs, ans, processor, img_tok_idx, inference_only=False)

            with autocast(enabled=(DEVICE.type == "cuda"), dtype=dtype_to_use):
                out = model(**full_batch)
                gen_ids = model.llava.generate(
                    input_ids=inf_batch["input_ids"], 
                    pixel_values=inf_batch["pixel_values"], 
                    attention_mask=inf_batch["attention_mask"], 
                    max_new_tokens=40
                )
            
            prompt_len = inf_batch["input_ids"].shape[1]
            gen_ids = gen_ids[:, prompt_len:]
            decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)
            clf_eval.update(ans, decoded)

            pred_mask = (torch.sigmoid(out["seg_logits"]) > 0.5).float()
            gt_mask = full_batch["seg_masks_gt"].squeeze(1)
            inter = (pred_mask * gt_mask).sum(dim=(1, 2))
            union = pred_mask.sum(dim=(1, 2)) + gt_mask.sum(dim=(1, 2)) - inter
            total_iou += ((inter + 1e-6) / (union + 1e-6)).mean().item()
            count += 1

    m = clf_eval.compute_metrics()
    avg_iou = total_iou / count
    print(f"\n--- {description} ---")
    print(f"  Accuracy: {m['Accuracy']*100:.2f}% | F1: {m['F1_Weighted']:.4f} | IoU: {avg_iou:.4f}")
    return m['Accuracy'] + avg_iou

if __name__ == "__main__":
    config = {
        "model_path": "./llava_med_local",
        "csv_path": "/home/ealam/Downloads/LGG dataset Cameron/lgg_physics_metadata_v2.csv",
        "base_img_dir": "/home/ealam/Downloads/LGG dataset Cameron/",
        "ext_seg_path": "/home/ealam/myenv/best_model_segmentation_v2.pth", 
        "save_path": "/home/ealam/Desktop/vlm_clf_seg_output",
        "llm_lr": 2e-5, "seg_lr": 1e-4,
        "epochs": 15, "batch_size": 2, "grad_accum": 4, "patience": 3, "seed": 42
    }

    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])

    ext_seg_model = get_external_segmentation_model()
    ext_seg_model.load_state_dict(torch.load(config["ext_seg_path"], map_location=DEVICE), strict=False)
    ext_seg_model.to(DEVICE).eval()
    ext_seg_transform = get_external_seg_transforms()

    processor = AutoProcessor.from_pretrained(config["model_path"])
    base_llava = LlavaForConditionalGeneration.from_pretrained(config["model_path"], torch_dtype=dtype_to_use, device_map={"": DEVICE})
    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, task_type="CAUSAL_LM")
    peft_llava = get_peft_model(base_llava, lora_cfg)
    
    model = LlavaMultiTaskClassification(peft_llava).to(DEVICE)
    model.seg_head = model.seg_head.to(dtype=dtype_to_use)
    model.proj = model.proj.to(dtype=dtype_to_use)

    for param in model.vision_tower.parameters(): param.requires_grad = False

    llm_params = [p for n, p in model.named_parameters() if p.requires_grad and "llava" in n]
    seg_params = [p for n, p in model.named_parameters() if p.requires_grad and ("seg_head" in n or "proj" in n)]
    optimizer = AdamW([{"params": llm_params, "lr": config["llm_lr"]}, {"params": seg_params, "lr": config["seg_lr"]}])

    df = pd.read_csv(config["csv_path"])
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=config["seed"])
    train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=config["seed"])

    train_ds = VLM_DelineatedDataset(train_df, config["base_img_dir"], ext_seg_model, ext_seg_transform, DEVICE)
    val_ds = VLM_DelineatedDataset(val_df, config["base_img_dir"], ext_seg_model, ext_seg_transform, DEVICE)
    test_ds = VLM_DelineatedDataset(test_df, config["base_img_dir"], ext_seg_model, ext_seg_transform, DEVICE)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=multitask_collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], collate_fn=multitask_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], collate_fn=multitask_collate_fn)

    seg_loss_fn = CombinedSegLoss().to(DEVICE)
    scaler = GradScaler(enabled=use_scaler)
    best_score, patience_counter, img_tok_idx = -float("inf"), 0, getattr(base_llava.config, "image_token_index", 32000)

    print("\nTraining Started (Classification + Segmentation)...")
    for epoch in range(config["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()
        
        for step, (imgs, masks, qs, ans) in enumerate(pbar):
            batch = build_batch(imgs, masks, qs, ans, processor, img_tok_idx, inference_only=False)
            with autocast(enabled=(DEVICE.type == "cuda"), dtype=dtype_to_use):
                out = model(**batch)
                vqa_loss = out["vqa_loss"]
                seg_loss = seg_loss_fn(out["seg_logits"], batch["seg_masks_gt"].squeeze(1))
                loss = (vqa_loss + 2.0 * seg_loss) / config["grad_accum"]
            
            if use_scaler:
                scaler.scale(loss).backward()
                if (step + 1) % config["grad_accum"] == 0:
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % config["grad_accum"] == 0:
                    optimizer.step(); optimizer.zero_grad()
            
            pbar.set_postfix(loss=loss.item() * config["grad_accum"])

        score = run_evaluation(model, processor, val_loader, f"Val Ep {epoch+1}")
        if score > best_score:
            best_score = score; patience_counter = 0
            os.makedirs(config["save_path"], exist_ok=True)
            model.llava.save_pretrained(os.path.join(config["save_path"], "llava_lora"))
            torch.save(model.seg_head.state_dict(), os.path.join(config["save_path"], "seg_head.pth"))
            print("Model Improved and Saved.")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]: break

    print("\nFinal Benchmarking...")
    run_evaluation(model, processor, test_loader, "Test Set Results")
