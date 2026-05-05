import os
import glob
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
import torchvision.transforms as transforms
from tqdm import tqdm
import segmentation_models_pytorch as smp
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.segmentation import deeplabv3_resnet101
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    get_linear_schedule_with_warmup,
)


warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True


dtype_to_use = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print(f"Using precision: {dtype_to_use}")



def get_guidance_seg_model() -> nn.Module:
    """Loads the ResNet101 model for pre-calculating bounding boxes."""
    model = deeplabv3_resnet101(weights=None, aux_logits=True)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    return model

def get_guidance_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def delineate_and_bbox_from_pred(
    pil_image: Image.Image, 
    seg_model: nn.Module, 
    seg_transform: A.Compose, 
    device: str
) -> Tuple[Optional[Tuple[int, int, int, int]], bool]:
    """Predicts a mask and extracts the bounding box coordinates."""
    img_rgb = np.array(pil_image.convert("RGB"))
    augmented = seg_transform(image=img_rgb)
    image_tensor = augmented['image'].to(device).unsqueeze(0)
    
    seg_model.eval()
    with torch.no_grad():
        output = seg_model(image_tensor)['out']
    
    mask = torch.sigmoid(output).squeeze().cpu().numpy()
    binary_mask = (mask > 0.5).astype(np.uint8)
    

    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    if binary_mask.max() == 0:
        return None, False

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, False

   
    largest_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_cnt)
    
    
    return (int(x), int(y), int(w), int(h)), True



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

def compute_iou_batch(pred_logits: torch.Tensor, true_masks: torch.Tensor, threshold: float = 0.5) -> float:
    with torch.no_grad():
        pred_mask = (torch.sigmoid(pred_logits) > threshold).float()
        true_mask = true_masks.float().squeeze(1) if true_masks.dim() == 4 else true_masks.float()
        intersection = (pred_mask * true_mask).sum(dim=(1, 2))
        union = pred_mask.sum(dim=(1, 2)) + true_mask.sum(dim=(1, 2)) - intersection
        return ((intersection + 1e-6) / (union + 1e-6)).mean().item()

class PhysicsEvaluator:
    def __init__(self): self.reset()
    def reset(self):
        self.true_grades, self.pred_grades = [], []
        self.true_areas, self.pred_areas = [], []
        self.true_masses, self.pred_masses = [], []
        self.true_diams, self.pred_diams = [], []
        self.total_samples = 0

    def parse_text(self, text):
        grade, area, mass, diam = 0, 0.0, 0.0, 0.0
        g_match = re.search(r"grade:\s*(\d)", text, re.IGNORECASE)
        if g_match: grade = int(g_match.group(1))
        def _f(p, t): 
            m = re.search(p, t, re.IGNORECASE)
            try: return float(m.group(1)) if m else 0.0
            except: return 0.0
        area = _f(r"area:\s*([\d\.]+)", text)
        mass = _f(r"mass:\s*([\d\.]+)", text)
        diam = _f(r"diameter:\s*([\d\.]+)", text)
        return grade, area, mass, diam

    def update(self, true_texts, pred_texts):
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



class VLM_QASegDataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, guidance_model: nn.Module, guidance_transform: A.Compose, device: str):
        self.image_paths, self.mask_paths, self.questions, self.answers, self.has_tumors = [], [], [], [], []
        
        self.image_transform = transforms.Resize((336, 336))
        self.mask_transform = transforms.Compose([
            transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        script_dir = os.path.dirname(os.path.abspath(__file__))

        print(f"Building Dataset v204.1 (Textual Guidance): {len(metadata_df)} rows")
        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing metadata"):
            raw_path = str(row['image_path'])
            
            
            if "kaggle_3m/" in raw_path:
                idx = raw_path.find("kaggle_3m/")
                sub_path = raw_path[idx:]
                img_path = os.path.join("/workspace/mri_dataset/", sub_path)
            else:
                img_path = raw_path

            if not os.path.exists(img_path):
                alt_path = os.path.join(script_dir, img_path)
                if os.path.exists(alt_path): img_path = alt_path
                else: continue

            mask_path = img_path.replace(".tif", "_mask.tif")
            if not os.path.exists(mask_path): continue

      
            image_pil = Image.open(img_path).convert("RGB")
            bbox, found_guidance = delineate_and_bbox_from_pred(image_pil, guidance_model, guidance_transform, device)

           
            if bbox is not None:
                x, y, w, h = bbox
                q = (f"Analyze this MRI slice. A suspicious region is detected at box (x:{x}, y:{y}, w:{w}, h:{h}) in 256x256 space. "
                     f"Is a tumor visible? If yes, provide the histologic grade (1 or 2), "
                     f"tumor area (mm^2), estimated mass (g), and max diameter (mm).")
            else:
                q = ("Analyze this MRI slice. Is a tumor visible? "
                     "If yes, provide the histologic grade (1 or 2), "
                     "tumor area (mm^2), estimated mass (g), and max diameter (mm).")

            if row['has_tumor']:
                grade = int(float(row['grade']))
                a = (f"Yes, a tumor is visible. Grade: {grade}. "
                     f"Area: {row['tumor_area_mm2']} mm^2. Mass: {row['tumor_mass_g']} g. "
                     f"Diameter: {row['tumor_diameter_mm']} mm.")
            else:
                a = ("No tumor is visible in this MRI scan. Grade: 0. "
                     "Area: 0.0 mm^2. Mass: 0.0 g. Diameter: 0.0 mm.")

            self.image_paths.append(img_path)
            self.mask_paths.append(mask_path)
            self.questions.append(q)
            self.answers.append(a)
            self.has_tumors.append(bool(row['has_tumor']))

    def __len__(self) -> int: return len(self.image_paths)

    def __getitem__(self, idx: int):
       
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.image_transform(image)
        
        mask_pil = Image.open(self.mask_paths[idx]).convert("L")
        mask_tensor = (self.mask_transform(mask_pil) > 0).float()
        
        return image, mask_tensor, self.questions[idx], self.answers[idx], self.has_tumors[idx]

def vlm_seg_collate_fn(batch):
    images, masks, questions, answers, has_tumors = zip(*batch)
    return list(images), torch.stack(masks, dim=0), list(questions), list(answers), torch.tensor(has_tumors, dtype=torch.bool)

def build_training_batch_cpu_main(images, masks, questions, answers, has_tumors, processor: AutoProcessor):
    prompts_list, full_texts_list = [], []
    for q, a in zip(questions, answers):
        msg_prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        msg_full = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
                    {"role": "assistant", "content": [{"type": "text", "text": a}]}]
        prompts_list.append(processor.apply_chat_template(msg_prompt, tokenize=False, add_generation_prompt=True))
        full_texts_list.append(processor.apply_chat_template(msg_full, tokenize=False, add_generation_prompt=False) + processor.tokenizer.eos_token)

    toks_full = processor(text=full_texts_list, images=images, return_tensors="pt", padding=True)
    labels = toks_full.input_ids.clone()
    toks_prompt = processor(text=prompts_list, images=images, return_tensors="pt", padding=True)
    prompt_lens = torch.sum(toks_prompt.attention_mask, dim=1)
    for i in range(labels.size(0)): labels[i, :prompt_lens[i]] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    batch_cpu = {k: v for k, v in toks_full.items()}
    batch_cpu["labels"] = labels
    batch_cpu["seg_masks_gt"] = masks
    batch_cpu["has_tumor"] = has_tumors
    return batch_cpu

def _to_device(batch_cpu: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch_cpu.items():
        if torch.is_tensor(v): out[k] = v.to(device, non_blocking=True)
        else: out[k] = v
    return out


class QwenVLWithSegmentation(nn.Module):
    def __init__(self, qwen_model: nn.Module, seg_out_size: Tuple[int, int] = (336, 336)):
        super().__init__()
        self.qwen = qwen_model
        self.seg_out_size = seg_out_size
        base = self.qwen.base_model if hasattr(self.qwen, "base_model") else self.qwen
        self.visual = base.model.visual
        vis_hidden = base.config.vision_config.out_hidden_size

        self.deeplab = smp.DeepLabV3(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        for p in self.deeplab.encoder.parameters(): p.requires_grad = False
        
        encoder_out_channels = self.deeplab.encoder.out_channels[-1]
        self.qwen_to_smp = nn.Conv2d(vis_hidden, encoder_out_channels, kernel_size=1)

    def _visual_to_grid(self, pixel_values, grid_thw, batch_size):
        vis_out = self.visual(pixel_values, grid_thw=grid_thw)
        tokens = vis_out if isinstance(vis_out, torch.Tensor) else vis_out.last_hidden_state
        S = tokens.shape[0] // batch_size
        C = tokens.shape[-1]
        H = W = int(math.sqrt(S))
        return tokens.view(batch_size, S, C).transpose(1, 2).contiguous().view(batch_size, C, H, W)

    def forward(self, **batch):
        seg_masks_gt = batch.pop("seg_masks_gt", None)
        has_tumor = batch.pop("has_tumor", None)
        batch_size = batch["input_ids"].size(0)
        pixel_values = batch["pixel_values"]
        image_grid_thw = batch.get("image_grid_thw", None)

        with autocast(enabled=True, dtype=torch.float16):
            vis_feats = self._visual_to_grid(pixel_values, image_grid_thw, batch_size)
            decoder_out = self.deeplab.decoder([self.qwen_to_smp(vis_feats)])
            seg_logits = F.interpolate(self.deeplab.segmentation_head(decoder_out), size=self.seg_out_size, mode="bilinear", align_corners=False).squeeze(1)

        out = self.qwen(**batch, return_dict=True)
        return {"vqa_loss": out.loss, "vqa_logits": out.logits, "seg_logits": seg_logits}


def run_evaluation(model, processor, data_loader, device, description="Evaluating"):
    model.eval()
    physics_eval = PhysicsEvaluator()
    vlm_correct, total_samples = 0, 0
    total_iou, total_count = 0.0, 0

    with torch.no_grad():
        for images, masks_gt, questions, answers, has_tumors in tqdm(data_loader, desc=description):
            B = len(answers)
            prompts = [processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}], tokenize=False, add_generation_prompt=True) for q in questions]
            gen_inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
            generated_ids = model.qwen.generate(**gen_inputs, max_new_tokens=100, pad_token_id=processor.tokenizer.pad_token_id)
            gen_trimmed = [g[len(p):] for g, p in zip(generated_ids, gen_inputs.input_ids)]
            pred_answers = processor.batch_decode(gen_trimmed, skip_special_tokens=True)
            
            physics_eval.update(answers, pred_answers)
            
            for i in range(B):
                pred_span = pred_answers[i].strip().lower()
                true_answer = answers[i].lower()
                if ("no tumor" in true_answer and "no tumor" in pred_span) or ("yes" in pred_span and "yes" in true_answer):
                    vlm_correct += 1
            total_samples += B

            batch_cpu = build_training_batch_cpu_main(images, masks_gt, questions, answers, has_tumors, processor)
            batch_gpu = _to_device(batch_cpu, device)
            with autocast(enabled=True, dtype=torch.float16):
                outputs = model(**batch_gpu)
                total_iou += compute_iou_batch(outputs["seg_logits"], batch_gpu["seg_masks_gt"])
                total_count += 1

    metrics = physics_eval.compute_metrics()
    print(f"\n--- {description} Results ---")
    print(f"  - QA Score: {vlm_correct/max(1, total_samples)*100:.2f}% | IoU: {total_iou/max(1, total_count):.4f}")
    print(f"  - Area R2: {metrics['Area_R2']:.4f} | Mass R2: {metrics['Mass_R2']:.4f} | Diam R2: {metrics['Diam_R2']:.4f}")
    return vlm_correct/max(1, total_samples), total_iou/max(1, total_count)



if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:3")
        print(f"Using device: {DEVICE}")
    else:
        DEVICE = torch.device("cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = {
        "device": str(DEVICE),
        "csv_path": "/workspace/qwen/lgg_physics_metadata_v2.csv",
        "guidance_model_path": "/workspace/best_model_segmentation_v2.pth",
        "local_qwen_path": "/workspace/qwen/saved_model",
        "save_path": "/workspace/qwen/qwen_vlm204_1_text_guidance",
        "lr": 1e-4, "batch_size": 2, "epochs": 25, "patience": 5, "seed": 42,
        "seg_weight": 3.0, "grad_accum": 4
    }

    torch.manual_seed(config["seed"]); np.random.seed(config["seed"]); random.seed(config["seed"])

    print("Step 1: Loading Guidance Model for Pre-calculation...")
    guidance_model = get_guidance_seg_model()
    guidance_model.load_state_dict(torch.load(config["guidance_model_path"], map_location=DEVICE), strict=False)
    guidance_model.to(DEVICE).eval()
    guidance_transforms = get_guidance_transforms()

    print("Step 2: Loading VLM...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config["local_qwen_path"], torch_dtype=torch.float16, low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(config["local_qwen_path"])
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        base_model.resize_token_embeddings(len(processor.tokenizer))


    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_model = get_peft_model(base_model, LoraConfig(r=32, lora_alpha=64, target_modules=target_modules, task_type="CAUSAL_LM")).to(DEVICE)
    multitask_model = QwenVLWithSegmentation(peft_model).to(DEVICE)

    print("Step 3: Preparing DataLoaders with Textual BBoxes...")
    df = pd.read_csv(config["csv_path"])
    train_val_df, test_df = train_test_split(df, test_size=0.20, random_state=config["seed"])
    train_df, val_df = train_test_split(train_val_df, test_size=0.20, random_state=config["seed"])

    train_ds = VLM_QASegDataset(train_df, guidance_model, guidance_transforms, str(DEVICE))
    val_ds = VLM_QASegDataset(val_df, guidance_model, guidance_transforms, str(DEVICE))
    test_ds = VLM_QASegDataset(test_df, guidance_model, guidance_transforms, str(DEVICE))

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=vlm_seg_collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], collate_fn=vlm_seg_collate_fn)

    optimizer = AdamW(multitask_model.parameters(), lr=config["lr"])
    scaler = GradScaler()
    seg_loss_iou_fn = JaccardLoss(reduction="none").to(DEVICE)
    seg_loss_bce_fn = nn.BCEWithLogitsLoss(reduction="none").to(DEVICE)

    best_metric = 0.0
    print("Step 4: Training v204.1 (Text Guidance)...")
    for epoch in range(config["epochs"]):
        multitask_model.train()
        for step, (imgs, masks, qs, ans, has_t) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            batch_gpu = _to_device(build_training_batch_cpu_main(imgs, masks, qs, ans, has_t, processor), DEVICE)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=True, dtype=torch.float16):
                outputs = multitask_model(**batch_gpu)
                v_loss = outputs["vqa_loss"]
                gt = batch_gpu["seg_masks_gt"].squeeze(1)
                
                iou_l = seg_loss_iou_fn(outputs["seg_logits"], gt)
                bce_l = seg_loss_bce_fn(outputs["seg_logits"], gt).mean(dim=(1, 2))
                seg_loss_per_sample = (0.5 * bce_l) + (0.5 * iou_l)
                
                weights = torch.ones_like(seg_loss_per_sample)
                weights[batch_gpu["has_tumor"]] = 1.0 
                combined_loss = (v_loss + config["seg_weight"] * (seg_loss_per_sample * weights).mean()) / config["grad_accum"]

            scaler.scale(combined_loss).backward()
            if (step + 1) % config["grad_accum"] == 0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

        val_acc, val_iou = run_evaluation(multitask_model, processor, val_loader, DEVICE, "Validation")
        if (val_acc + val_iou) > best_metric:
            best_metric = val_acc + val_iou
            os.makedirs(config["save_path"], exist_ok=True)
            torch.save(multitask_model.state_dict(), os.path.join(config["save_path"], "multitask_model.pth"))
            multitask_model.qwen.save_pretrained(os.path.join(config["save_path"], "qwen_lora"))

    print("Training v204.1 Complete.")
