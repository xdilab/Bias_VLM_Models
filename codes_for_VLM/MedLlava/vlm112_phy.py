import os
import glob
import math
import re
import random
import logging
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
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


if not torch.cuda.is_available():
    print("CRITICAL ERROR: CUDA is not available. This script requires a GPU.")
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda:1")
    print(f"Local Environment Initialized: Using {DEVICE} ({torch.cuda.get_device_name(0)})")



def get_segmentation_model() -> nn.Module:
    
    model = deeplabv3_resnet101(weights=None)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    return model

def get_segmentation_transforms() -> A.Compose:
  
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
    return processed_mask.astype(np.uint8)

def delineate_roi_on_image(pil_image: Image.Image, seg_model: nn.Module, seg_transform: A.Compose, device: str) -> Tuple[Image.Image, bool]:
    
    open_cv_image = np.array(pil_image.convert("RGB"))
    augmented = seg_transform(image=open_cv_image)
    image_tensor = augmented['image'].to(device).unsqueeze(0)

    seg_model.eval()
    with torch.no_grad():
        output = seg_model(image_tensor)['out']

    mask = torch.sigmoid(output).squeeze().cpu().numpy()
    binary_mask = (mask > 0.5).astype(np.uint8)
    cleaned_mask = post_process_mask(binary_mask)
    
    
    cleaned_mask_resized = cv2.resize(cleaned_mask, (open_cv_image.shape[1], open_cv_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(cleaned_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    has_tumor = bool(contours)
    if has_tumor:
        cv2.drawContours(open_cv_image, contours, -1, (0, 255, 255), 2) # Yellow contour

    return Image.fromarray(open_cv_image), has_tumor



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
            
            self.true_grades.append(t_grade)
            self.pred_grades.append(p_grade)
            self.true_areas.append(t_area)
            self.pred_areas.append(p_area)
            self.true_masses.append(t_mass)
            self.pred_masses.append(p_mass)
            self.true_diams.append(t_diam)
            self.pred_diams.append(p_diam)

    def compute_metrics(self):
        metrics = {}
        metrics["Grade_Acc"] = accuracy_score(self.true_grades, self.pred_grades) if self.true_grades else 0.0
        if len(self.true_areas) > 1:
            metrics["Area_R2"] = r2_score(self.true_areas, self.pred_areas)
            metrics["Mass_R2"] = r2_score(self.true_masses, self.pred_masses)
            metrics["Diam_R2"] = r2_score(self.true_diams, self.pred_diams)
        else:
            for k in ["Area_R2", "Mass_R2", "Diam_R2"]: metrics[k] = 0.0
        return metrics



class VLM_PhysicsDelineatedDataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, base_img_dir: str, seg_model: nn.Module, seg_transform: A.Compose, device: str):
        self.image_paths, self.questions, self.answers = [], [], []
        self.seg_model = seg_model
        self.seg_transform = seg_transform
        self.device = device
        self.vlm_resize = transforms.Compose([transforms.Resize((336, 336))])
        
        for _, row in metadata_df.iterrows():
            raw_path = str(row['image_path'])
            img_path = raw_path
            if not os.path.exists(img_path):
                if "kaggle_3m/" in raw_path:
                    relative_part = raw_path.split("kaggle_3m/")[-1]
                    img_path = os.path.join(base_img_dir, "kaggle_3m", relative_part)
                else:
                    img_path = os.path.join(base_img_dir, os.path.basename(raw_path))

            if not os.path.exists(img_path):
                continue

            has_tumor = row['has_tumor']
        
            q = (
                "Analyze this MRI slice where the tumor region is delineated by a yellow contour. "
                "Is a tumor visible? If yes, provide the histologic grade (1 or 2), "
                "tumor area (mm^2), estimated mass (g), and max diameter (mm)."
            )

            if has_tumor:
                a = (
                    f"Yes, a tumor is visible in the delineated region. "
                    f"Grade: {int(float(row['grade']))}. "
                    f"Area: {row['tumor_area_mm2']:.2f} mm^2. "
                    f"Mass: {row['tumor_mass_g']:.4f} g. "
                    f"Diameter: {row['tumor_diameter_mm']:.2f} mm."
                )
            else:
                a = "No tumor is visible. Grade: 0. Area: 0.0 mm^2. Mass: 0.0 g. Diameter: 0.0 mm."

            self.image_paths.append(img_path)
            self.questions.append(q)
            self.answers.append(a)

    def __len__(self) -> int: return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_pil = Image.open(self.image_paths[idx]).convert("RGB")
        
     
        delineated_image, _ = delineate_roi_on_image(image_pil, self.seg_model, self.seg_transform, self.device)
        final_vlm_image = self.vlm_resize(delineated_image)
        
        return final_vlm_image, self.questions[idx], self.answers[idx]

def vlm_collate_fn(batch):
    images, questions, answers = zip(*batch)
    return list(images), list(questions), list(answers)


def align_image_tokens(input_ids, attention_mask, labels, img_tok_idx, expected=576):
    new_input_ids = input_ids.clone()
    new_attention_mask = attention_mask.clone()
    new_labels = labels.clone() if labels is not None else None

    for i in range(input_ids.shape[0]):
        count = (input_ids[i] == img_tok_idx).sum().item()
        if count == expected - 1:
            img_indices = (input_ids[i] == img_tok_idx).nonzero(as_tuple=True)[0]
            last_idx = img_indices[-1]
            new_input_ids[i] = torch.cat([input_ids[i, :last_idx+1], torch.tensor([img_tok_idx]).to(input_ids.device), input_ids[i, last_idx+1:-1]])
            new_attention_mask[i] = torch.cat([attention_mask[i, :-1], torch.tensor([1]).to(attention_mask.device)])
            if new_labels is not None:
                new_labels[i] = torch.cat([labels[i, :last_idx+1], torch.tensor([-100]).to(labels.device), labels[i, last_idx+1:-1]])
    return new_input_ids, new_attention_mask, new_labels

def build_training_batch(images, questions, answers, processor, img_tok_idx):
    prompts = [f"USER: <image>\n{q}\nASSISTANT:" for q in questions]
    full_texts = [f"USER: <image>\n{q}\nASSISTANT: {a}{processor.tokenizer.eos_token}" for q, a in zip(questions, answers)]

    toks_full = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
    toks_prompt = processor(text=prompts, images=images, return_tensors="pt", padding=True)

    input_ids, labels = toks_full.input_ids, toks_full.input_ids.clone()
    prompt_lens = torch.sum(toks_prompt.attention_mask, dim=1)
    for i in range(labels.size(0)):
        labels[i, :prompt_lens[i]] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100

    input_ids, attention_mask, labels = align_image_tokens(input_ids, toks_full.attention_mask, labels, img_tok_idx)

    return {
        "input_ids": input_ids,
        "pixel_values": toks_full.pixel_values.to(dtype=torch.float16),
        "attention_mask": attention_mask,
        "labels": labels,
    }



def run_evaluation(model, processor, data_loader, device, description="Evaluating"):
    model.eval()
    physics_eval = PhysicsEvaluator()
    img_tok_idx = getattr(model.config, "image_token_index", 32000)
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=description):
            images, questions, answers = batch
            prompts = [f"USER: <image>\n{q}\nASSISTANT:" for q in questions]
            
            gen_inputs_raw = processor(text=prompts, images=images, return_tensors="pt", padding=True)
            ids_aligned, mask_aligned, _ = align_image_tokens(gen_inputs_raw.input_ids, gen_inputs_raw.attention_mask, None, img_tok_idx)
            
            gen_inputs = {
                "input_ids": ids_aligned.to(device),
                "pixel_values": gen_inputs_raw.pixel_values.to(device, dtype=torch.float16),
                "attention_mask": mask_aligned.to(device)
            }
            
            with autocast():
                generated_ids = model.generate(**gen_inputs, max_new_tokens=100, pad_token_id=processor.tokenizer.pad_token_id)
            
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
            responses = [d.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in d else d for d in decoded]
            physics_eval.update(answers, responses)

    metrics = physics_eval.compute_metrics()
    print(f"\n--- {description} Stats ---")
    print(f"  Grade Acc: {metrics['Grade_Acc']*100:.2f}%")
    print(f"  Physics R2 -> Area: {metrics['Area_R2']:.4f} | Mass: {metrics['Mass_R2']:.4f} | Diam: {metrics['Diam_R2']:.4f}")
    return metrics['Grade_Acc'] + (metrics['Area_R2'] + metrics['Mass_R2'] + metrics['Diam_R2']) / 3.0

def discover_lora_targets(model) -> List[str]:
    text_keys = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    target_suffixes = set()
    for name, _ in model.named_modules():
        if any(k in name for k in text_keys):
            target_suffixes.add(name.split(".")[-1])
    return sorted(list(target_suffixes))



if __name__ == "__main__":
    config = {
        "model_path": "/home/ealam/vlm/Medllava/llava_med_local/",
        "csv_path": "/home/ealam/vlm/mri_dataset/lgg_physics_metadata_v2.csv",
        "base_img_dir": "/home/ealam/vlm/mri_dataset/",
        "seg_model_path": "/home/ealam/vlm/best_model_segmentation_v2.pth",
        "save_path": "./Llava_med_vlm112_physics",
        "lr": 2e-5,
        "epochs": 25,
        "batch_size": 2,
        "grad_accum": 4,
        "patience": 5,
        "seed": 42
    }

    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])

    print("Step 1: Initializing Segmentation Model for Delineation...")
    seg_model = get_segmentation_model()
    seg_model.load_state_dict(torch.load(config["seg_model_path"], map_location=DEVICE), strict=False)
    seg_model.to(DEVICE).eval()
    seg_transform = get_segmentation_transforms()

    print(f"\nStep 2: Loading LLaVA-Med on {DEVICE}...")
    processor = AutoProcessor.from_pretrained(config["model_path"])
    model = LlavaForConditionalGeneration.from_pretrained(
        config["model_path"],
        torch_dtype=torch.float16,
        device_map={"": DEVICE},
        low_cpu_mem_usage=True
    )

    processor.patch_size = model.config.vision_config.patch_size
    processor.vision_feature_select_strategy = "default"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    img_tok_idx = getattr(model.config, "image_token_index", 32000)

    print("Step 3: Attaching LoRA Adapters...")
    target_modules = discover_lora_targets(model)
    lora_config = LoraConfig(
        r=32, lora_alpha=64, target_modules=target_modules,
        lora_dropout=0.05, task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    print("Step 4: Preparing Delineated Physics Data...")
    df = pd.read_csv(config["csv_path"])
    train_val, test_df = train_test_split(df, test_size=0.20, random_state=config["seed"])
    train_df, val_df = train_test_split(train_val, test_size=0.20, random_state=config["seed"])

    train_ds = VLM_PhysicsDelineatedDataset(train_df, config["base_img_dir"], seg_model, seg_transform, DEVICE)
    val_ds = VLM_PhysicsDelineatedDataset(val_df, config["base_img_dir"], seg_model, seg_transform, DEVICE)
    test_ds = VLM_PhysicsDelineatedDataset(test_df, config["base_img_dir"], seg_model, seg_transform, DEVICE)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=vlm_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], collate_fn=vlm_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], collate_fn=vlm_collate_fn)

    optimizer = AdamW(model.parameters(), lr=config["lr"])
    scaler = GradScaler()
    best_score = -float("inf")
    patience_counter = 0

    print("Step 5: Starting Delineated Physics Training Loop...")
    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for imgs, qs, ans in pbar:
            batch = build_training_batch(imgs, qs, ans, processor, img_tok_idx)
            batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            with autocast():
                loss = model(**batch).loss
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        score = run_evaluation(model, processor, val_loader, DEVICE, f"Val Ep {epoch+1}")
        if score > best_score:
            best_score = score
            patience_counter = 0
            model.save_pretrained(config["save_path"])
            print(f"New Best Score: {best_score:.4f}. Model saved.")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print("Early stopping triggered.")
                break

    print("Process complete.")
