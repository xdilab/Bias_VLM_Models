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
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import cv2
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



def get_segmentation_model() -> nn.Module:
    """Initializes the DeepLabV3 model for binary segmentation."""
    model = deeplabv3_resnet101(weights=None)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    return model

def get_segmentation_transforms() -> A.Compose:
    """Standard normalization and resizing for the segmentation model."""
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def post_process_mask(mask: np.ndarray, kernel_size: int = 5, min_area: int = 100) -> np.ndarray:
    """Cleans up the binary mask using morphological operations."""
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
    """Runs segmentation and draws a yellow contour on the original image."""
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



def align_image_tokens(input_ids, attention_mask, labels, img_tok_idx, expected=576):
    """Manually fixes the 575 vs 576 image token mismatch for LLaVA-Med."""
    new_input_ids = input_ids.clone()
    new_attention_mask = attention_mask.clone()
    new_labels = labels.clone() if labels is not None else None

    for i in range(input_ids.shape[0]):
        count = (input_ids[i] == img_tok_idx).sum().item()
        if count == expected - 1:
            img_indices = (input_ids[i] == img_tok_idx).nonzero(as_tuple=True)[0]
            last_idx = img_indices[-1]
            new_input_ids[i] = torch.cat([input_ids[i, :last_idx+1], torch.tensor([img_tok_idx], device=input_ids.device), input_ids[i, last_idx+1:-1]])
            new_attention_mask[i] = torch.cat([attention_mask[i, :-1], torch.tensor([1], device=attention_mask.device)])
            if new_labels is not None:
                new_labels[i] = torch.cat([labels[i, :last_idx+1], torch.tensor([-100], device=labels.device), labels[i, last_idx+1:-1]])
    return new_input_ids, new_attention_mask, new_labels



class VLM_QADataset(Dataset):
    def __init__(self, image_paths: List[str], metadata_df: pd.DataFrame, seg_model: nn.Module, seg_transform: A.Compose, device: str, is_train: bool = True):
        self.image_paths: List[str] = []
        self.questions: List[str] = []
        self.answers: List[str] = []
        self.seg_model = seg_model
        self.seg_transform = seg_transform
        self.device = device
        self.is_train = is_train

        self.vlm_resize = transforms.Compose([transforms.Resize((336, 336))])
        mdx = metadata_df.set_index("Patient")
        
        for img_path in tqdm(image_paths, desc=f"Loading Dataset ({'Train' if is_train else 'Val'})"):
            pid_folder = os.path.basename(os.path.dirname(img_path))
            pid_key = "_".join(pid_folder.split("_")[0:3])
            
            if pid_key in mdx.index:
                row = mdx.loc[[pid_key]].iloc[0]
                grade = row.get("neoplasm_histologic_grade")
                if pd.notna(grade) and int(grade) in [1, 2]:
                    self.image_paths.append(img_path)
                    self.answers.append(int(grade))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image_pil = Image.open(img_path).convert("RGB")
        
        delineated_image, seg_found_tumor = delineate_roi_on_image(image_pil, self.seg_model, self.seg_transform, self.device)
        final_vlm_image = self.vlm_resize(delineated_image)
        
        grade = self.answers[idx]
        
        if seg_found_tumor:
            q = "What is the histologic grade of the brain tumor (delineated) in the MRI: one or two?"
            a = f"The grade of the tumor is {'two' if grade == 2 else 'one'}."
        else:
            q = "Is a tumor visible in the delineated region of the MRI?"
            a = "No tumor is visible."

        return final_vlm_image, q, a

def vlm_collate_fn(batch):
    images, questions, answers = zip(*batch)
    return list(images), list(questions), list(answers)

def build_training_batch_cpu_main(images, questions, answers, processor: AutoProcessor, img_tok_idx: int):
    prompts = [f"USER: <image>\n{q}\nASSISTANT:" for q in questions]
    full_texts = [f"USER: <image>\n{q}\nASSISTANT: {a}{processor.tokenizer.eos_token}" for q, a in zip(questions, answers)]
    
    toks_prompt = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    toks_full = processor(text=full_texts, images=images, return_tensors="pt", padding=True)

    input_ids, labels = toks_full.input_ids, toks_full.input_ids.clone()
    prompt_lens = torch.sum(toks_prompt.attention_mask, dim=1)
    for i in range(labels.size(0)):
        labels[i, : prompt_lens[i]] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100

    input_ids, attention_mask, labels = align_image_tokens(input_ids, toks_full.attention_mask, labels, img_tok_idx)

    return {
        "input_ids": input_ids,
        "pixel_values": toks_full.pixel_values,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def run_evaluation(model, processor, data_loader: DataLoader, device, description="Evaluating"):
    model.eval()
    vlm_correct, total_samples = 0, 0
    total_loss_sum, total_loss_count = 0.0, 0
    img_tok_idx = getattr(model.config, "image_token_index", 32000)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=description):
            images, questions, answers = batch
            prompts = [f"USER: <image>\n{q}\nASSISTANT:" for q in questions]
            
            with autocast():
                gen_inputs_raw = processor(text=prompts, images=images, return_tensors="pt", padding=True)
                ids_aligned, mask_aligned, _ = align_image_tokens(gen_inputs_raw.input_ids, gen_inputs_raw.attention_mask, None, img_tok_idx)
                
                gen_inputs = {
                    "input_ids": ids_aligned.to(device),
                    "pixel_values": gen_inputs_raw.pixel_values.to(device, dtype=torch.float16),
                    "attention_mask": mask_aligned.to(device)
                }
                generated_ids = model.generate(**gen_inputs, max_new_tokens=25, pad_token_id=processor.tokenizer.pad_token_id)
            
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
            for i in range(len(decoded)):
                pred = decoded[i].split("ASSISTANT:")[-1].strip().lower()
                gold = answers[i].lower()
                
                if ("no tumor" in gold and "no tumor" in pred) or \
                   ("one" in gold and "one" in pred and "two" not in pred) or \
                   ("two" in gold and "two" in pred and "one" not in pred):
                    vlm_correct += 1
                total_samples += 1

            batch_cpu = build_training_batch_cpu_main(images, questions, answers, processor, img_tok_idx)
            with autocast():
                ce_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_cpu.items()}
                ce_inputs["pixel_values"] = ce_inputs["pixel_values"].to(dtype=torch.float16)
                out = model(**ce_inputs, return_dict=True)
                total_loss_sum += out.loss.item()
                total_loss_count += 1

    vlm_acc = (vlm_correct / total_samples) * 100 if total_samples else 0.0
    avg_loss = (total_loss_sum / total_loss_count) if total_loss_count else float("inf")
    print(f"\n--- {description} Result: Accuracy: {vlm_acc:.2f}% | Loss: {avg_loss:.4f} ---")
    return vlm_acc

def discover_lora_targets(llava_model) -> List[str]:
    text_keys = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    target_suffixes: set[str] = set()
    for name, _ in llava_model.named_modules():
        if any(k in name for k in text_keys):
            target_suffixes.add(name.split(".")[-1])
    return sorted(list(target_suffixes))

if __name__ == "__main__":
  
    config = {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "local_llava_path": "/home/ealam/vlm/Medllava/llava_med_local/",
        "base_path": "/home/ealam/vlm/mri_dataset/kaggle_3m/",
        "csv_path": "/home/ealam/vlm/mri_dataset/kaggle_3m/data.csv",
        "seg_model_path": "/home/ealam/vlm/best_model_segmentation_v2.pth",
        "save_path": "./Llava_med_vlm112",
        "lr": 2e-5,
        "batch_size": 2, 
        "epochs": 25, # Updated to 25 epochs
        "early_stopping_patience": 5, # Early stopping patience added back
        "seed": 42,
    }

    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    DEVICE = config["device"]

    print("Step 1: Initializing Segmentation Model...")
    seg_model = get_segmentation_model()
    seg_model.load_state_dict(torch.load(config["seg_model_path"], map_location=DEVICE), strict=False)
    seg_model.to(DEVICE).eval()
    seg_transform = get_segmentation_transforms()

    print("\nStep 2: Loading LLaVA-Med...")
    base_model = LlavaForConditionalGeneration.from_pretrained(config["local_llava_path"], torch_dtype=torch.float16, low_cpu_mem_usage=True).to(DEVICE)
    processor = AutoProcessor.from_pretrained(config["local_llava_path"])
    
    processor.patch_size = base_model.config.vision_config.patch_size
    processor.vision_feature_select_strategy = "default"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    img_tok_idx = getattr(base_model.config, "image_token_index", 32000)

    target_modules = discover_lora_targets(base_model)
    lora_cfg = LoraConfig(r=32, lora_alpha=64, target_modules=target_modules, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    peft_model = get_peft_model(base_model, lora_cfg)

    print("\nStep 3: Preparing Data...")
    metadata = pd.read_csv(config["csv_path"])
    
    all_imgs = [p.replace("_mask.tif", ".tif") for p in glob.glob(os.path.join(config["base_path"], "**", "*_mask.tif"), recursive=True)]
    all_imgs = [p for p in all_imgs if os.path.exists(p)]
    
    train_val_paths, test_paths = train_test_split(all_imgs, test_size=0.20, random_state=config["seed"])
    train_paths, val_paths = train_test_split(train_val_paths, test_size=0.20, random_state=config["seed"])

    train_ds = VLM_QADataset(train_paths, metadata, seg_model, seg_transform, DEVICE, is_train=True)
    val_ds = VLM_QADataset(val_paths, metadata, seg_model, seg_transform, DEVICE, is_train=False)
    test_ds = VLM_QADataset(test_paths, metadata, seg_model, seg_transform, DEVICE, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=vlm_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=vlm_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=vlm_collate_fn)

    print("\nStep 4: Training Delineated Llava_med_vlm112...")
    optimizer = AdamW(peft_model.parameters(), lr=config["lr"])
    scaler = GradScaler()
    best_acc = 0.0
    patience_counter = 0 

    for epoch in range(config["epochs"]):
        peft_model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, qs, ans in pbar:
            batch_cpu = build_training_batch_cpu_main(imgs, qs, ans, processor, img_tok_idx)
            batch = {k: v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch_cpu.items()}
            batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.float16)

            optimizer.zero_grad()
            with autocast():
                out = peft_model(**batch)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        val_acc = run_evaluation(peft_model, processor, val_loader, DEVICE, "Validation")
        
 
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0 
            peft_model.save_pretrained(config["save_path"])
            print(f"New Best Accuracy ({val_acc:.2f}%)! Adapters saved to {config['save_path']}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s). Best Accuracy: {best_acc:.2f}%")
            if patience_counter >= config["early_stopping_patience"]:
                print(f"\n--- Early stopping triggered after {epoch+1} epochs. ---")
                break

    print("\nStep 5: Final Evaluation...")
    if os.path.exists(config["save_path"]):
        final_peft = PeftModel.from_pretrained(base_model, config["save_path"]).to(DEVICE)
        run_evaluation(final_peft, processor, test_loader, DEVICE, "Final Test")
