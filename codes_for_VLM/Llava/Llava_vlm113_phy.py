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
import cv2
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import albumentations as A
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


dtype_to_use = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print(f"Using precision: {dtype_to_use}")



class JaccardLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred_probs = torch.sigmoid(y_pred)
        y_pred_flat = y_pred_probs.view(-1)
        y_true_flat = y_true.view(-1)
        intersection = (y_pred_flat * y_true_flat).sum()
        total = (y_pred_flat + y_true_flat).sum()
        union = total - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou



def _assistant_span(text: str) -> str:
    if not isinstance(text, str):
        return ""
    parts = text.split("ASSISTANT:")
    span = parts[-1] if parts else text
    return span.strip().lower()

def compute_token_accuracy_shifted(logits: torch.Tensor, labels: torch.Tensor, eos_id: int = None) -> tuple:
    with torch.no_grad():
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
        if eos_id is not None:
            labels = labels.clone()
            labels[labels == eos_id] = -100
        preds = torch.argmax(logits, dim=-1)
        mask = labels != -100
        correct = (preds[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        return correct, total

def compute_iou(pred_mask, true_mask, threshold=0.5):
    with torch.no_grad():
        pred_mask = (torch.sigmoid(pred_mask) > threshold).float()
        true_mask = true_mask.float()
        intersection = (pred_mask * true_mask).sum(dim=(1, 2))
        union = pred_mask.sum(dim=(1, 2)) + true_mask.sum(dim=(1, 2)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean().item()

def discover_lora_targets(llava_model, include_vision: bool = True) -> List[str]:
    text_keys = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    projector_keys = {"multi_modal_projector"}
    vision_keys = {"q_proj", "k_proj", "v_proj", "out_proj"}
    target_modules: set[str] = set()

    for name, module in llava_model.named_modules():
        if any(k in name for k in text_keys) and "language_model" in name:
            target_modules.add(name.split(".")[-1])
        if any(k in name for k in projector_keys):
            if hasattr(module, "weight") and getattr(module, "weight", None) is not None:
                target_modules.add(name.split(".")[-1])
        if include_vision and ("vision_tower" in name) and any(k in name for k in vision_keys):
            target_modules.add(name.split(".")[-1])

    return sorted(list(target_modules))



class VLM_Physics_Seg_Dataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, is_train: bool = True):
        self.image_paths: List[str] = []
        self.mask_paths: List[str] = []
        self.questions: List[str] = []
        self.answers: List[str] = []
        
       
        if is_train:
            self.image_transform = transforms.Compose([transforms.Resize((336, 336))])
        else:
            self.image_transform = transforms.Compose([transforms.Resize((336, 336))])
            
        self.mask_transform = transforms.Compose([
            transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

       
        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing Dataset"):
            img_path = row['image_path']
          
            mask_path = img_path.replace(".tif", "_mask.tif")
            
            if not os.path.exists(img_path):
                continue
            
         
            has_tumor = row['has_tumor']
            if not os.path.exists(mask_path):
            
                mask_exists = False
            else:
                mask_exists = True

            q = (
                "Analyze this MRI slice. Is a tumor visible? "
                "If yes, provide the histologic grade (1 or 2), "
                "tumor area (mm^2), estimated mass (g), and max diameter (mm)."
            )

            if has_tumor:
                grade = int(row['grade'])
                area = row['tumor_area_mm2']
                mass = row['tumor_mass_g']
                diameter = row['tumor_diameter_mm']
                
                a = (
                    f"Yes, a tumor is visible. "
                    f"Grade: {grade}. "
                    f"Area: {area} mm^2. "
                    f"Mass: {mass} g. "
                    f"Diameter: {diameter} mm."
                )
            else:
                a = (
                    "No tumor is visible in this MRI scan. "
                    "Grade: 0. "
                    "Area: 0.0 mm^2. "
                    "Mass: 0.0 g. "
                    "Diameter: 0.0 mm."
                )

            self.image_paths.append(img_path)
            self.mask_paths.append(mask_path if mask_exists else None)
            self.questions.append(q)
            self.answers.append(a)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
       
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.image_transform(image)
        
  
        mask_path = self.mask_paths[idx]
        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")
            mask_tensor = self.mask_transform(mask)
            mask_tensor = (mask_tensor > 0).float()
        else:
        
            mask_tensor = torch.zeros((1, 336, 336), dtype=torch.float32)

        return image, mask_tensor, self.questions[idx], self.answers[idx]



class LlavaPhysicsSegModel(nn.Module):
    def __init__(self, llava_model):
        super().__init__()
        self.llava = llava_model
        self.vision_tower = self.llava.vision_tower
        
       
        self.seg_model = smp.DeepLabV3Plus(
            encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1,
        )
        smp_encoder_channels = self.seg_model.encoder.out_channels
        
        
        self.clip_hidden_size = 1024
        
       
        self.projection = nn.ModuleList([
            nn.Conv2d(self.clip_hidden_size, smp_encoder_channels[1], kernel_size=1),
            nn.Conv2d(self.clip_hidden_size, smp_encoder_channels[2], kernel_size=1),
            nn.Conv2d(self.clip_hidden_size, smp_encoder_channels[3], kernel_size=1),
            nn.Conv2d(self.clip_hidden_size, smp_encoder_channels[4], kernel_size=1),
            nn.Conv2d(self.clip_hidden_size, smp_encoder_channels[5], kernel_size=1),
        ])

    def forward(self, input_ids, pixel_values, attention_mask, labels=None, seg_masks_gt=None, **kwargs):
       
        image_features_output = self.vision_tower(pixel_values, output_hidden_states=True)
        
      
        image_features_grid = image_features_output.hidden_states[-1][:, 1:, :]
        
        batch_size, patch_grid_size_sq, hidden_size = image_features_grid.shape
        patch_grid_size = int(math.sqrt(patch_grid_size_sq)) 
        
        
        seg_features = image_features_grid.reshape(
            batch_size, patch_grid_size, patch_grid_size, hidden_size
        ).permute(0, 3, 1, 2).contiguous()

   
        projected_features = [proj(seg_features) for proj in self.projection]

      
        scaled_projected_features = list(projected_features)
        scaled_projected_features[1] = F.interpolate(
            scaled_projected_features[1], scale_factor=4, mode='bilinear', align_corners=False
        )

        
        decoder_features = [None] + scaled_projected_features
        decoder_output = self.seg_model.decoder(decoder_features)
        seg_logits = self.seg_model.segmentation_head(decoder_output)
        
        
        seg_logits = F.interpolate(seg_logits, size=(336, 336), mode='bilinear', align_corners=False)

      
        vqa_output = self.llava(
            input_ids=input_ids, 
            pixel_values=pixel_values, 
            attention_mask=attention_mask, 
            labels=labels, 
            return_dict=True
        )

        return {
            "vqa_loss": vqa_output.loss, 
            "vqa_logits": vqa_output.logits, 
            "seg_logits": seg_logits.squeeze(1)
        }


def vlm_collate_fn_for_training(batch):
    images, masks, questions, answers = zip(*batch)
    masks_tensor = torch.stack(masks)
    return list(images), masks_tensor, list(questions), list(answers)

def vlm_collate_fn_for_evaluation(batch):
    images, masks, questions, answers = zip(*batch)
    masks_tensor = torch.stack(masks)
    return list(images), masks_tensor, list(questions), list(answers)

def build_training_batch_cpu_main(images, masks, questions, answers, processor: AutoProcessor):
    prompts = [f"USER: <image>\n{q}\nASSISTANT:" for q in questions]
    full_texts = [
        f"USER: <image>\n{q}\nASSISTANT: {a}{processor.tokenizer.eos_token}"
        for q, a in zip(questions, answers)
    ]

    toks_prompt = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    toks_full = processor(text=full_texts, images=images, return_tensors="pt", padding=True)

    labels = toks_full.input_ids.clone()
    prompt_lens = torch.sum(toks_prompt.attention_mask, dim=1)
    for i in range(labels.size(0)):
        labels[i, : prompt_lens[i]] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "input_ids": toks_full.input_ids,
        "pixel_values": toks_full.pixel_values,
        "attention_mask": toks_full.attention_mask,
        "labels": labels,
        "seg_masks_gt": masks,
    }



class PhysicsEvaluator:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.true_grades, self.pred_grades = [], []
        self.true_areas, self.pred_areas = [], []
        self.true_masses, self.pred_masses = [], []
        self.true_diams, self.pred_diams = [], []

    def parse_text(self, text):
        grade, area, mass, diam = 0, 0.0, 0.0, 0.0
        text_lower = text.lower()
        
        if "no tumor" in text_lower:
            has_tumor = False
        else:
            has_tumor = True
            g_match = re.search(r"Grade:\s*(\d)", text, re.IGNORECASE)
            if g_match: grade = int(g_match.group(1))
            
            a_match = re.search(r"Area:\s*([\d\.]+)", text, re.IGNORECASE)
            if a_match:
                try: area = float(a_match.group(1)) 
                except: pass
            
            m_match = re.search(r"Mass:\s*([\d\.]+)", text, re.IGNORECASE)
            if m_match:
                try: mass = float(m_match.group(1)) 
                except: pass

            d_match = re.search(r"Diameter:\s*([\d\.]+)", text, re.IGNORECASE)
            if d_match:
                try: diam = float(d_match.group(1)) 
                except: pass
                
        return has_tumor, grade, area, mass, diam

    def update(self, true_texts, pred_texts):
        for t_txt, p_txt in zip(true_texts, pred_texts):
            t_has, t_grade, t_area, t_mass, t_diam = self.parse_text(t_txt)
            p_has, p_grade, p_area, p_mass, p_diam = self.parse_text(p_txt)
            
            self.true_grades.append(t_grade)
            self.pred_grades.append(p_grade)
            self.true_areas.append(t_area)
            self.pred_areas.append(p_area)
            self.true_masses.append(t_mass)
            self.pred_masses.append(p_mass)
            self.true_diams.append(t_diam)
            self.pred_diams.append(p_diam)

    def _calc_tolerance_acc(self, true_vals, pred_vals, rel_tol=0.20, abs_tol=1.0):
        if not true_vals: return 0.0
        correct_count = 0
        for t, p in zip(true_vals, pred_vals):
            diff = abs(t - p)
            allowed_dev = max(t * rel_tol, abs_tol)
            if diff <= allowed_dev: correct_count += 1
        return correct_count / len(true_vals)

    def compute_metrics(self):
        metrics = {}
       
        if self.true_grades:
            metrics["Grade_Acc"] = accuracy_score(self.true_grades, self.pred_grades)
        else:
            metrics["Grade_Acc"] = 0.0
            
    
        if len(self.true_areas) > 1:
            metrics["Area_R2"] = r2_score(self.true_areas, self.pred_areas)
            metrics["Mass_R2"] = r2_score(self.true_masses, self.pred_masses)
            metrics["Diam_R2"] = r2_score(self.true_diams, self.pred_diams)
            
            
            metrics["Area_Tol_Acc"] = self._calc_tolerance_acc(self.true_areas, self.pred_areas, rel_tol=0.20, abs_tol=75.0)
            metrics["Mass_Tol_Acc"] = self._calc_tolerance_acc(self.true_masses, self.pred_masses, rel_tol=0.20, abs_tol=0.1)
            metrics["Diam_Tol_Acc"] = self._calc_tolerance_acc(self.true_diams, self.pred_diams, rel_tol=0.20, abs_tol=3.0) 
        else:
            for k in ["Area_R2", "Mass_R2", "Diam_R2", "Area_Tol_Acc", "Mass_Tol_Acc", "Diam_Tol_Acc"]: 
                metrics[k] = 0.0
        return metrics



def run_evaluation(model, processor, data_loader: DataLoader, device, description="Evaluating"):
    model.eval()
    physics_eval = PhysicsEvaluator()
    
    total_vqa_loss_sum, total_seg_loss_sum, total_iou = 0.0, 0.0, 0.0
    total_tok_correct, total_tok_count = 0, 0
    total_count = 0
    seg_loss_fn = JaccardLoss().to(device)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=description):
            images, masks_gt, questions, answers = batch
            
           
            prompts = [f"USER: <image>\n{q}\nASSISTANT:" for q in questions]
            with autocast():
                
                gen_inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
                generated_ids = model.llava.generate(
                    **gen_inputs, max_new_tokens=100, pad_token_id=processor.tokenizer.pad_token_id
                )
            
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
            pred_answers = [_assistant_span(d) for d in decoded]
            physics_eval.update(answers, pred_answers)

            
            batch_cpu = build_training_batch_cpu_main(images, masks_gt, questions, answers, processor)
            batch_gpu = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_cpu.items()}
            
            with autocast():
                outputs = model(**batch_gpu)
                vqa_loss = outputs["vqa_loss"]
                seg_logits = outputs["seg_logits"]
                seg_loss = seg_loss_fn(seg_logits, batch_gpu["seg_masks_gt"].squeeze(1).to(device))

            if vqa_loss is not None: total_vqa_loss_sum += vqa_loss.item()
            if seg_loss is not None: total_seg_loss_sum += seg_loss.item()
            
           
            c, n = compute_token_accuracy_shifted(outputs["vqa_logits"].detach(), batch_gpu["labels"], eos_id=processor.tokenizer.eos_token_id)
            total_tok_correct += c
            total_tok_count += n
            
            
            current_iou = compute_iou(seg_logits, batch_gpu["seg_masks_gt"].squeeze(1).to(device))
            total_iou += current_iou
            total_count += 1

    
    metrics = physics_eval.compute_metrics()
    grade_acc = metrics["Grade_Acc"]
    area_r2 = metrics["Area_R2"]
    
    avg_iou = total_iou / total_count if total_count else 0.0
    avg_vqa_loss = total_vqa_loss_sum / total_count if total_count else float("inf")
    ppl = math.exp(avg_vqa_loss) if avg_vqa_loss < 50 else float("inf")
    tok_acc = (total_tok_correct / total_tok_count) * 100 if total_tok_count else 0.0

    print(f"\n--- Results for {description} ---")
    print(f"  - Perplexity:         {ppl:.4f}")
    print(f"  - Token Accuracy:     {tok_acc:.2f}%")
    print(f"  - Class. Accuracy:    {grade_acc*100:.2f}%")
    print(f"  - Segmentation IoU:   {avg_iou:.4f}")
    print("-" * 20)
    print(f"  - Area R2: {metrics['Area_R2']:.4f}  |  Acc(@20%): {metrics['Area_Tol_Acc']*100:.1f}%")
    print(f"  - Mass R2: {metrics['Mass_R2']:.4f}  |  Acc(@20%): {metrics['Mass_Tol_Acc']*100:.1f}%")
    print(f"  - Diam R2: {metrics['Diam_R2']:.4f}  |  Acc(@20%): {metrics['Diam_Tol_Acc']*100:.1f}%")
    print("-" * 40)
    
    return grade_acc, area_r2, avg_iou



if __name__ == "__main__":
    config = {
        "device": "cuda:2" if torch.cuda.is_available() else "cpu",
        "csv_path": "/home/ealam/Downloads/LGG dataset Cameron/lgg_physics_metadata_v2.csv",
        "local_llava_path": "/home/ealam/Desktop/llava-1.5-7b-local",
        "save_path": "/home/ealam/Desktop/llava-physics-multitask-seg113",
        "learning_rate": 2e-5,
        "batch_size": 2, 
        "num_epochs": 25,
        "early_stopping_patience": 5,
        "seed": 42,
        "include_vision_lora": True,
        "seg_loss_weight": 1.0, 
        "num_workers": 4,
    }

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    print("Step 1: Gathering and splitting data...")
    df = pd.read_csv(config["csv_path"])
    usable_df, _ = train_test_split(df, test_size=0.01, random_state=config["seed"])
    train_val_df, test_df = train_test_split(usable_df, test_size=0.20, random_state=config["seed"])
    train_df, val_df = train_test_split(train_val_df, test_size=0.20, random_state=config["seed"])

    print("\nStep 2: Setting up Multi-Task Model...")
    DEVICE = config["device"]
    base_model = LlavaForConditionalGeneration.from_pretrained(
        config["local_llava_path"], torch_dtype=dtype_to_use, low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(config["local_llava_path"])
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        base_model.resize_token_embeddings(len(processor.tokenizer))

    target_modules = discover_lora_targets(base_model, include_vision=config["include_vision_lora"])
    lora_cfg = LoraConfig(r=32, lora_alpha=64, target_modules=target_modules, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    
    peft_model = get_peft_model(base_model, lora_cfg)
    
    
    multitask_model = LlavaPhysicsSegModel(peft_model).to(DEVICE)
    
   
    print("\nStep 3: Preparing DataLoaders...")
    train_ds = VLM_Physics_Seg_Dataset(train_df, is_train=True)
    val_ds = VLM_Physics_Seg_Dataset(val_df, is_train=False)
    test_ds = VLM_Physics_Seg_Dataset(test_df, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], collate_fn=vlm_collate_fn_for_training)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], collate_fn=vlm_collate_fn_for_evaluation)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], collate_fn=vlm_collate_fn_for_evaluation)

    print("\nStep 4: Starting Fine-Tuning...")
    trainable_params = [p for p in multitask_model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config["learning_rate"])
    
    
    use_scaler = (dtype_to_use == torch.float16)
    scaler = GradScaler(enabled=use_scaler)
    
    seg_loss_fn = JaccardLoss().to(DEVICE)
    
    num_training_steps = len(train_loader) * config["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps)

    best_combined_metric = -float("inf")
    patience = 0

    for epoch in range(config["num_epochs"]):
        multitask_model.train()
        total_loss = 0.0
        
        for images, masks, questions, answers in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            batch_cpu = build_training_batch_cpu_main(images, masks, questions, answers, processor)
            batch_gpu = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch_cpu.items()}
            
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                outputs = multitask_model(**batch_gpu)
                vqa_loss = outputs["vqa_loss"]
                seg_logits = outputs["seg_logits"]
                
                seg_loss = seg_loss_fn(seg_logits, batch_gpu["seg_masks_gt"].squeeze(1).to(DEVICE))
                
               
                loss = vqa_loss + (config["seg_loss_weight"] * seg_loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        print(f"\nEpoch {epoch+1} Avg Loss -> {total_loss / len(train_loader):.4f}")

     
        val_grade_acc, val_area_r2, val_iou = run_evaluation(
            multitask_model, processor, val_loader, DEVICE, "Validation Set Eval"
        )
        
        
        current_combined_metric = val_grade_acc + val_area_r2 + val_iou
        
        if current_combined_metric > best_combined_metric:
            print(f"  -> New best metric ({current_combined_metric:.4f}). Saving...")
            best_combined_metric = current_combined_metric
            patience = 0
            
            os.makedirs(config["save_path"], exist_ok=True)
            
            torch.save(multitask_model.seg_model.state_dict(), os.path.join(config["save_path"], "seg_model.pth"))
            torch.save(multitask_model.projection.state_dict(), os.path.join(config["save_path"], "projection.pth"))
            
            multitask_model.llava.save_pretrained(os.path.join(config["save_path"], "llava_lora"))
            processor.save_pretrained(os.path.join(config["save_path"], "processor"))
        else:
            patience += 1
            print(f"  -> No improvement for {patience} epoch(s).")
            if patience >= config["early_stopping_patience"]:
                print("Early stopping.")
                break
        print("="*80)

    print("\nStep 5: Final Evaluation...")
    if os.path.exists(os.path.join(config["save_path"], "seg_model.pth")):
        # Reconstruct model
        base = LlavaForConditionalGeneration.from_pretrained(config["local_llava_path"], torch_dtype=dtype_to_use, low_cpu_mem_usage=True)
        peft = PeftModel.from_pretrained(base, os.path.join(config["save_path"], "llava_lora"))
        final_model = LlavaPhysicsSegModel(peft).to(DEVICE)
        
        final_model.seg_model.load_state_dict(torch.load(os.path.join(config["save_path"], "seg_model.pth")))
        final_model.projection.load_state_dict(torch.load(os.path.join(config["save_path"], "projection.pth")))
        
        run_evaluation(final_model, processor, test_loader, DEVICE, "Final Test Evaluation")
    else:
        print("No model saved.")
