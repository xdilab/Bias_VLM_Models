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
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
from PIL import Image
import torchvision.transforms as transforms
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

def discover_lora_targets(llava_model, include_vision: bool = True) -> List[str]:
    
    text_keys = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    projector_keys = {"multi_modal_projector", "linear_1", "linear_2"}
    vision_keys = {"q_proj", "k_proj", "v_proj", "out_proj"}

    target_suffixes: set[str] = set()

    for name, module in llava_model.named_modules():
     
        if not isinstance(module, torch.nn.Linear):
            continue
            
        if any(k in name for k in text_keys):
            target_suffixes.add(name.split(".")[-1])
        if any(k in name for k in projector_keys):
            target_suffixes.add(name.split(".")[-1])
        if include_vision and ("vision_tower" in name) and any(k in name for k in vision_keys):
            target_suffixes.add(name.split(".")[-1])

    if not target_suffixes:
        target_suffixes = text_keys

    return sorted(target_suffixes)



class VLM_Physics_Dataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, is_train: bool = True):
        self.image_paths: List[str] = []
        self.questions: List[str] = []
        self.answers: List[str] = []

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((336, 336)), 
            ])
        else:
            self.transform = transforms.Compose([transforms.Resize((336, 336))])

        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing dataset"):
            img_path = row['image_path']
            if not os.path.exists(img_path):
                continue

            has_tumor = row['has_tumor']
            
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
            self.questions.append(q)
            self.answers.append(a)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        return image, self.questions[idx], self.answers[idx]



def vlm_collate_fn_for_training(batch):
    images, questions, answers = zip(*batch)
    return list(images), list(questions), list(answers)

def vlm_collate_fn_for_evaluation(batch):
    images, questions, answers = zip(*batch)
    return list(images), list(questions), list(answers)

def build_training_batch_cpu_main(images, questions, answers, processor: AutoProcessor):
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

    batch_cpu = {
        "input_ids": toks_full.input_ids,
        "pixel_values": toks_full.pixel_values,
        "attention_mask": toks_full.attention_mask,
        "labels": labels,
    }
    return batch_cpu



class PhysicsEvaluator:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.true_grades, self.pred_grades = [], []
        self.true_areas, self.pred_areas = [], []
        self.true_masses, self.pred_masses = [], []
        self.true_diams, self.pred_diams = [], []
        self.correct_detection, self.total_samples = 0, 0

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
                try: 
                    area = float(a_match.group(1)) 
                except: 
                    pass
            
            m_match = re.search(r"Mass:\s*([\d\.]+)", text, re.IGNORECASE)
            if m_match:
                try: 
                    mass = float(m_match.group(1)) 
                except: 
                    pass

            d_match = re.search(r"Diameter:\s*([\d\.]+)", text, re.IGNORECASE)
            if d_match:
                try: 
                    diam = float(d_match.group(1)) 
                except: 
                    pass
          
                
        return has_tumor, grade, area, mass, diam

    def update(self, true_texts, pred_texts):
        for t_txt, p_txt in zip(true_texts, pred_texts):
            self.total_samples += 1
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
    total_loss_sum, total_loss_count = 0.0, 0
    total_tok_correct, total_tok_count = 0, 0
    physics_eval = PhysicsEvaluator()
    debug_printed = False

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=description):
            images, questions, answers = batch 

            prompts = [f"USER: <image>\n{q}\nASSISTANT:" for q in questions]
            with autocast():
                gen_inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
                generated_ids = model.generate(**gen_inputs, max_new_tokens=100, pad_token_id=processor.tokenizer.pad_token_id)
            
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
            pred_answers = [_assistant_span(d) for d in decoded]
            physics_eval.update(answers, pred_answers)

            if not debug_printed:
                print(f"\n[DEBUG]\n  pred_raw=\n{decoded[0]}\n  true=\n{answers[0]}")
                debug_printed = True

            
            batch_cpu = build_training_batch_cpu_main(images, questions, answers, processor)
            ce_inputs = {
                "input_ids": batch_cpu["input_ids"].to(device),
                "pixel_values": batch_cpu["pixel_values"].to(device, dtype=dtype_to_use),
                "attention_mask": batch_cpu["attention_mask"].to(device),
                "labels": batch_cpu["labels"].to(device),
            }
            
            with autocast():
                out = model(**ce_inputs, return_dict=True)
                loss = out.loss
                logits = out.logits

            if not math.isnan(loss.item()):
                total_loss_sum += loss.item()
                total_loss_count += 1
            
            c, n = compute_token_accuracy_shifted(logits.detach(), ce_inputs["labels"], eos_id=processor.tokenizer.eos_token_id)
            total_tok_correct += c
            total_tok_count += n

    avg_loss = (total_loss_sum / total_loss_count) if total_loss_count else float("inf")
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    tok_acc = (total_tok_correct / total_tok_count) * 100 if total_tok_count else 0.0
    
    metrics = physics_eval.compute_metrics()
    grade_acc = metrics.get('Grade_Acc', 0)
    area_r2 = metrics.get("Area_R2", 0.0)
    mass_r2 = metrics.get("Mass_R2", 0.0)
    diam_r2 = metrics.get("Diam_R2", 0.0)
    
    print("\n--- Results for {} ---".format(description))
    print(f"  - Perplexity:         {ppl:.4f}")
    print(f"  - Token Accuracy:     {tok_acc:.2f}%")
    print(f"  - Class. Accuracy:    {grade_acc*100:.2f}%")
    print("-" * 20)
    print(f"  - Area R2: {area_r2:.4f}  |  Acc(@20%): {metrics['Area_Tol_Acc']*100:.1f}%")
    print(f"  - Mass R2: {mass_r2:.4f}  |  Acc(@20%): {metrics['Mass_Tol_Acc']*100:.1f}%")
    print(f"  - Diam R2: {diam_r2:.4f}  |  Acc(@20%): {metrics['Diam_Tol_Acc']*100:.1f}%")
    print("-" * 40)
    
    return grade_acc, area_r2, mass_r2, diam_r2



if __name__ == "__main__":
    config = {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "csv_path": "/home/ealam/Downloads/LGG dataset Cameron/lgg_physics_metadata_v2.csv",
        "local_llava_path": "/home/ealam/Desktop/llava-1.5-7b-local",
        "save_path": "/home/ealam/Desktop/llava-physics-pure-vlm111",
        "learning_rate": 2e-5,  
        "batch_size": 2,        
        "gradient_accumulation_steps": 4, 
        "num_epochs": 25,
        "early_stopping_patience": 5,
        "seed": 42,
        "include_vision_lora": True,
        "num_workers": 0,
    }

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config["seed"])

    print("Step 1: Gathering and splitting data...")
    df = pd.read_csv(config["csv_path"])
    print(f"Loaded {len(df)} rows from CSV.")
    
    
    usable_df, unused_df = train_test_split(df, test_size=0.01, random_state=config["seed"])
    print(f"Setting aside {len(unused_df)} images. Using {len(usable_df)} for experiment.")
    train_val_df, test_df = train_test_split(usable_df, test_size=0.20, random_state=config["seed"])
    train_df, val_df = train_test_split(train_val_df, test_size=0.20, random_state=config["seed"])
    print(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    print("\nStep 2: Setting up model and processor...")
    DEVICE = config["device"]
    base_model = LlavaForConditionalGeneration.from_pretrained(
        config["local_llava_path"], torch_dtype=dtype_to_use, low_cpu_mem_usage=True
    )
    
    base_model.gradient_checkpointing_enable()
    
    processor = AutoProcessor.from_pretrained(config["local_llava_path"])
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        base_model.resize_token_embeddings(len(processor.tokenizer))

    target_modules = discover_lora_targets(base_model, include_vision=config["include_vision_lora"])
    print("LoRA target modules:", target_modules)

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_cfg).to(DEVICE)
    
   
    
    optimizer = AdamW(peft_model.parameters(), lr=config["learning_rate"])
    
   
    use_scaler = (dtype_to_use == torch.float16)
    scaler = GradScaler(enabled=use_scaler)
    print(f"GradScaler Enabled: {use_scaler}")

    print("\nStep 3: Preparing DataLoaders...")
    train_ds = VLM_Physics_Dataset(train_df, is_train=True)
    val_ds = VLM_Physics_Dataset(val_df, is_train=False)
    test_ds = VLM_Physics_Dataset(test_df, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=vlm_collate_fn_for_training,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=vlm_collate_fn_for_evaluation,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=vlm_collate_fn_for_evaluation,
    )

    print("\nStep 4: Starting fine-tuning (Pure VLM)...")
    best_combined_score = -float("inf")
    patience = 0
    accum_steps = config["gradient_accumulation_steps"]

    def _to_device(batch_cpu):
        out = {}
        for k, v in batch_cpu.items():
            if k == "pixel_values":
                out[k] = v.to(DEVICE, dtype=dtype_to_use, non_blocking=True)
            elif torch.is_tensor(v):
                out[k] = v.to(DEVICE, non_blocking=True)
            else:
                out[k] = v
        return out

    optimizer.zero_grad() 
    
    for epoch in range(config["num_epochs"]):
        peft_model.train()
        total_loss = 0.0
        
        for step, (images, questions, answers) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            batch_cpu = build_training_batch_cpu_main(images, questions, answers, processor)
            batch = _to_device(batch_cpu)
            
            with autocast():
                out = peft_model(**batch, return_dict=True)
                loss = out.loss / accum_steps
            
            scaler.scale(loss).backward()
            
            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() 
            
            if not math.isnan(loss.item()):
                total_loss += loss.item() * accum_steps

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"\nEpoch {epoch+1} Avg Loss -> {avg_loss:.4f}")

        val_grade_acc, val_area_r2, val_mass_r2, val_diam_r2 = run_evaluation(
            peft_model, processor, val_loader, DEVICE, description="Validation Set Eval"
        )

        avg_regression_score = (val_area_r2 + val_mass_r2 + val_diam_r2) / 3.0
        current_combined_score = val_grade_acc + avg_regression_score

        print(f"  -> Combined Score: {current_combined_score:.4f}")

        if current_combined_score > best_combined_score:
            print(f"  -> New best Combined Score ({current_combined_score:.4f}). Saving adapters...")
            best_combined_score = current_combined_score
            patience = 0
            peft_model.save_pretrained(config["save_path"])
            processor.save_pretrained(config["save_path"])
        else:
            patience += 1
            print(f"  -> No improvement for {patience} epoch(s).")
            if patience >= config["early_stopping_patience"]:
                print("\n--- Early stopping triggered. ---")
                break
        print("=" * 80)

    print("\nStep 5: Loading best adapters for final evaluation...")
    if os.path.exists(config["save_path"]):
        base = LlavaForConditionalGeneration.from_pretrained(config["local_llava_path"], torch_dtype=dtype_to_use, low_cpu_mem_usage=True)
        final_peft = PeftModel.from_pretrained(base, config["save_path"]).to(DEVICE)
        run_evaluation(final_peft, processor, test_loader, DEVICE, description="Final Test Evaluation")
    else:
        print("No adapters were saved.")
