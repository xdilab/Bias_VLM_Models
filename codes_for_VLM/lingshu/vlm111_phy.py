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
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    get_linear_schedule_with_warmup,
)
from torch.cuda.amp import GradScaler, autocast


warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype_to_use = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print(f"Local Environment Initialized: Using {DEVICE} with {dtype_to_use} precision.")



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
        
        # Regex extraction for numerical data in response
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



class LingshuPhysicsDataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, base_img_dir: str):
        self.image_paths, self.questions, self.answers = [], [], []

        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Indexing Local Files"):
            raw_path = str(row['image_path'])
            if "kaggle_3m/" in raw_path:
                sub_path = raw_path[raw_path.find("kaggle_3m/"):]
                img_path = os.path.join(base_img_dir, sub_path)
            else:
                img_path = raw_path

            if not os.path.exists(img_path):
                continue

            has_tumor = row['has_tumor']
            q = (
                "Analyze this MRI slice. Is a tumor visible? "
                "If yes, provide the histologic grade (1 or 2), "
                "tumor area (mm^2), estimated mass (g), and max diameter (mm)."
            )

            if has_tumor:
                a = (
                    f"Yes, a tumor is visible. "
                    f"Grade: {int(float(row['grade']))}. "
                    f"Area: {row['tumor_area_mm2']:.2f} mm^2. "
                    f"Mass: {row['tumor_mass_g']:.4f} g. "
                    f"Diameter: {row['tumor_diameter_mm']:.2f} mm."
                )
            else:
                a = "No tumor is visible in this MRI scan. Grade: 0. Area: 0.0 mm^2. Mass: 0.0 g. Diameter: 0.0 mm."

            self.image_paths.append(img_path)
            self.questions.append(q)
            self.answers.append(a)

    def __len__(self) -> int: return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return image, self.questions[idx], self.answers[idx]

def vlm_collate_fn(batch):
    images, questions, answers = zip(*batch)
    return list(images), list(questions), list(answers)

def build_batch(images, questions, answers, processor):
    texts = []
    for q, a in zip(questions, answers):
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
            {"role": "assistant", "content": [{"type": "text", "text": a}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text + processor.tokenizer.eos_token)

    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    labels = inputs.input_ids.clone()
    
    marker = "<|im_start|>assistant\n"
    for i, text in enumerate(texts):
        if marker in text:
            prompt_tokens = processor.tokenizer.encode(text.split(marker)[0] + marker, add_special_tokens=False)
            labels[i, :len(prompt_tokens)] = -100
            
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs



def run_evaluation(model, processor, data_loader, device, description="Evaluating"):
    model.eval()
    physics_eval = PhysicsEvaluator()
    debug_shown = False
    
    with torch.no_grad():
        for images, questions, answers in tqdm(data_loader, desc=description):
            prompts = [processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}], 
                       tokenize=False, add_generation_prompt=True) for q in questions]
            
            inputs_gen = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(device)
            with autocast(dtype=dtype_to_use):
                generated_ids = model.generate(**inputs_gen, max_new_tokens=100)
            
            generated_ids = [g[len(i):] for g, i in zip(generated_ids, inputs_gen["input_ids"])]
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)

            if not debug_shown:
                print(f"\n[SAMPLE] PRED: {decoded[0]} | TRUE: {answers[0]}")
                debug_shown = True

            physics_eval.update(answers, decoded)

    metrics = physics_eval.compute_metrics()
    print(f"\n--- {description} Stats ---")
    print(f"  Grade Acc: {metrics['Grade_Acc']*100:.2f}%")
   
    print(f"  Physics R2 -> Area: {metrics['Area_R2']:.4f} | Mass: {metrics['Mass_R2']:.4f} | Diam: {metrics['Diam_R2']:.4f}")
    
    return metrics['Grade_Acc'] + (metrics['Area_R2'] + metrics['Mass_R2'] + metrics['Diam_R2']) / 3.0



if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config = {
        "model_path": "/home/ealam/vlm/models/Lingshu-7B/",
        "csv_path": "/home/ealam/vlm/mri_dataset/lgg_physics_metadata_v2.csv",
        "base_img_dir": "/home/ealam/vlm/mri_dataset/",
        "save_path": os.path.join(script_dir, "lingshu-physics-v111-local"),
        "lr": 2e-5,
        "epochs": 25,
        "batch_size": 2,
        "grad_accum": 4,
        "patience": 5,
        "seed": 42
    }

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    print(f"Step 1: Loading Local Lingshu-7B Model on {DEVICE}...")
    processor = AutoProcessor.from_pretrained(config["model_path"], trust_remote_code=True)
    
   
    model = AutoModelForVision2Seq.from_pretrained(
        config["model_path"],
        torch_dtype=dtype_to_use,
        device_map={"": DEVICE},
        trust_remote_code=True,
    )

    print("Step 2: Attaching LoRA Adapters...")
    lora_config = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Step 3: Loading Local Metadata...")
    df = pd.read_csv(config["csv_path"])
    train_val, test_df = train_test_split(df, test_size=0.20, random_state=config["seed"])
    train_df, val_df = train_test_split(train_val, test_size=0.20, random_state=config["seed"])

    train_loader = DataLoader(LingshuPhysicsDataset(train_df, config["base_img_dir"]), 
                              batch_size=config["batch_size"], shuffle=True, collate_fn=vlm_collate_fn)
    val_loader = DataLoader(LingshuPhysicsDataset(val_df, config["base_img_dir"]), 
                            batch_size=config["batch_size"], collate_fn=vlm_collate_fn)
    test_loader = DataLoader(LingshuPhysicsDataset(test_df, config["base_img_dir"]), 
                             batch_size=config["batch_size"], collate_fn=vlm_collate_fn)

    optimizer = AdamW(model.parameters(), lr=config["lr"])
    scaler = GradScaler()
    best_score = -float("inf")
    patience_counter = 0

    print("Step 4: Starting Physics Training Loop...")
    for epoch in range(config["epochs"]):
        model.train()
        optimizer.zero_grad()
        
        for step, (imgs, qs, ans) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            batch = build_batch(imgs, qs, ans, processor)
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with autocast(dtype=dtype_to_use):
                loss = model(**batch).loss / config["grad_accum"]
            
            scaler.scale(loss).backward()
            
            if (step + 1) % config["grad_accum"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        score = run_evaluation(model, processor, val_loader, DEVICE, f"Val Ep {epoch+1}")
        
        if score > best_score:
            best_score = score
            patience_counter = 0
            model.save_pretrained(config["save_path"])
            print(f"Metrics Improved. Best Score: {best_score:.4f} - Saved.")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print("Early stopping triggered.")
                break

    if os.path.exists(config["save_path"]):
        print("\nStep 5: Final Performance Benchmarking...")
        del model
        torch.cuda.empty_cache()
        base = AutoModelForVision2Seq.from_pretrained(config["model_path"], torch_dtype=dtype_to_use, device_map={"": DEVICE})
        final_model = PeftModel.from_pretrained(base, config["save_path"])
        run_evaluation(final_model, processor, test_loader, DEVICE, "Final Test Set")

    print("Training process complete.")
