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
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
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

class VLM_QADataset(Dataset):
    """
    Dataset class tailored for MRI classification and grading.
    """
    def __init__(self, image_paths: List[str], metadata_df: pd.DataFrame):
        self.image_paths: List[str] = []
        self.questions: List[str] = []
        self.answers: List[str] = []

        mdx = metadata_df.set_index("Patient")
        for img_path in tqdm(image_paths, desc="Processing dataset"):
            mask_path = img_path.replace(".tif", "_mask.tif")
            if not os.path.exists(mask_path):
                continue

            mask_image = Image.open(mask_path)
            mask_array = np.array(mask_image)
            is_tumor_visible = np.any(mask_array > 0)

            q = "Is there a tumor visible in this MRI? If so, what is its histologic grade: one or two?"

            if is_tumor_visible:
                pid_folder = os.path.basename(os.path.dirname(img_path))
                pid_key = "_".join(pid_folder.split("_")[0:3])
                
                if pid_key in mdx.index:
                    row = mdx.loc[[pid_key]].iloc[0]
                    grade = row.get("neoplasm_histologic_grade")
                    if pd.notna(grade) and int(grade) in [1, 2]:
                        self.image_paths.append(img_path)
                        a = f"A tumor is visible. The grade of the tumor is {'two' if int(grade) == 2 else 'one'}."
                        self.questions.append(q)
                        self.answers.append(a)
            else:
                self.image_paths.append(img_path)
                a = "No tumor is visible in this MRI scan."
                self.questions.append(q)
                self.answers.append(a)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return image, self.questions[idx], self.answers[idx]

def vlm_collate_fn(batch):
    images, questions, answers = zip(*batch)
    return list(images), list(questions), list(answers)

def build_training_batch_lingshu(images, questions, answers, processor):
    texts = []
    for q, a in zip(questions, answers):
        messages = [
            {"role": "user", "content": [{"type": "image", "image": None}, {"type": "text", "text": q}]},
            {"role": "assistant", "content": [{"type": "text", "text": a}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text + processor.tokenizer.eos_token)

    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    labels = inputs.input_ids.clone()
    
    for i, text in enumerate(texts):
        sep = "<|im_start|>assistant\n"
        parts = text.split(sep)
        if len(parts) > 1:
            prompt_tokens = processor.tokenizer.encode(parts[0] + sep, add_special_tokens=False)
            labels[i, :len(prompt_tokens)] = -100
            
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs

def compute_token_accuracy(logits, labels):
    with torch.no_grad():
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
        preds = torch.argmax(logits, dim=-1)
        mask = labels != -100
        correct = (preds[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        return correct, total

def run_evaluation(model, processor, data_loader, device, description="Evaluating"):
    model.eval()
    vlm_correct = 0
    total_samples = 0
    total_loss_sum = 0.0
    total_loss_count = 0
    total_tok_correct = 0
    total_tok_count = 0

    with torch.no_grad():
        for images, questions, answers in tqdm(data_loader, desc=description):
            prompts = []
            for q in questions:
                msg = [{"role": "user", "content": [{"type": "image", "image": None}, {"type": "text", "text": q}]}]
                prompts.append(processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
            
         
            inputs_gen = processor(text=prompts, images=images, padding=True, return_tensors="pt")
            inputs_gen = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs_gen.items()}
            
            generated_ids = model.generate(**inputs_gen, max_new_tokens=30)
            generated_ids = [g[len(i):] for g, i in zip(generated_ids, inputs_gen["input_ids"])]
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for pred, true in zip(decoded, answers):
                pred_c, true_c = pred.lower(), true.lower()
                is_correct = False
                if "no tumor" in true_c:
                    if "no tumor" in pred_c: is_correct = True
                else:
                    want_two = "two" in true_c
                    has_one = "one" in pred_c or "1" in pred_c
                    has_two = "two" in pred_c or "2" in pred_c
                    if (want_two and has_two and not has_one) or (not want_two and has_one and not has_two):
                        is_correct = True
                if is_correct: vlm_correct += 1

            batch = build_training_batch_lingshu(images, questions, answers, processor)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with autocast(dtype=torch.bfloat16):
                out = model(**batch)
                loss = out.loss
                logits = out.logits

            total_loss_sum += loss.item()
            total_loss_count += 1
            c, n = compute_token_accuracy(logits.detach(), batch["labels"])
            total_tok_correct += c
            total_tok_count += n
            total_samples += len(answers)

    vlm_acc = (vlm_correct / total_samples) * 100 if total_samples else 0.0
    avg_loss = (total_loss_sum / total_loss_count) if total_loss_count else float("inf")
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    tok_acc = (total_tok_correct / total_tok_count) * 100 if total_tok_count else 0.0

    print(f"\n--- {description} ---")
    print(f"  VLM Accuracy (QA):    {vlm_acc:.2f}%")
    print(f"  Perplexity:           {ppl:.4f}")
    print(f"  Token Accuracy:       {tok_acc:.2f}%")
    return vlm_acc

if __name__ == "__main__":
    config = {
        "device": "cuda:0", 
        "model_path": "/workspace/models/Lingshu-7B",
        "csv_path": "/workspace/mri_dataset/kaggle_3m/data.csv",
        "base_path": "/workspace/mri_dataset/kaggle_3m/",
        "save_path": "/workspace/ling/lingshu-lora-mri_vlm111",
        "lr": 1e-4,
        "epochs": 25,
        "batch_size": 1, 
        "early_stopping_patience": 5,
        "seed": 42
    }

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    print("Step 1: Loading Lingshu-7B (Qwen2.5-VL architecture)...")
    processor = AutoProcessor.from_pretrained(config["model_path"], trust_remote_code=True)
    
   
    model = AutoModelForVision2Seq.from_pretrained(
        config["model_path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Step 2: Injecting LoRA...")
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Step 3: Data Loading...")
    all_image_paths = [p.replace("_mask.tif", ".tif") for p in glob.glob(os.path.join(config["base_path"], "*", "*_mask.tif"))]
    metadata_df = pd.read_csv(config["csv_path"])
    train_val_paths, test_paths = train_test_split(all_image_paths, test_size=0.20, random_state=config["seed"])
    train_paths, val_paths = train_test_split(train_val_paths, test_size=0.20, random_state=config["seed"])
    
    train_ds = VLM_QADataset(train_paths, metadata_df)
    val_ds = VLM_QADataset(val_paths, metadata_df)
    test_ds = VLM_QADataset(test_paths, metadata_df)
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=vlm_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], collate_fn=vlm_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], collate_fn=vlm_collate_fn)

    optimizer = AdamW(model.parameters(), lr=config["lr"])
    scaler = GradScaler()
    best_acc = 0.0
    patience_counter = 0

    print("Step 4: Training...")
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        for images, questions, answers in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = build_training_batch_lingshu(images, questions, answers, processor)
           
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

      
        val_acc = run_evaluation(model, processor, val_loader, model.device, "Validation")
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            model.save_pretrained(config["save_path"])
            print(f"Saved Checkpoint: {best_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

    print("\nStep 5: Final Evaluation...")
    if os.path.exists(config["save_path"]):
        print("Reloading best adapters to a SINGLE device (cuda:0) for final test...")
        # Free up memory
        del model
        torch.cuda.empty_cache()
        
      
        final_device = "cuda:0"
        base_model = AutoModelForVision2Seq.from_pretrained(
            config["model_path"],
            torch_dtype=torch.bfloat16,
            device_map={"": final_device}, 
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, config["save_path"]).to(final_device)
        run_evaluation(model, processor, test_loader, final_device, "Final Test Set")
    else:
        print("No save path found. Skipping final evaluation.")
