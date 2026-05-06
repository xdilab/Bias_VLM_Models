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

class VLM_QADataset(Dataset):
    def __init__(self, image_paths: List[str], metadata_df: pd.DataFrame, is_train: bool = True):
        self.image_paths: List[str] = []
        self.questions: List[str] = []
        self.answers: List[str] = []

        self.transform = transforms.Compose([
            transforms.Resize((336, 336)),
        ])

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
        image = self.transform(image)
        return image, self.questions[idx], self.answers[idx]

def vlm_collate_fn_for_training(batch):
    images, questions, answers = zip(*batch)
    return list(images), list(questions), list(answers)

def vlm_collate_fn_for_evaluation(batch):
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
            
           
            new_input_ids[i] = torch.cat([input_ids[i, :last_idx+1], torch.tensor([img_tok_idx]), input_ids[i, last_idx+1:-1]])
            new_attention_mask[i] = torch.cat([attention_mask[i, :-1], torch.tensor([1])])
            if new_labels is not None:
                new_labels[i] = torch.cat([labels[i, :last_idx+1], torch.tensor([-100]), labels[i, last_idx+1:-1]])
    
    return new_input_ids, new_attention_mask, new_labels

def build_training_batch_cpu_main(images, questions, answers, processor: AutoProcessor, img_tok_idx: int):
    prompts = [f"USER: <image>\n{q}\nASSISTANT:" for q in questions]
    full_texts = [
        f"USER: <image>\n{q}\nASSISTANT: {a}{processor.tokenizer.eos_token}"
        for q, a in zip(questions, answers)
    ]

    toks_prompt = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    toks_full = processor(text=full_texts, images=images, return_tensors="pt", padding=True)

    input_ids = toks_full.input_ids
    labels = input_ids.clone()
    
   
    prompt_lens = torch.sum(toks_prompt.attention_mask, dim=1)
    for i in range(labels.size(0)):
        labels[i, : prompt_lens[i]] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100

    
    input_ids, attention_mask, labels = align_image_tokens(input_ids, toks_full.attention_mask, labels, img_tok_idx)

    batch_cpu = {
        "input_ids": input_ids,
        "pixel_values": toks_full.pixel_values,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    return batch_cpu

def _assistant_span(text: str) -> str:
    if not isinstance(text, str):
        return ""
    parts = text.split("ASSISTANT:")
    span = parts[-1] if parts else text
    return span.strip().lower()

def _has_one_two_flags(answer_text: str) -> Tuple[bool, bool]:
    answer_text = answer_text.replace("\u2019", "'")
    tokens = set(re.findall(r"\b(one|two|1|2)\b", answer_text))
    has_one = ("one" in tokens) or ("1" in tokens)
    has_two = ("two" in tokens) or ("2" in tokens)
    return has_one, has_two

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

def run_evaluation(model, processor, data_loader: DataLoader, device, description="Evaluating"):
    model.eval()
    vlm_correct = 0
    total_samples = 0
    total_loss_sum = 0.0
    total_loss_count = 0
    total_tok_correct = 0
    total_tok_count = 0

    img_tok_idx = getattr(model.config, "image_token_index", 32000)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=description):
            images, questions, answers = batch
            prompts = [f"USER: <image>\n{q}\nASSISTANT:" for q in questions]
            
            with autocast():
                # Prepare generation inputs with alignment fix
                gen_inputs_raw = processor(text=prompts, images=images, return_tensors="pt", padding=True)
                ids_aligned, mask_aligned, _ = align_image_tokens(gen_inputs_raw.input_ids, gen_inputs_raw.attention_mask, None, img_tok_idx)
                
                gen_inputs = {
                    "input_ids": ids_aligned.to(device),
                    "pixel_values": gen_inputs_raw.pixel_values.to(device, dtype=torch.float16),
                    "attention_mask": mask_aligned.to(device)
                }
                
                generated_ids = model.generate(
                    **gen_inputs,
                    max_new_tokens=25,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for i in range(len(decoded)):
                pred_span = _assistant_span(decoded[i])
                true_answer = answers[i]
                is_correct = False

                if "No tumor" in true_answer:
                    if "no tumor" in pred_span and "one" not in pred_span and "two" not in pred_span:
                        is_correct = True
                else:
                    want_two = "two" in true_answer
                    has_one, has_two = _has_one_two_flags(pred_span)
                    if (want_two and has_two and not has_one) or ((not want_two) and has_one and not has_two):
                        is_correct = True
                if is_correct:
                    vlm_correct += 1

           
            batch_cpu = build_training_batch_cpu_main(images, questions, answers, processor, img_tok_idx)
            with autocast():
                ce_inputs = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch_cpu.items()}
                if "pixel_values" in ce_inputs:
                    ce_inputs["pixel_values"] = ce_inputs["pixel_values"].to(dtype=torch.float16)
                
                out = model(**ce_inputs, return_dict=True)
                loss = out.loss
                logits = out.logits

            total_loss_sum += loss.item()
            total_loss_count += 1
            c, n = compute_token_accuracy_shifted(logits.detach(), ce_inputs["labels"], eos_id=processor.tokenizer.eos_token_id)
            total_tok_correct += c
            total_tok_count += n
            total_samples += len(answers)

    vlm_acc = (vlm_correct / total_samples) * 100 if total_samples else 0.0
    avg_loss = (total_loss_sum / total_loss_count) if total_loss_count else float("inf")
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    tok_acc = (total_tok_correct / total_tok_count) * 100 if total_tok_count else 0.0

    print(f"\n--- Results for {description} ---")
    print(f"  VLM Acc: {vlm_acc:.2f}% | PPL: {ppl:.4f} | Tok Acc: {tok_acc:.2f}%")
    return vlm_acc, ppl, tok_acc

def discover_lora_targets(llava_model) -> List[str]:
    text_keys = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    target_suffixes: set[str] = set()
    for name, _ in llava_model.named_modules():
        if any(k in name for k in text_keys):
            target_suffixes.add(name.split(".")[-1])
    return sorted(list(target_suffixes))

if __name__ == "__main__":
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "local_llava_path": "/home/ealam/vlm/Medllava/llava_med_local/",
        "base_path": "/home/ealam/vlm/mri_dataset/kaggle_3m/",
        "csv_path": "/home/ealam/vlm/mri_dataset/kaggle_3m/data.csv",
        "save_path": "./Llava_med_vlm111",
        "learning_rate": 2e-5,
        "batch_size": 2, 
        "num_epochs": 15,
        "early_stopping_patience": 5,
        "seed": 42,
    }

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    print("Step 1: Gathering MRI data...")
    all_image_paths = [p.replace("_mask.tif", ".tif") for p in glob.glob(os.path.join(config["base_path"], "*", "*_mask.tif"))]
    train_val_paths, test_paths = train_test_split(all_image_paths, test_size=0.20, random_state=config["seed"])
    train_paths, val_paths = train_test_split(train_val_paths, test_size=0.20, random_state=config["seed"])

    print("\nStep 2: Loading Llava_med_vlm111 base...")
    DEVICE = config["device"]
    base_model = LlavaForConditionalGeneration.from_pretrained(
        config["local_llava_path"], torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(DEVICE)
    processor = AutoProcessor.from_pretrained(config["local_llava_path"])

    if not hasattr(processor, "patch_size") or processor.patch_size is None:
        processor.patch_size = base_model.config.vision_config.patch_size
    if not hasattr(processor, "vision_feature_select_strategy") or processor.vision_feature_select_strategy is None:
        processor.vision_feature_select_strategy = "default"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    img_tok_idx = getattr(base_model.config, "image_token_index", 32000)

    target_modules = discover_lora_targets(base_model)
    lora_cfg = LoraConfig(
        r=32, lora_alpha=64, target_modules=target_modules,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_cfg)

    print("\nStep 3: Preparing DataLoaders...")
    metadata_df = pd.read_csv(config["csv_path"])
    train_ds = VLM_QADataset(train_paths, metadata_df, is_train=True)
    val_ds = VLM_QADataset(val_paths, metadata_df, is_train=False)
    test_ds = VLM_QADataset(test_paths, metadata_df, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=vlm_collate_fn_for_training)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=vlm_collate_fn_for_evaluation)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=vlm_collate_fn_for_evaluation)

    print("\nStep 4: Training Loop...")
    optimizer = AdamW(peft_model.parameters(), lr=config["learning_rate"])
    scaler = GradScaler()
    best_val_acc = 0.0
    patience = 0

    for epoch in range(config["num_epochs"]):
        peft_model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for images, questions, answers in pbar:
            batch_cpu = build_training_batch_cpu_main(images, questions, answers, processor, img_tok_idx)
            
            batch = {k: v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch_cpu.items()}
            if "pixel_values" in batch:
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

        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(train_loader):.4f}")
        val_acc, _, _ = run_evaluation(peft_model, processor, val_loader, DEVICE, "Validation Set")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            peft_model.save_pretrained(config["save_path"])
        else:
            patience += 1
            if patience >= config["early_stopping_patience"]:
                break

    print("\nStep 5: Final Evaluation...")
    if os.path.exists(config["save_path"]):
        final_peft = PeftModel.from_pretrained(base_model, config["save_path"]).to(DEVICE)
        run_evaluation(final_peft, processor, test_loader, DEVICE, "Final Test Set")
