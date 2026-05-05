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
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
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
    Qwen2_5_VLForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from torch.cuda.amp import GradScaler, autocast


warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.backends.cuda.matmul.allow_tf32 = True

# Define script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Check for BFloat16 support
dtype_to_use = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print(f"Using precision: {dtype_to_use}")



def get_segmentation_model() -> nn.Module:
    model = deeplabv3_resnet101(weights=None, aux_logits=True)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
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
        largest_label_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        if stats[largest_label_idx, cv2.CC_STAT_AREA] > min_area:
            processed_mask[labels == largest_label_idx] = 255
    return processed_mask.astype(np.uint8)

def delineate_roi_on_image(pil_image: Image.Image, seg_model: nn.Module, seg_transform: A.Compose, device: str) -> Image.Image:
    open_cv_image_rgb = np.array(pil_image.convert("RGB"))
    open_cv_image_bgr = cv2.cvtColor(open_cv_image_rgb, cv2.COLOR_RGB2BGR)
    augmented = seg_transform(image=open_cv_image_rgb)
    image_tensor = augmented['image'].to(device).unsqueeze(0)
    
    seg_model.eval()
    with torch.no_grad():
        output = seg_model(image_tensor)['out']
    
    mask = torch.sigmoid(output).squeeze().cpu().numpy()
    binary_mask = (mask > 0.5).astype(np.uint8)
    cleaned_mask = post_process_mask(binary_mask)
    
    original_size = (pil_image.width, pil_image.height)
    resized_mask = cv2.resize(cleaned_mask, original_size, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        cv2.drawContours(open_cv_image_bgr, contours, -1, (0, 255, 255), 2)
    
    delineated_rgb = cv2.cvtColor(open_cv_image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(delineated_rgb)



def _assistant_span(text: str) -> str:
    if not isinstance(text, str): return ""
    parts = text.split("assistant\n")
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

def discover_lora_targets(model, include_vision: bool = True) -> List[str]:
    text_keys = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    projector_keys = {"vision_projector", "linear_1", "linear_2"}
    vision_keys = {"q_proj", "k_proj", "v_proj", "out_proj"}
    target_suffixes: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear): continue
        if any(k in name for k in text_keys): target_suffixes.add(name.split(".")[-1])
        if any(k in name for k in projector_keys): target_suffixes.add(name.split(".")[-1])
        if include_vision and ("visual" in name or "vision_tower" in name) and any(k in name for k in vision_keys):
            target_suffixes.add(name.split(".")[-1])
    return sorted(target_suffixes)



class VLM_Physics_Dataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, seg_model: nn.Module, seg_transform: A.Compose, device: str, is_train: bool = True):
        self.image_paths, self.questions, self.answers = [], [], []
        self.seg_model = seg_model
        self.seg_transform = seg_transform
        self.device = device
        self.vlm_transform = transforms.Compose([transforms.Resize((336, 336))])

        missing_count = 0
        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing dataset"):
            raw_path = str(row['image_path'])
            
            # --- Specific Path Fix from vlm111 ---
            if "kaggle_3m/" in raw_path:
                idx = raw_path.find("kaggle_3m/")
                sub_path = raw_path[idx:]
                img_path = os.path.join("/workspace/mri_dataset/", sub_path)
            else:
                img_path = raw_path

           
            if not os.path.exists(img_path):
                alt_path = os.path.join(script_dir, img_path)
                if os.path.exists(alt_path):
                    img_path = alt_path
                else:
                    missing_count += 1
                    continue
            
            q = ("Analyze this MRI slice. Is a tumor visible? "
                 "If yes, provide the histologic grade (1 or 2), "
                 "tumor area (mm^2), estimated mass (g), and max diameter (mm).")

            if row['has_tumor']:
                
                grade_val = int(float(row['grade']))
                a = (f"Yes, a tumor is visible. Grade: {grade_val}. "
                     f"Area: {row['tumor_area_mm2']} mm^2. Mass: {row['tumor_mass_g']} g. "
                     f"Diameter: {row['tumor_diameter_mm']} mm.")
            else:
                a = ("No tumor is visible in this MRI scan. Grade: 0. "
                     "Area: 0.0 mm^2. Mass: 0.0 g. Diameter: 0.0 mm.")
            
            self.image_paths.append(img_path)
            self.questions.append(q)
            self.answers.append(a)
        
        if missing_count > 0:
            print(f"[WARNING] Skipping {missing_count} samples because images were not found.")
            print(f"Sample path attempted: {img_path if 'img_path' in locals() else 'None'}")

    def __len__(self) -> int: return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_pil = Image.open(self.image_paths[idx]).convert("RGB")
        delineated_image = delineate_roi_on_image(image_pil, self.seg_model, self.seg_transform, self.device)
        final_image = self.vlm_transform(delineated_image)
        return final_image, self.questions[idx], self.answers[idx]



def vlm_collate_fn(batch):
    images, questions, answers = zip(*batch)
    return list(images), list(questions), list(answers)

def build_training_batch_cpu_main(images, questions, answers, processor: AutoProcessor):
    prompts_list, full_texts_list = [], []
    for q, a in zip(questions, answers):
        msg_prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        msg_full = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
                    {"role": "assistant", "content": [{"type": "text", "text": a}]}]
        prompts_list.append(processor.apply_chat_template(msg_prompt, tokenize=False, add_generation_prompt=True))
        full_texts_list.append(processor.apply_chat_template(msg_full, tokenize=False, add_generation_prompt=False) + processor.tokenizer.eos_token)

    toks_prompt = processor(text=prompts_list, images=images, return_tensors="pt", padding=True)
    toks_full = processor(text=full_texts_list, images=images, return_tensors="pt", padding=True)
    labels = toks_full.input_ids.clone()
    prompt_lens = torch.sum(toks_prompt.attention_mask, dim=1)
    for i in range(labels.size(0)): labels[i, :prompt_lens[i]] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch_cpu = {k: v for k, v in toks_full.items()}
    batch_cpu["labels"] = labels
    return batch_cpu



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

def run_evaluation(model, processor, data_loader, device, description="Evaluating"):
    model.eval()
    physics_eval = PhysicsEvaluator()
    total_loss_sum, total_loss_count = 0.0, 0
    total_tok_correct, total_tok_count = 0, 0
    debug_printed = False
    with torch.no_grad():
        for images, questions, answers in tqdm(data_loader, desc=description):
            prompts = [processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}], tokenize=False, add_generation_prompt=True) for q in questions]
            with autocast():
                gen_inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
                generated_ids = model.generate(**gen_inputs, max_new_tokens=100, pad_token_id=processor.tokenizer.pad_token_id)
            gen_trimmed = [g[len(p):] for g, p in zip(generated_ids, gen_inputs.input_ids)]
            pred_answers = processor.batch_decode(gen_trimmed, skip_special_tokens=True)
            physics_eval.update(answers, pred_answers)
            if not debug_printed:
                print(f"\n[DEBUG] Pred: {pred_answers[0]} | True: {answers[0]}"); debug_printed = True
            batch_cpu = build_training_batch_cpu_main(images, questions, answers, processor)
            ce_inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_cpu.items()}
            if "pixel_values" in ce_inputs: ce_inputs["pixel_values"] = ce_inputs["pixel_values"].to(dtype=dtype_to_use)
            with autocast():
                out = model(**ce_inputs, return_dict=True)
                total_loss_sum += out.loss.item(); total_loss_count += 1
            c, n = compute_token_accuracy_shifted(out.logits.detach(), ce_inputs["labels"], processor.tokenizer.eos_token_id)
            total_tok_correct += c; total_tok_count += n
    avg_loss = total_loss_sum / max(1, total_loss_count); ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    metrics = physics_eval.compute_metrics()
    print(f"\n--- {description} Results ---")
    print(f"Perplexity: {ppl:.4f} | Token Acc: {(total_tok_correct/max(1,total_tok_count))*100:.2f}% | Grade Acc: {metrics['Grade_Acc']*100:.2f}%")
    print(f"Area R2: {metrics['Area_R2']:.4f} | Mass R2: {metrics['Mass_R2']:.4f} | Diam R2: {metrics['Diam_R2']:.4f}\n")
    return metrics['Grade_Acc'], metrics["Area_R2"], metrics["Mass_R2"], metrics["Diam_R2"]


if __name__ == "__main__":
    config = {
        "device": "cuda:1" if torch.cuda.is_available() else "cpu",
        "csv_path": os.path.join(script_dir, "lgg_physics_metadata_v2.csv"),
        "local_qwen_path": "/workspace/qwen/saved_model",
        "save_path": "/workspace/qwen/qwen-physics-vlm112",
        "segmentation_model_path": "/workspace/best_model_segmentation_v2.pth",
        "lr": 2e-5,
        "batch_size": 2,
        "grad_accum": 4,
        "epochs": 25,
        "patience": 5,
        "seed": 42,
    }

    torch.manual_seed(config["seed"]); np.random.seed(config["seed"]); random.seed(config["seed"])
    DEVICE = config["device"]

    print("Step 1: Loading Segmentation Model...")
    seg_model = get_segmentation_model()
    state_dict = torch.load(config['segmentation_model_path'], map_location=DEVICE)
    msg = seg_model.load_state_dict(state_dict, strict=False)
    print(f"Loaded Segmentation Model. Unexpected keys (ignored): {msg.unexpected_keys}")
    
    seg_model.to(DEVICE).eval()
    seg_transform = get_segmentation_transforms()

    print("Step 2: Loading VLM Model...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config["local_qwen_path"], torch_dtype=dtype_to_use, low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(config["local_qwen_path"])
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        base_model.resize_token_embeddings(len(processor.tokenizer))

    target_modules = discover_lora_targets(base_model)
    lora_cfg = LoraConfig(r=32, lora_alpha=64, target_modules=target_modules, lora_dropout=0.05, task_type="CAUSAL_LM")
    peft_model = get_peft_model(base_model, lora_cfg).to(DEVICE)

    print("Step 3: Preparing DataLoaders...")
    df = pd.read_csv(config["csv_path"])
    train_val_df, test_df = train_test_split(df, test_size=0.20, random_state=config["seed"])
    train_df, val_df = train_test_split(train_val_df, test_size=0.20, random_state=config["seed"])

    train_ds = VLM_Physics_Dataset(train_df, seg_model, seg_transform, DEVICE)
    val_ds = VLM_Physics_Dataset(val_df, seg_model, seg_transform, DEVICE)
    test_ds = VLM_Physics_Dataset(test_df, seg_model, seg_transform, DEVICE)

    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty! Please check your CSV 'image_path' and ensure they match '/workspace/mri_dataset/'.")

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=vlm_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], collate_fn=vlm_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], collate_fn=vlm_collate_fn)

    optimizer = AdamW(peft_model.parameters(), lr=config["lr"])
    scaler = GradScaler(enabled=(dtype_to_use == torch.float16))
    best_score, patience_count = -float("inf"), 0

    for epoch in range(config["epochs"]):
        peft_model.train(); total_loss = 0.0
        for step, (imgs, qs, ans) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in build_training_batch_cpu_main(imgs, qs, ans, processor).items()}
            if "pixel_values" in batch: batch["pixel_values"] = batch["pixel_values"].to(dtype=dtype_to_use)
            with autocast():
                loss = peft_model(**batch).loss / config["grad_accum"]
            scaler.scale(loss).backward()
            if (step + 1) % config["grad_accum"] == 0:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            total_loss += loss.item() * config["grad_accum"]

        val_score = sum(run_evaluation(peft_model, processor, val_loader, DEVICE, "Validation")) / 4.0
        if val_score > best_score:
            best_score = val_score; patience_count = 0
            peft_model.save_pretrained(config["save_path"])
        else:
            patience_count += 1
            if patience_count >= config["patience"]: break

    if os.path.exists(config["save_path"]):
        final_peft = PeftModel.from_pretrained(Qwen2_5_VLForConditionalGeneration.from_pretrained(config["local_qwen_path"], torch_dtype=dtype_to_use), config["save_path"]).to(DEVICE)
        run_evaluation(final_peft, processor, test_loader, DEVICE, "Final Test")
