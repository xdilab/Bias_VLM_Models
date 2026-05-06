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
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2
import segmentation_models_pytorch as smp
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



class CombinedSegLoss(nn.Module):
    """Combines Dice and Jaccard for high-precision medical segmentation."""
    def __init__(self, smooth=1e-6):
        super(CombinedSegLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_pred_f = y_pred.view(-1)
        y_true_f = y_true.view(-1)
        
        intersection = (y_pred_f * y_true_f).sum()
        union = y_pred_f.sum() + y_true_f.sum() - intersection
        jaccard = (intersection + self.smooth) / (union + self.smooth)
        dice = (2. * intersection + self.smooth) / (y_pred_f.sum() + y_true_f.sum() + self.smooth)
        
        
        return 1.0 - (0.5 * jaccard + 0.5 * dice)



def align_image_tokens(input_ids, attention_mask, labels, img_tok_idx, expected=576):
    
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



class VLM_QASegDataset(Dataset):
    def __init__(self, image_paths: List[str], metadata_df: pd.DataFrame, is_train: bool = True):
        self.image_paths, self.mask_paths, self.questions, self.answers = [], [], [], []
        
       
        self.image_resize = transforms.Resize((336, 336))
        self.mask_transform = transforms.Compose([
            transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        mdx = metadata_df.set_index("Patient")
        for img_path in tqdm(image_paths, desc="Processing Multitask Dataset"):
            mask_path = img_path.replace(".tif", "_mask.tif")
            if not os.path.exists(mask_path): continue
            pid_folder = os.path.basename(os.path.dirname(img_path))
            pid_key = "_".join(pid_folder.split("_")[0:3])
            if pid_key in mdx.index:
                row = mdx.loc[[pid_key]].iloc[0]
                grade = row.get("neoplasm_histologic_grade")
                if pd.notna(grade) and int(grade) in [1, 2]:
                    has_tumor = np.sum(np.array(Image.open(mask_path))) > 0
                    q = "Analyze the provided medical image. Is a tumor visible? If so, identify it and determine its histologic grade: one or two."
                    a = f"A tumor is visible, and its grade is {'two' if int(grade) == 2 else 'one'}." if has_tumor else "There is no tumor visible in this image."
                    self.image_paths.append(img_path); self.mask_paths.append(mask_path)
                    self.questions.append(q); self.answers.append(a)

    def __len__(self) -> int: return len(self.image_paths)
    def __getitem__(self, idx: int):
        image = self.image_resize(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = self.mask_transform(Image.open(self.mask_paths[idx]).convert("L"))
        return image, (mask > 0).float(), self.questions[idx], self.answers[idx]

def vlm_collate_fn(batch):
    images, masks, questions, answers = zip(*batch)
    return list(images), torch.stack(masks), list(questions), list(answers)



class LlavaWithSegmentationHead(nn.Module):
    def __init__(self, llava_model):
        super().__init__()
        self.llava = llava_model
        
        if isinstance(self.llava, PeftModel):
            self.vision_tower = self.llava.base_model.model.model.vision_tower
        else:
            self.vision_tower = self.llava.model.vision_tower
        
     
        self.seg_model = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        smp_channels = self.seg_model.encoder.out_channels
        
       
        self.proj_os4  = nn.Conv2d(1024, smp_channels[1], kernel_size=1) 
        self.proj_os8  = nn.Conv2d(1024, smp_channels[3], kernel_size=1) 
        self.proj_os16 = nn.Conv2d(1024, smp_channels[5], kernel_size=1)

    def forward(self, input_ids, pixel_values, attention_mask, labels=None, **kwargs):
        
        vision_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        
      
        feat_early = vision_outputs.hidden_states[4][:, 1:, :] 
        feat_mid   = vision_outputs.hidden_states[12][:, 1:, :]
        feat_late  = vision_outputs.hidden_states[-1][:, 1:, :]
        
        B, N, C = feat_late.shape
        grid = int(math.sqrt(N)) 
        
        def to_2d(x): return x.transpose(1, 2).reshape(B, C, grid, grid).contiguous()
        
        f4, f12, f24 = to_2d(feat_early), to_2d(feat_mid), to_2d(feat_late)

        s1 = self.proj_os4(F.interpolate(f4, size=(84, 84), mode='bilinear', align_corners=True))
        s3 = self.proj_os8(F.interpolate(f12, size=(42, 42), mode='bilinear', align_corners=True))
        s5 = self.proj_os16(F.interpolate(f24, size=(21, 21), mode='bilinear', align_corners=True))
        
        
        s2 = F.interpolate(s1, scale_factor=1.0, mode='nearest')
        s4 = F.interpolate(s3, scale_factor=0.5, mode='bilinear', align_corners=True)

        features = [None, s1, s2, s3, s4, s5]
        
       
        decoder_output = self.seg_model.decoder(features)
        seg_logits = F.interpolate(self.seg_model.segmentation_head(decoder_output), size=(336, 336), mode='bilinear', align_corners=True)

       
        vqa_outputs = self.llava(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=labels, return_dict=True)
        
        return {"vqa_loss": vqa_outputs.loss, "vqa_logits": vqa_outputs.logits, "seg_logits": seg_logits.squeeze(1)}



def run_evaluation(model, processor, loader, device, description="Evaluating"):
    model.eval()
    vlm_correct, total_samples, total_iou = 0, 0, 0.0
    llava_base = model.llava if not isinstance(model.llava, PeftModel) else model.llava.base_model
    img_tok_idx = getattr(llava_base.config, "image_token_index", 32000)
    
    with torch.no_grad():
        for imgs, masks, qs, ans in tqdm(loader, desc=description):
            masks = masks.to(device)
            gen_raw = processor(text=[f"USER: <image>\n{q}\nASSISTANT:" for q in qs], images=imgs, return_tensors="pt", padding=True)
            ids, mask, _ = align_image_tokens(gen_raw.input_ids, gen_raw.attention_mask, None, img_tok_idx)
            
            with autocast():
                gen_out = model.llava.generate(input_ids=ids.to(device), pixel_values=gen_raw.pixel_values.to(device, dtype=torch.float16), attention_mask=mask.to(device), max_new_tokens=25)
                outputs = model(input_ids=ids.to(device), pixel_values=gen_raw.pixel_values.to(device, dtype=torch.float16), attention_mask=mask.to(device))

            decoded = processor.batch_decode(gen_out, skip_special_tokens=True)
            for d, a in zip(decoded, ans):
                pred_ans = d.split("ASSISTANT:")[-1].strip().lower()
                if (("no tumor" in a.lower() and "no tumor" in pred_ans) or ("one" in a.lower() and "one" in pred_ans and "two" not in pred_ans) or ("two" in a.lower() and "two" in pred_ans and "one" not in pred_ans)): vlm_correct += 1
                total_samples += 1
            
            pred_mask = (torch.sigmoid(outputs["seg_logits"]) > 0.5).float()
            intersection = (pred_mask * masks.squeeze(1)).sum(dim=(1, 2))
            union = pred_mask.sum(dim=(1, 2)) + masks.squeeze(1).sum(dim=(1, 2)) - intersection
            total_iou += ((intersection + 1e-6) / (union + 1e-6)).mean().item()
    
    acc = (vlm_correct / total_samples) * 100 if total_samples else 0.0
    miou = total_iou / len(loader) if len(loader) else 0.0
    print(f"\n--- {description} Result: VQA Acc: {acc:.2f}% | IoU: {miou:.4f} ---")
    return acc + (miou * 100)

def build_training_batch_cpu_main(images, masks, questions, answers, processor, img_tok_idx):
    prompts = [f"USER: <image>\n{q}\nASSISTANT:" for q in questions]
    full_texts = [f"USER: <image>\n{q}\nASSISTANT: {a}{processor.tokenizer.eos_token}" for q, a in zip(questions, answers)]
    toks_full = processor(text=full_texts, images=images, return_tensors="pt", padding=True)
    toks_prompt = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    
    input_ids, labels = toks_full.input_ids, toks_full.input_ids.clone()
    prompt_lens = torch.sum(toks_prompt.attention_mask, dim=1)
    for i in range(labels.size(0)): labels[i, : prompt_lens[i]] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    input_ids, attention_mask, labels = align_image_tokens(input_ids, toks_full.attention_mask, labels, img_tok_idx)
    return {"input_ids": input_ids, "pixel_values": toks_full.pixel_values, "attention_mask": attention_mask, "labels": labels, "seg_masks_gt": masks}

if __name__ == "__main__":
    config = {"device": "cuda:2", "model_path": "/home/ealam/vlm/Medllava/llava_med_local/", "base_path": "/home/ealam/vlm/mri_dataset/kaggle_3m/", "csv_path": "/home/ealam/vlm/mri_dataset/kaggle_3m/data.csv", "save_path": "./Llava_med_vlm113", "lr": 1e-5, "batch_size": 2, "epochs": 25, "patience": 5, "seg_weight": 0.5, "seed": 42}
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    DEVICE = config["device"]

    print("Step 1: Initializing Hierarchical Multi-task Architecture...")
    base_llava = LlavaForConditionalGeneration.from_pretrained(config["model_path"], torch_dtype=torch.float16, low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(config["model_path"])
    processor.patch_size = base_llava.config.vision_config.patch_size
    if processor.tokenizer.pad_token is None: processor.tokenizer.pad_token = processor.tokenizer.eos_token

    lora_cfg = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], task_type="CAUSAL_LM")
    peft_llava = get_peft_model(base_llava, lora_cfg)
    model = LlavaWithSegmentationHead(peft_llava).to(DEVICE)
    
  
    for name, param in model.vision_tower.named_parameters():
        param.requires_grad = False
        
    
    for param in model.parameters():
        if param.requires_grad: param.data = param.data.float()
    
    img_tok_idx = getattr(base_llava.config, "image_token_index", 32000)

    print("\nStep 2: Preparing Data (Train/Val/Test)...")
    metadata = pd.read_csv(config["csv_path"])
    all_imgs = [p.replace("_mask.tif", ".tif") for p in glob.glob(os.path.join(config["base_path"], "**", "*_mask.tif"), recursive=True)]
    train_paths, vt_paths = train_test_split(all_imgs, test_size=0.98, random_state=config["seed"])
    val_paths, test_paths = train_test_split(vt_paths, test_size=0.98, random_state=config["seed"])
    
    train_loader = DataLoader(VLM_QASegDataset(train_paths, metadata), batch_size=config["batch_size"], shuffle=True, collate_fn=vlm_collate_fn, drop_last=True)
    val_loader = DataLoader(VLM_QASegDataset(val_paths, metadata, is_train=False), batch_size=config["batch_size"], shuffle=False, collate_fn=vlm_collate_fn)
    test_loader = DataLoader(VLM_QASegDataset(test_paths, metadata, is_train=False), batch_size=config["batch_size"], shuffle=False, collate_fn=vlm_collate_fn)

    print("\nStep 3: Starting Training Loop...")
    optimizer = AdamW(model.parameters(), lr=config["lr"]); scaler = GradScaler(); seg_loss_fn = CombinedSegLoss().to(DEVICE)
    best_metric, patience_counter = 0.0, 0
    for epoch in range(config["epochs"]):
        model.train(); total_loss = 0.0; pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, masks, qs, ans in pbar:
            batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in build_training_batch_cpu_main(imgs, masks, qs, ans, processor, img_tok_idx).items()}
            batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.float16)
            optimizer.zero_grad()
            with autocast():
                out = model(**batch)
                loss = out["vqa_loss"] + config["seg_weight"] * seg_loss_fn(out["seg_logits"], batch["seg_masks_gt"].squeeze(1))
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); total_loss += loss.item(); pbar.set_postfix(loss=loss.item())

        metric = run_evaluation(model, processor, val_loader, DEVICE, "Validation")
        if metric > best_metric:
            best_metric = metric; patience_counter = 0; os.makedirs(config["save_path"], exist_ok=True)
            torch.save(model.seg_model.state_dict(), os.path.join(config["save_path"], "seg_head.pth"))
            model.llava.save_pretrained(os.path.join(config["save_path"], "vqa_adapters"))
            print(f"Best Metric Improved. Model Saved.")
        elif (patience_counter := patience_counter + 1) >= config["patience"]: break

    print("\nStep 4: Final Test...")
    if os.path.exists(config["save_path"]):
        best_base = LlavaForConditionalGeneration.from_pretrained(config["model_path"], torch_dtype=torch.float16)
        best_peft = PeftModel.from_pretrained(best_base, os.path.join(config["save_path"], "vqa_adapters"))
        best_model = LlavaWithSegmentationHead(best_peft).to(DEVICE)
        for p in best_model.parameters():
            if p.requires_grad: p.data = p.data.float()
        best_model.seg_model.load_state_dict(torch.load(os.path.join(config["save_path"], "seg_head.pth")))
        run_evaluation(best_model, processor, test_loader, DEVICE, "Final Test Set")
