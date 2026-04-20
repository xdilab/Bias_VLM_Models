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
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

import segmentation_models_pytorch as smp
import cv2  
from torchvision.models.segmentation import deeplabv3_resnet101 
import albumentations as A 
from albumentations.pytorch import ToTensorV2 

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


def get_bounding_box_from_mask(mask_pil: Image.Image) -> Optional[Tuple[int, int, int, int]]:
 
    gt_mask_array = (np.array(mask_pil) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(gt_mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    all_points = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    return int(x), int(y), int(w), int(h)


def get_external_segmentation_model() -> nn.Module:
    model = deeplabv3_resnet101(weights='DeepLabV3_ResNet101_Weights.DEFAULT')
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    return model

def get_external_segmentation_transforms() -> A.Compose:
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

def get_predicted_mask_from_model(pil_image: Image.Image, seg_model: nn.Module, seg_transform: A.Compose, device: str) -> Image.Image:

    open_cv_image_rgb = np.array(pil_image.convert("RGB"))
    augmented = seg_transform(image=open_cv_image_rgb)
    image_tensor = augmented['image'].to(device).unsqueeze(0)

    seg_model.eval()
    with torch.no_grad():
        output = seg_model(image_tensor)['out']
    
    mask = torch.sigmoid(output).squeeze().cpu().numpy()
    binary_mask = (mask > 0.5).astype(np.uint8)
    cleaned_mask_resized = post_process_mask(binary_mask)
    
    original_size = (pil_image.width, pil_image.height)
    cleaned_mask_original_size = cv2.resize(cleaned_mask_resized, original_size, interpolation=cv2.INTER_NEAREST)
    
    return Image.fromarray(cleaned_mask_original_size)


class JaccardLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 4: y_pred = y_pred[:, 0]
        if y_true.dim() == 4: y_true = y_true[:, 0]
        y_pred_probs = torch.sigmoid(y_pred)
        y_pred_flat = y_pred_probs.view(y_pred.shape[0], -1)
        y_true_flat = y_true.view(y_true.shape[0], -1).float()
        intersection = (y_pred_flat * y_true_flat).sum(1)
        total = (y_pred_flat + y_true_flat).sum(1)
        union = total - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - iou
        return loss.mean() if self.reduction == "mean" else loss


class VLM_QASegDataset_TextHint(Dataset):
  
    def __init__(self, 
                 image_paths: List[str], 
                 metadata_df: pd.DataFrame, 
                 hint_seg_model: nn.Module, 
                 hint_seg_transform: A.Compose, 
                 device: str,
                 is_train: bool = True):
        
        self.image_paths: List[str] = []
        self.mask_paths: List[str] = []
        self.questions: List[str] = []
        self.answers: List[str] = []
        self.has_tumors: List[bool] = [] # Ground Truth

        
        self.hint_seg_model = hint_seg_model
        self.hint_seg_transform = hint_seg_transform
        self.device = device

        self.gt_mask_transform = transforms.Compose([
            transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        mdx = metadata_df.set_index("Patient")

        print(f"Building VLM_QASegDataset_TextHint (is_train={is_train}):", len(image_paths), "candidate images")
        for img_path in tqdm(image_paths, desc="Building Text Hint Dataset"):
            mask_path = img_path.replace(".tif", "_mask.tif")
            if not os.path.exists(mask_path):
                continue
            
            try:
               s
                gt_mask_pil = Image.open(mask_path).convert("L")
                gt_mask_arr = np.array(gt_mask_pil)
                gt_has_tumor = np.any(gt_mask_arr > 0)
                
                
                raw_image_pil = Image.open(img_path).convert("RGB")

            except Exception as e:
                print(f"Warning: Skipping corrupted file {img_path}, error: {e}")
                continue
            
            q, a = "", ""

            if gt_has_tumor:
               
                pid_folder = os.path.basename(os.path.dirname(img_path))
                pid_key = "_".join(pid_folder.split("_")[0:3])
                if pid_key not in mdx.index: continue
                row = mdx.loc[[pid_key]].iloc[0]
                grade = row.get("neoplasm_histologic_grade")
                if pd.isna(grade): continue
                try: grade_int = int(grade)
                except ValueError: continue
                if grade_int not in [1, 2]: continue
                
                a = f"A tumor is visible. The grade of the tumor is {'two' if grade_int == 2 else 'one'}."

           
                predicted_mask_pil = get_predicted_mask_from_model(raw_image_pil, self.hint_seg_model, self.hint_seg_transform, self.device)
                bbox_from_model = get_bounding_box_from_mask(predicted_mask_pil)

                if bbox_from_model:
                    x, y, w, h = bbox_from_model
                    q = f"A tumor is located within the bounding box [x={x}, y={y}, width={w}, height={h}]. What is the histologic grade of the brain tumor in the MRI: one or two?"
                else:
                    # Model failed to find a box, use a generic prompt
                    q = "A tumor is visible. What is the histologic grade of the brain tumor in the MRI: one or two?"
            
            else:
                # Ground truth is NO TUMOR
                a = "No tumor is visible in this MRI scan."
                q = "Is a tumor visible in the MRI?"

            self.image_paths.append(img_path)
            self.mask_paths.append(mask_path)
            self.questions.append(q)
            self.answers.append(a)
            self.has_tumors.append(gt_has_tumor) # Store ground truth status
        print(f"Final dataset size: {len(self.image_paths)} samples.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        q = self.questions[idx]
        a = self.answers[idx]
        has_tumor = self.has_tumors[idx]

       
        raw_image_pil = Image.open(img_path).convert("RGB")
        
     
        gt_mask_pil = Image.open(mask_path).convert("L")
        
        
        gt_mask_tensor = self.gt_mask_transform(gt_mask_pil)
        gt_mask_tensor = (gt_mask_tensor > 0).float()

       
        return raw_image_pil, gt_mask_tensor, q, a, has_tumor


def vlm_seg_collate_fn(batch):
    images, masks, questions, answers, has_tumors = zip(*batch)
    masks_tensor = torch.stack(masks, dim=0)
    has_tumors_tensor = torch.tensor(has_tumors, dtype=torch.bool)
    return list(images), masks_tensor, list(questions), list(answers), has_tumors_tensor

def build_training_batch_cpu_main(
    images, masks, questions, answers, has_tumors, processor: AutoProcessor,
):
    prompts_list = []
    full_texts_list = []
    for q, a in zip(questions, answers):
        prompt_msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        prompts_list.append(processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True))
        full_msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
                       {"role": "assistant", "content": a}]
        full_texts_list.append(processor.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False) + processor.tokenizer.eos_token)
    
    toks_prompt = processor(text=prompts_list, images=images, return_tensors="pt", padding=True)
    toks_full = processor(text=full_texts_list, images=images, return_tensors="pt", padding=True)

    labels = toks_full.input_ids.clone()
    prompt_lens = torch.sum(toks_prompt.attention_mask, dim=1)
    for i in range(labels.size(0)):
        labels[i, :prompt_lens[i]] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100

    batch_cpu = {k: v for k, v in toks_full.items()}
    batch_cpu["labels"] = labels
    batch_cpu["seg_masks_gt"] = masks
    batch_cpu["has_tumor"] = has_tumors
    return batch_cpu

def _to_device(batch_cpu: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch_cpu.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out



def _has_one_two_flags(answer_text: str) -> Tuple[bool, bool]:
    answer_text = answer_text.replace("\u2019", "'")
    tokens = set(re.findall(r"\b(one|two|1|2)\b", answer_text.lower()))
    has_one = ("one" in tokens) or ("1" in tokens)
    has_two = ("two" in tokens) or ("2" in tokens)
    return has_one, has_two

def compute_iou_batch(pred_logits: torch.Tensor, true_masks: torch.Tensor, threshold: float = 0.5) -> float:
    if pred_logits.dim() == 4: pred_logits = pred_logits[:, 0]
    if true_masks.dim() == 4: true_masks = true_masks[:, 0]
    with torch.no_grad():
        pred_mask = (torch.sigmoid(pred_logits) > threshold).float()
        true_mask = true_masks.float()
        intersection = (pred_mask * true_mask).sum(dim=(1, 2))
        union = pred_mask.sum(dim=(1, 2)) + true_mask.sum(dim=(1, 2)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean().item()

class QwenVLWithSegmentation(nn.Module):
    def __init__(self, qwen_model: nn.Module, seg_out_size: Tuple[int, int] = (336, 336), deeplab_encoder_name: str = "resnet34"):
        super().__init__()
        self.qwen = qwen_model
        self.seg_out_size = seg_out_size
        if hasattr(self.qwen, "base_model"): base = self.qwen.base_model
        else: base = self.qwen
        self.base_qwen = base
        self.visual = self.base_qwen.model.visual
        vis_hidden = self.base_qwen.config.vision_config.out_hidden_size
        self.deeplab = smp.DeepLabV3(encoder_name=deeplab_encoder_name, encoder_weights=None, in_channels=3, classes=1)
        for p in self.deeplab.encoder.parameters(): p.requires_grad = False
        encoder_out_channels = self.deeplab.encoder.out_channels[-1]
        self.qwen_to_smp = nn.Conv2d(vis_hidden, encoder_out_channels, kernel_size=1)

    def _visual_to_grid(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor, batch_size: int):
        vis_out = self.visual(pixel_values, grid_thw=image_grid_thw)
        if isinstance(vis_out, torch.Tensor): tokens = vis_out
        elif hasattr(vis_out, "last_hidden_state"): tokens = vis_out.last_hidden_state
        else: raise RuntimeError("Unexpected visual output type")
        N, C = tokens.shape
        B = batch_size
        if N % B != 0: raise RuntimeError(f"Visual shape {tokens.shape} not divisible by batch {B}")
        S = N // B
        H = int(math.sqrt(S))
        if H * H != S:
            try:
                patch_size = self.base_qwen.config.vision_config.patch_size
                image_size = self.base_qwen.config.vision_config.image_size
                H = W = image_size // patch_size
                if H * W != S: raise RuntimeError(f"Cannot infer grid shape. S={S}, H={H}, W={W}")
            except Exception: raise RuntimeError(f"Visual tokens S={S} not a perfect square")
        else: W = H
        tokens = tokens.view(B, S, C)
        feat_map = tokens.transpose(1, 2).contiguous().view(B, C, H, W)
        return feat_map

    def forward(self, **batch):
        seg_masks_gt = batch.pop("seg_masks_gt", None)
        has_tumor = batch.pop("has_tumor", None)
        input_ids = batch["input_ids"]
        batch_size = input_ids.size(0)
        pixel_values = batch["pixel_values"]
        image_grid_thw = batch.get("image_grid_thw", None)
        if image_grid_thw is None:
            try:
                image_grid_thw = self.base_qwen.model.image_grid_thw.to(pixel_values.device)
                image_grid_thw = image_grid_thw.repeat(batch_size, 1)
            except Exception as e: raise RuntimeError(f"image_grid_thw missing. Error: {e}")
        with autocast(enabled=True, dtype=torch.float16):
            vis_feats = self._visual_to_grid(pixel_values, image_grid_thw, batch_size)
            enc_last = self.qwen_to_smp(vis_feats)
            features = [enc_last]
            decoder_out = self.deeplab.decoder(features)
            seg_logits_full = self.deeplab.segmentation_head(decoder_out)
            seg_logits = F.interpolate(seg_logits_full, size=self.seg_out_size, mode="bilinear", align_corners=False)
        seg_logits_squeezed = seg_logits.squeeze(1)
        out = self.qwen(**batch, return_dict=True)
        vqa_loss = out.loss
        vqa_logits = out.logits
        return {"vqa_loss": vqa_loss, "vqa_logits": vqa_logits, "seg_logits": seg_logits_squeezed}

def run_evaluation(
    model: QwenVLWithSegmentation,
    processor: AutoProcessor,
    data_loader: DataLoader,
    device: torch.device,
    description: str = "Evaluating",
):
    model.eval()
    vlm_correct = 0
    total_samples = 0
    total_vqa_loss_sum = 0.0
    total_seg_loss_sum = 0.0
    total_loss_count = 0
    total_iou = 0.0
    seg_loss_fn_iou = JaccardLoss(reduction="mean").to(device)
    seg_loss_fn_bce = nn.BCEWithLogitsLoss(reduction="mean").to(device)
    debug_printed = False

    with torch.no_grad():
        for images, masks_gt, questions, answers, has_tumors in tqdm(data_loader, desc=description):
            masks_gt = masks_gt.to(device)
            has_tumors = has_tumors.to(device)
            B = len(answers)
            prompt_messages_list = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}] for q in questions]
            prompts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in prompt_messages_list]
            gen_inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
            generated_ids = model.qwen.generate(
                **gen_inputs, max_new_tokens=25, do_sample=False, num_beams=1,
                pad_token_id=processor.tokenizer.pad_token_id, eos_token_id=processor.tokenizer.eos_token_id,
            )
            generated_ids_trimmed = [g_ids[len(i_ids):] for i_ids, g_ids in zip(gen_inputs.input_ids, generated_ids)]
            decoded_spans = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for i in range(B):
                pred_span = decoded_spans[i].strip().lower()
                true_answer = answers[i]
                is_correct = False
                if "no tumor" in true_answer.lower():
                    if "no tumor" in pred_span and "one" not in pred_span and "two" not in pred_span: is_correct = True
                else:
                    want_two = "two" in true_answer.lower()
                    has_one, has_two = _has_one_two_flags(pred_span)
                    if (want_two and has_two and not has_one) or ((not want_two) and has_one and not has_two): is_correct = True
                if not debug_printed:
                    raw_full = processor.batch_decode(generated_ids, skip_special_tokens=True)[i]
                    print(f"\n[DEBUG Qwen Generation]\n  Prompt:\n {prompts[i]}\n  pred_raw:\n {raw_full}\n  pred_span:\n {pred_span}\n  true:\n {true_answer}\n  is_correct: {is_correct}")
                    debug_printed = True
                if is_correct: vlm_correct += 1
            total_samples += B

            batch_cpu = build_training_batch_cpu_main(images=images, masks=masks_gt.cpu(), questions=questions, answers=answers, has_tumors=has_tumors.cpu(), processor=processor)
            batch_gpu = _to_device(batch_cpu, device)
            with autocast(enabled=True, dtype=torch.float16):
                outputs = model(**batch_gpu)
                vqa_loss = outputs["vqa_loss"]
                seg_logits = outputs["seg_logits"]
                gt_masks_squeezed = batch_gpu["seg_masks_gt"].squeeze(1)
                seg_loss_iou = seg_loss_fn_iou(seg_logits, gt_masks_squeezed)
                seg_loss_bce = seg_loss_fn_bce(seg_logits, gt_masks_squeezed)
                seg_loss = (0.5 * seg_loss_bce) + (0.5 * seg_loss_iou)
            if vqa_loss is not None and torch.isfinite(vqa_loss): total_vqa_loss_sum += vqa_loss.item()
            if seg_loss is not None and torch.isfinite(seg_loss): total_seg_loss_sum += seg_loss.item()
            total_loss_count += 1
            total_iou += compute_iou_batch(seg_logits, gt_masks_squeezed)

    vlm_acc = (vlm_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    avg_vqa_loss = total_vqa_loss_sum / total_loss_count if total_loss_count > 0 else float("inf")
    avg_seg_loss = total_seg_loss_sum / total_loss_count if total_loss_count > 0 else float("inf")
    avg_iou = total_iou / total_loss_count if total_loss_count > 0 else 0.0
    ppl = math.exp(avg_vqa_loss) if avg_vqa_loss < 50 else float("inf")

    print(f"\n--- Results for {description} ---")
    print(f"  - VLM Accuracy (QA):              {vlm_acc:.2f}%")
    print(f"  - Perplexity (teacher-forced):    {ppl:.4f}")
    print(f"  - Segmentation IoU:               {avg_iou:.4f}")
    print(f"  - Avg VQA Loss:                   {avg_vqa_loss:.4f}")
    print(f"  - Avg Segmentation Loss (BCE+IoU):{avg_seg_loss:.4f}")
    print("-" * 40)
    return vlm_acc, avg_iou

def discover_lora_targets(model, include_vision: bool = True) -> List[str]:
    text_keys = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    vision_keys = {"q_proj", "k_proj", "v_proj", "out_proj"}
    target_suffixes = set()
    for name, module in model.named_modules():
        is_linear = isinstance(module, (nn.Linear, nn.Conv2d))
        is_lora = hasattr(module, 'base_layer')
        if is_linear or is_lora:
            name_suffix = name.split(".")[-1]
            if any(key == name_suffix for key in text_keys):
                if "lora_" not in name:
                    target_suffixes.add(name_suffix)
            if include_vision and "visual" in name and any(key == name_suffix for key in vision_keys):
                if "lora_" not in name:
                    target_suffixes.add(name_suffix)
    if not target_suffixes:
        print("Warning: No LoRA targets found; defaulting to common text_keys.")
        return sorted(list(text_keys))
    return sorted(list(target_suffixes))



if __name__ == "__main__":
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        preferred = 2
        gpu_index = preferred if (n_gpus > preferred) and (preferred >= 0) else 0
        DEVICE = torch.device(f"cuda:{gpu_index}")
    else:
        DEVICE = torch.device("cpu")
    print(f"Using device: {DEVICE}")

    config = {
        "device": str(DEVICE),
        "base_path": "/home/ealam/Downloads/LGG dataset Cameron/lgg-mri-segmentation/kaggle_3m",
        "local_qwen_path": "./saved_model",
        "csv_path": "/home/ealam/Downloads/LGG dataset Cameron/lgg-mri-segmentation/kaggle_3m/data.csv",
        
       
        "hint_segmentation_model_path": "best_model_segmentation_v2.pth", 
        
       
        "save_path": "./qwen7b_text_guidance_vlm204.1", 

        "learning_rate": 1e-4,
        "batch_size": 2,
        "num_epochs": 25,
        "early_stopping_patience": 5,
        "seed": 42,
        "seg_loss_weight": 3.0,
        "tumor_seg_loss_weight": 1.0,
        "include_vision_lora": True,
        "num_workers": 0,
        "grad_clip_val": 1.0,
    }


    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    
    print("Step 1: Loading EXTERNAL segmentation model (for text hints)...")
    if not os.path.exists(config['hint_segmentation_model_path']):
        raise FileNotFoundError(f"Hint segmentation model not found at: {config['hint_segmentation_model_path']}")
    
    hint_seg_model = get_external_segmentation_model()
    hint_seg_model.load_state_dict(torch.load(config['hint_segmentation_model_path'], map_location=DEVICE))
    hint_seg_model.to(DEVICE).eval() # Set to eval mode
    hint_seg_transform = get_external_segmentation_transforms()
    print("Hint model loaded successfully.")

 
    print("\nStep 2: Gathering and splitting data...")
    all_image_paths = [
        p.replace("_mask.tif", ".tif")
        for p in glob.glob(os.path.join(config["base_path"], "**", "*_mask.tif"), recursive=True)
    ]
    all_image_paths = [p for p in all_image_paths if os.path.exists(p)]
    if not all_image_paths:
        raise FileNotFoundError(f"No images found at {config['base_path']}.")

    print(f"Found {len(all_image_paths)} total images with masks.")
    train_val_paths, test_paths = train_test_split(
        all_image_paths, test_size=0.20, random_state=config["seed"]
    )
    train_paths, val_paths = train_test_split(
        train_val_paths, test_size=0.20, random_state=config["seed"]
    )
    print(f"Splits -> Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    
    print("\nStep 3: Setting up Qwen2.5-VL model and processor...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["local_qwen_path"],
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(config["local_qwen_path"])

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        base_model.resize_token_embeddings(len(processor.tokenizer))
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = processor.tokenizer.pad_token_id

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
    peft_model.print_trainable_parameters()

    if config["include_vision_lora"]:
        for name, p in peft_model.named_parameters():
            if "visual" in name and "lora_" in name:
                p.requires_grad = True

  
    multitask_model = QwenVLWithSegmentation(peft_model).to(DEVICE)

   
    print("\nStep 4: Preparing DataLoaders (with text hints)...")
    try:
        metadata_df = pd.read_csv(config["csv_path"])
    except Exception as e:
        raise FileNotFoundError(f"Failed to read metadata CSV at {config['csv_path']}: {e}")

    
    train_ds = VLM_QASegDataset_TextHint(
        train_paths, metadata_df, hint_seg_model, hint_seg_transform, DEVICE, is_train=True
    )
    val_ds = VLM_QASegDataset_TextHint(
        val_paths, metadata_df, hint_seg_model, hint_seg_transform, DEVICE, is_train=False
    )
    test_ds = VLM_QASegDataset_TextHint(
        test_paths, metadata_df, hint_seg_model, hint_seg_transform, DEVICE, is_train=False
    )
    
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Training or Validation dataset is empty. Check paths and metadata.")

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=vlm_seg_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=vlm_seg_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=vlm_seg_collate_fn,
    )

    
    print("\nStep 5: Starting multitask fine-tuning (with text hints)...")

    trainable_params = [p for p in multitask_model.parameters() if p.requires_grad]
    print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    optimizer = AdamW(trainable_params, lr=config["learning_rate"])
    
    seg_loss_fn_iou = JaccardLoss(reduction="none").to(DEVICE)
    seg_loss_fn_bce = nn.BCEWithLogitsLoss(reduction="none").to(DEVICE)

    num_training_steps = len(train_loader) * config["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )
    
    scaler = GradScaler() 
    best_val_metric = 0.0
    patience = 0

    for epoch in range(config["num_epochs"]):
        multitask_model.train()
        total_loss = 0.0
        total_vqa_loss_epoch = 0.0
        total_seg_loss_epoch = 0.0
        steps_in_epoch = 0

        
        for images, masks, questions, answers, has_tumors in tqdm(
            train_loader, desc=f"Training Epoch {epoch+1}"
        ):
            batch_cpu = build_training_batch_cpu_main(
                images=images, # Raw PILs
                masks=masks,     # Ground Truth masks
                questions=questions, # Qs with text hints
                answers=answers,
                has_tumors=has_tumors,
                processor=processor,
            )
            batch_gpu = _to_device(batch_cpu, DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=True, dtype=torch.float16):
                outputs = multitask_model(**batch_gpu)
                vqa_loss = outputs["vqa_loss"]
                seg_logits = outputs["seg_logits"] # Prediction from raw image
                
                gt_masks = batch_gpu["seg_masks_gt"].squeeze(1) # Ground Truth
                
                per_sample_iou_loss = seg_loss_fn_iou(seg_logits, gt_masks)
                per_sample_bce_loss = seg_loss_fn_bce(seg_logits, gt_masks).mean(dim=(1, 2))
                per_sample_seg_loss = (0.5 * per_sample_bce_loss) + (0.5 * per_sample_iou_loss)
                
                weights = torch.ones_like(per_sample_seg_loss, device=DEVICE)
                weights[batch_gpu["has_tumor"]] = config["tumor_seg_loss_weight"]
                weighted_seg_loss = (per_sample_seg_loss * weights).mean()

                combined_loss = vqa_loss + config["seg_loss_weight"] * weighted_seg_loss

            combined_loss_float32 = combined_loss.float()

            if not torch.isfinite(combined_loss_float32):
                vqa_val = vqa_loss.item() if torch.is_tensor(vqa_loss) and torch.isfinite(vqa_loss) else float('inf')
                seg_val = weighted_seg_loss.item() if torch.is_tensor(weighted_seg_loss) and torch.isfinite(weighted_seg_loss) else float('nan')
                print(f"Warning: non-finite loss detected (VQA: {vqa_val}, Seg: {seg_val}), skipping step.")
                continue

            scaler.scale(combined_loss_float32).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, config["grad_clip_val"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += combined_loss_float32.item()
            if vqa_loss is not None:
                total_vqa_loss_epoch += vqa_loss.item()
            if weighted_seg_loss is not None:
                total_seg_loss_epoch += weighted_seg_loss.item()
            steps_in_epoch += 1

        avg_train_loss = total_loss / max(1, steps_in_epoch)
        avg_vqa_train_loss = total_vqa_loss_epoch / max(1, steps_in_epoch)
        avg_seg_train_loss = total_seg_loss_epoch / max(1, steps_in_epoch)
        
        print(f"\nEpoch {epoch+1} Avg Combined Loss -> {avg_train_loss:.4f} (VQA: {avg_vqa_train_loss:.4f}, Seg: {avg_seg_train_loss:.4f})")

        # Validation
        val_acc, val_iou = run_evaluation(
            multitask_model,
            processor,
            val_loader,
            DEVICE,
            description="Validation Set Eval",
        )
        current_metric = val_acc + (val_iou * 100.0)

        if current_metric > best_val_metric:
            print(f"  -> New best validation metric ({current_metric:.2f}). Saving model...")
            best_val_metric = current_metric
            patience = 0
            save_dir = config["save_path"]
            os.makedirs(save_dir, exist_ok=True)
            torch.save(multitask_model.state_dict(), os.path.join(save_dir, "multitask_model.pth"))
            multitask_model.qwen.save_pretrained(os.path.join(save_dir, "qwen_lora"))
            processor.save_pretrained(os.path.join(save_dir, "processor"))
        else:
            patience += 1
            print(f"  -> No improvement for {patience} epoch(s).")
            if patience >= config["early_stopping_patience"]:
                print("\n--- Early stopping triggered. ---")
                break
        print("=" * 80)

 
    print("\nStep 5: Loading best model for final evaluation...")
    save_path = config["save_path"]
    multitask_model_path = os.path.join(save_path, "multitask_model.pth")

    if os.path.exists(multitask_model_path) and len(test_loader) > 0:
        print("Reloading model from scratch for final test...")
        final_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config["local_qwen_path"],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        final_processor = AutoProcessor.from_pretrained(os.path.join(save_path, "processor"))
        
        if final_processor.tokenizer.pad_token is None:
            final_processor.tokenizer.pad_token = final_processor.tokenizer.eos_token
            final_base.config.pad_token_id = final_processor.tokenizer.pad_token_id

        final_peft = PeftModel.from_pretrained(final_base, os.path.join(save_path, "qwen_lora")).to(DEVICE)
        final_multitask_model = QwenVLWithSegmentation(final_peft).to(DEVICE)
        
        final_multitask_model.load_state_dict(torch.load(multitask_model_path, map_location=DEVICE))

        run_evaluation(
            final_multitask_model,
            final_processor,
            test_loader,
            DEVICE,
            description="Final Test Evaluation",
        )
    else:
        print("Skipping final test evaluation (no model saved or no test data).")

    print("\n--- Multitask Qwen2.5-VL + Text Hint experiment complete. ---")
