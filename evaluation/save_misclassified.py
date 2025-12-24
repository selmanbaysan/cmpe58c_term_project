import os
import numpy as np
import tensorflow as tf
import torch
import cv2
from datasets import load_dataset
from transformers import AutoImageProcessor, SiglipForImageClassification
from tqdm import tqdm
from PIL import Image

# ==========================================
# Configuration
# ==========================================
LOCAL_MODEL_PATH = "models/indoor_outdoor_efficientnet_finetuned.keras"
HF_MODEL_NAME = "prithivMLmods/IndoorOutdoorNet"
HF_DATASET_ID = "prithivMLmods/IndoorOutdoorNet-20K"
OUTPUT_DIR = "false_predictions"
IMG_SIZE = 224

# Ensure output directories exist
os.makedirs(f"{OUTPUT_DIR}/local", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/hf", exist_ok=True)

# Classes
CLASSES = ['Indoor', 'Outdoor']

def get_test_dataset():
    print(f"Loading dataset {HF_DATASET_ID}...")
    ds = load_dataset(HF_DATASET_ID)
    
    # Replicate split logic
    if 'train' in ds and 'test' not in ds:
        print("Splitting 'train' into Train/Val/Test...")
        split1 = ds['train'].train_test_split(test_size=0.1, seed=42, stratify_by_column="label")
        ds_test = split1['test']
    elif 'train' in ds and 'validation' in ds and 'test' in ds:
         ds_test = ds['test']
    else:
        full_ds = ds['train'] if 'train' in ds else ds[list(ds.keys())[0]]
        split1 = full_ds.train_test_split(test_size=0.2, seed=42)
        ds_test = split1['test']
        
    print(f"Test set size: {len(ds_test)}")
    return ds_test

def main():
    # 1. Load Data
    ds_test = get_test_dataset()
    
    # 2. Load Local Model
    print("Loading Local Model...")
    # Force CPU to avoid any potential MPS conflicts during mixed usage if needed, 
    # but separate calls usually fine. Let's use default (MPS if avail by env).
    local_model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
    
    # 3. Load HF Model
    print("Loading HF Model...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(HF_MODEL_NAME)
    hf_model = SiglipForImageClassification.from_pretrained(HF_MODEL_NAME)
    hf_model.to(device)
    hf_model.eval()
    
    print("Processing Test Set...")
    
    local_errors = 0
    hf_errors = 0
    
    for i, sample in tqdm(enumerate(ds_test), total=len(ds_test)):
        # Original PIL Image and Label
        pil_img = sample['image'].convert("RGB")
        true_label_idx = sample['label']
        true_label_str = CLASSES[true_label_idx]
        
        # --- Local Model Prediction ---
        # Preprocess
        img_np = np.array(pil_img)
        img_resized = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
        img_resized = img_resized.astype("float32")
        img_expanded = np.expand_dims(img_resized, axis=0)
        
        # Predict
        local_preds = local_model.predict(img_expanded, verbose=0)
        local_pred_idx = np.argmax(local_preds, axis=1)[0]
        local_pred_str = CLASSES[local_pred_idx]
        local_conf = local_preds[0][local_pred_idx]
        
        if local_pred_idx != true_label_idx:
            local_errors += 1
            filename = f"idx{i}_True{true_label_str}_Pred{local_pred_str}_Conf{local_conf:.2f}.jpg"
            pil_img.save(os.path.join(OUTPUT_DIR, "local", filename))
            
        # --- HF Model Prediction ---
        inputs = processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = hf_model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            hf_pred_idx = torch.argmax(probs, dim=1).item()
            hf_conf = probs[0][hf_pred_idx].item()
            
        hf_pred_str = CLASSES[hf_pred_idx]
        
        if hf_pred_idx != true_label_idx:
            hf_errors += 1
            filename = f"idx{i}_True{true_label_str}_Pred{hf_pred_str}_Conf{hf_conf:.2f}.jpg"
            pil_img.save(os.path.join(OUTPUT_DIR, "hf", filename))
            
    print("\n--- Summary ---")
    print(f"Test Size: {len(ds_test)}")
    print(f"Local Model Errors: {local_errors} saved to {OUTPUT_DIR}/local/")
    print(f"HF Model Errors:    {hf_errors} saved to {OUTPUT_DIR}/hf/")

if __name__ == "__main__":
    main()
