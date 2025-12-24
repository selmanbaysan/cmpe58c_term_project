import os
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoImageProcessor, SiglipForImageClassification
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================
HF_DATASET_ID = "prithivMLmods/IndoorOutdoorNet-20K"
HF_MODEL_NAME = "prithivMLmods/IndoorOutdoorNet"
OUTPUT_DIR = "evaluation_results_hf"
BATCH_SIZE = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Classes in the dataset are [0: Indoor, 1: Outdoor] based on previous checks.
# The HF model config likely follows this, but we should verify via config.id2label if possible.
# app.py uses CLASSES = ['Indoor', 'Outdoor'] for this model too.

def get_test_dataset():
    print(f"Loading dataset {HF_DATASET_ID}...")
    ds = load_dataset(HF_DATASET_ID)
    
    if 'train' in ds and 'test' not in ds:
        # train_indoor_outdoor.py usage:
        # split1 = ds['train'].train_test_split(test_size=0.1, seed=42, stratify_by_column="label") 
        # ds_test = split1['test']
        print("Splitting train sets to isolate Test set...")
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

def evaluate_hf_model():
    # Setup Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Model & Processor
    print(f"Loading model {HF_MODEL_NAME}...")
    processor = AutoImageProcessor.from_pretrained(HF_MODEL_NAME)
    model = SiglipForImageClassification.from_pretrained(HF_MODEL_NAME)
    model.to(device)
    model.eval()

    id2label = model.config.id2label
    print(f"Model id2label: {id2label}")
    
    ds_test = get_test_dataset()
    
    true_labels = []
    pred_labels = []
    
    print("Running predictions...")
    
    batch_images = []
    batch_labels = []
    
    for i, sample in tqdm(enumerate(ds_test), total=len(ds_test)):
        image = sample['image'].convert("RGB")
        label = sample['label']
        
        batch_images.append(image)
        batch_labels.append(label)
        
        if len(batch_images) == BATCH_SIZE or i == len(ds_test) - 1:
            # Prepare batch
            inputs = processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            pred_labels.extend(preds)
            true_labels.extend(batch_labels)
            
            batch_images = []
            batch_labels = []

    # Generate Metrics
    target_names = ["Indoor", "Outdoor"] # Assumption based on id2label {0: Indoor, 1: Outdoor}
    
    print("\nClassification Report:")
    report = classification_report(true_labels, pred_labels, target_names=target_names)
    print(report)
    
    with open(os.path.join(OUTPUT_DIR, 'classification_report_hf.txt'), 'w') as f:
        f.write(report)
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix (HF Model)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_hf.png'))
    print(f"Saved confusion matrix to {OUTPUT_DIR}/confusion_matrix_hf.png")

if __name__ == "__main__":
    evaluate_hf_model()
