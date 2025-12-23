import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# Configuration
# ==========================================
HF_DATASET_ID = "prithivMLmods/IndoorOutdoorNet-20K"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 2
CLASSES = ['Indoor', 'Outdoor']
MODEL_PATH = "models/indoor_outdoor_efficientnet_finetuned.keras"
LOG_PHASE1 = "training_log_phase1.csv"
LOG_PHASE2 = "training_log_phase2.csv"
OUTPUT_DIR = "evaluation_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ensure reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def configure_gpu():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU Error: {e}")

configure_gpu()

def get_test_dataset():
    print(f"Loading dataset for evaluation...")
    ds = load_dataset(HF_DATASET_ID)
    
    # EXACT SPLIT LOGIC FROM TRAINING SCRIPT
    if 'train' in ds and 'test' not in ds:
        split1 = ds['train'].train_test_split(test_size=0.1, seed=42, stratify_by_column="label")
        ds_test = split1['test']
    elif 'train' in ds and 'validation' in ds and 'test' in ds:
         ds_test = ds['test']
    else:
        full_ds = ds['train'] if 'train' in ds else ds[list(ds.keys())[0]]
        split1 = full_ds.train_test_split(test_size=0.2, seed=42)
        ds_test = split1['test']

    def generator():
        for sample in ds_test:
            img = sample['image'].convert("RGB")
            img = np.array(img)
            yield img, sample['label']
    
    # We return numpy arrays for sklearn easiest handling (if fits in memory) 
    # OR we use batched prediction. 2000 images is small enough for memory usually.
    # Let's use batched prediction to be safe but gather labels.
    
    print("Preparing test data...")
    images = []
    labels = []
    for sample in ds_test:
        img = sample['image'].convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        images.append(np.array(img))
        labels.append(sample['label'])
    
    return np.array(images), np.array(labels)

def plot_training_history():
    print("Plotting training history...")
    if not os.path.exists(LOG_PHASE1) or not os.path.exists(LOG_PHASE2):
        print("Log files not found. Skipping history plot.")
        return

    df1 = pd.read_csv(LOG_PHASE1)
    df2 = pd.read_csv(LOG_PHASE2)
    
    # Adjust epochs for Phase 2
    df2['epoch'] = df2['epoch'] + df1['epoch'].iloc[-1] + 1
    
    combined_acc = pd.concat([df1['accuracy'], df2['accuracy']]).reset_index(drop=True)
    combined_val_acc = pd.concat([df1['val_accuracy'], df2['val_accuracy']]).reset_index(drop=True)
    combined_loss = pd.concat([df1['loss'], df2['loss']]).reset_index(drop=True)
    combined_val_loss = pd.concat([df1['val_loss'], df2['val_loss']]).reset_index(drop=True)
    
    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(combined_acc, label='Train Acc')
    plt.plot(combined_val_acc, label='Val Acc')
    plt.axvline(x=len(df1)-0.5, color='gray', linestyle='--', label='Fine-tuning Start')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(combined_loss, label='Train Loss')
    plt.plot(combined_val_loss, label='Val Loss')
    plt.axvline(x=len(df1)-0.5, color='gray', linestyle='--', label='Fine-tuning Start')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history_combined.png'))
    print(f"Saved training history to {OUTPUT_DIR}/training_history_combined.png")

def evaluate_model():
    X_test, y_true = get_test_dataset()
    
    print(f"Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)
    
    print("Predicting...")
    # Evaluate inputs: EfficientNet usually expects [0-255]. 
    preds = model.predict(X_test, batch_size=BATCH_SIZE)
    y_pred = np.argmax(preds, axis=1)
    
    # Classification Report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    print(report)
    
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
        
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    print(f"Saved confusion matrix to {OUTPUT_DIR}/confusion_matrix.png")

def main():
    try:
        plot_training_history()
        evaluate_model()
        print("\nEvaluation Complete!")
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
