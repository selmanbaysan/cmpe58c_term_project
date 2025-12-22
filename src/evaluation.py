import os
import time
import numpy as np
import tensorflow as tf
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
from src.utils import preprocess_image, preprocess_image_for_hf

# Constants
TEST_DIR = 'data/test'
LOCAL_MODEL_PATH = 'models/indoor_outdoor_efficientnet.h5'
HF_MODEL_NAME = "prithivMLmods/IndoorOutdoorNet"
CLASSES = ['Indoor', 'Outdoor']

def evaluate_models():
    print("Loading Local Model...")
    local_model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
    
    print(f"Loading HF Model: {HF_MODEL_NAME}...")
    hf_processor = AutoImageProcessor.from_pretrained(HF_MODEL_NAME)
    hf_model = SiglipForImageClassification.from_pretrained(HF_MODEL_NAME)
    
    y_true = []
    y_pred_local = []
    y_pred_hf = []
    
    # Iterate over test directory
    print("\nStarting Evaluation...")
    for label_name in CLASSES:
        label_idx = 0 if label_name == 'Indoor' else 1
        class_dir = os.path.join(TEST_DIR, label_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found. Skipping.")
            continue
            
        image_files = os.listdir(class_dir)
        print(f"Processing {len(image_files)} images for class {label_name}...")
        
        for img_file in tqdm(image_files):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(class_dir, img_file)
            
            # --- Ground Truth ---
            y_true.append(label_idx)
            
            # --- Local Model Inference ---
            try:
                img_tensor_local = preprocess_image(img_path)
                pred_prob_local = local_model.predict(img_tensor_local, verbose=0)[0][0]
                pred_local = 1 if pred_prob_local > 0.5 else 0
                y_pred_local.append(pred_local)
            except Exception as e:
                print(f"Error local inference on {img_path}: {e}")
                y_pred_local.append(-1) # Error flag

            # --- HF Model Inference ---
            try:
                inputs_hf = preprocess_image_for_hf(img_path, hf_processor)
                with torch.no_grad():
                    outputs_hf = hf_model(**inputs_hf)
                    logits = outputs_hf.logits
                    probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
                    # HF Model: 0 -> Indoor, 1 -> Outdoor (Usually, verifying below)
                    # The HF model card says: {"0": "Indoor", "1": "Outdoor"}
                    pred_hf = np.argmax(probs)
                    y_pred_hf.append(pred_hf)
            except Exception as e:
                print(f"Error HF inference on {img_path}: {e}")
                y_pred_hf.append(-1)

    y_true = np.array(y_true)
    y_pred_local = np.array(y_pred_local)
    y_pred_hf = np.array(y_pred_hf)
    
    # Filter out errors
    valid_mask = (y_pred_local != -1) & (y_pred_hf != -1)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Images Evaluated: {np.sum(valid_mask)}")
    
    # Local Model Metrics
    print("\n--- LOCAL MODEL (EfficientNet) ---")
    print(f"Accuracy: {accuracy_score(y_true[valid_mask], y_pred_local[valid_mask]):.4f}")
    print("Classification Report:")
    print(classification_report(y_true[valid_mask], y_pred_local[valid_mask], target_names=CLASSES))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true[valid_mask], y_pred_local[valid_mask]))
    
    # HF Model Metrics
    print("\n--- HF MODEL (IndoorOutdoorNet) ---")
    print(f"Accuracy: {accuracy_score(y_true[valid_mask], y_pred_hf[valid_mask]):.4f}")
    print("Classification Report:")
    print(classification_report(y_true[valid_mask], y_pred_hf[valid_mask], target_names=CLASSES))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true[valid_mask], y_pred_hf[valid_mask]))

if __name__ == "__main__":
    evaluate_models()
