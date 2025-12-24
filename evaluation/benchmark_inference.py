import time
import os
import numpy as np
import tensorflow as tf
import torch
import cv2
from transformers import AutoImageProcessor, SiglipForImageClassification

# Disabling GPU for TF globally if possible, or effectively hiding it
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ==========================================
# Configuration
# ==========================================
LOCAL_MODEL_PATH = "models/indoor_outdoor_efficientnet_finetuned.keras"
HF_MODEL_NAME = "prithivMLmods/IndoorOutdoorNet"
IMG_SIZE = 224
NUM_ITERATIONS = 50
WARMUP_ITERATIONS = 10

def benchmark_local_model():
    print(f"\n--- Benchmarking Local Model Pipeline ({LOCAL_MODEL_PATH}) on CPU ---")
    try:
        # Force CPU context
        with tf.device('/CPU:0'):
            model = tf.keras.models.load_model(LOCAL_MODEL_PATH)
            
            # Simulate raw image
            raw_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            
            print("Warming up...")
            for _ in range(WARMUP_ITERATIONS):
                img = cv2.resize(raw_image, (IMG_SIZE, IMG_SIZE))
                img = img.astype("float32")
                img_expanded = np.expand_dims(img, axis=0)
                model.predict(img_expanded, verbose=0)
                
            print(f"Running {NUM_ITERATIONS} iterations (inc. Preprocessing)...")
            start_time = time.time()
            for _ in range(NUM_ITERATIONS):
                img = cv2.resize(raw_image, (IMG_SIZE, IMG_SIZE))
                img = img.astype("float32")
                img_expanded = np.expand_dims(img, axis=0)
                model.predict(img_expanded, verbose=0)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / NUM_ITERATIONS
            print(f"Local Pipeline (CPU) Average Time: {avg_time*1000:.2f} ms")
            return avg_time
    except Exception as e:
        print(f"Local Benchmark Failed: {e}")
        return None

def benchmark_hf_model():
    print(f"\n--- Benchmarking HF Model Pipeline ({HF_MODEL_NAME}) on CPU ---")
    try:
        device = "cpu"
        print(f"Using device: {device}")
        
        processor = AutoImageProcessor.from_pretrained(HF_MODEL_NAME)
        model = SiglipForImageClassification.from_pretrained(HF_MODEL_NAME)
        model.to(device)
        model.eval()
        
        # Simulate raw image
        raw_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        print("Warming up...")
        with torch.no_grad():
            for _ in range(WARMUP_ITERATIONS):
                inputs = processor(images=raw_image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model(**inputs)
                
        print(f"Running {NUM_ITERATIONS} iterations (inc. Preprocessing)...")
        start_time = time.time()
        with torch.no_grad():
            for _ in range(NUM_ITERATIONS):
                inputs = processor(images=raw_image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model(**inputs)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / NUM_ITERATIONS
        print(f"HF Pipeline (CPU) Average Time:    {avg_time*1000:.2f} ms")
        return avg_time
    except Exception as e:
        print(f"HF Benchmark Failed: {e}")
        return None

def main():
    print("Starting CPU Benchmark...")
    
    local_time = benchmark_local_model()
    hf_time = benchmark_hf_model()
    
    print("\n==================================")
    print("FINAL RESULTS (CPU)")
    print("==================================")
    
    if local_time:
        print(f"Fine-Tuned Pipeline: {local_time*1000:.2f} ms")
    if hf_time:
        print(f"HF Siglip Pipeline:  {hf_time*1000:.2f} ms")
        
    if local_time and hf_time:
        if local_time < hf_time:
            speedup = hf_time / local_time
            print(f"\nResult: Fine-Tuned Model is {speedup:.2f}x FASTER (CPU).")
        else:
            speedup = local_time / hf_time
            print(f"\nResult: HF Siglip is {speedup:.2f}x FASTER (CPU).")

if __name__ == "__main__":
    main()
