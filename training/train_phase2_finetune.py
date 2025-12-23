import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datasets import load_dataset
import matplotlib.pyplot as plt

# ==========================================
# Configuration / Hyperparameters
# ==========================================
HF_DATASET_ID = "prithivMLmods/IndoorOutdoorNet-20K"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 2
FINE_TUNE_EPOCHS = 10
LEARNING_RATE_FINE_TUNE = 1e-5
MODEL_PHASE1_PATH = "models/indoor_outdoor_phase1.keras"
MODEL_FINAL_PATH = "models/indoor_outdoor_efficientnet_finetuned.keras"
LOG_FILE = "training_log_phase2.csv"

# Ensure reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def configure_gpu():
    """Configures TensorFlow to use the GPU (MPS on Mac) if available."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU(s) Detected: {gpus}")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU(s) configured successfully.")
        else:
            print("No GPU detected. Training will run on CPU.")
    except RuntimeError as e:
        print(f"GPU Configuration Error: {e}")

configure_gpu()

def get_datasets():
    # ... Same dataset logic as Phase 1 ...
    # Ideally reuse code, but for simplicity of execution we copy.
    # We need same splits! Seed '42' ensures this.
    print(f"Loading dataset: {HF_DATASET_ID}...")
    ds = load_dataset(HF_DATASET_ID)
    
    if 'train' in ds and 'test' not in ds:
        print("Splitting 'train' into Train/Val/Test...")
        split1 = ds['train'].train_test_split(test_size=0.1, seed=42, stratify_by_column="label")
        ds_test = split1['test']
        split2 = split1['train'].train_test_split(test_size=0.1111, seed=42, stratify_by_column="label") 
        ds_train = split2['train']
        ds_val = split2['test']
    elif 'train' in ds and 'validation' in ds and 'test' in ds:
         ds_train = ds['train']
         ds_val = ds['validation']
         ds_test = ds['test']
    else:
        full_ds = ds['train'] if 'train' in ds else ds[list(ds.keys())[0]]
        split1 = full_ds.train_test_split(test_size=0.2, seed=42)
        ds_test = split1['test']
        split2 = split1['train'].train_test_split(test_size=0.1, seed=42) 
        ds_val = split2['test']
        ds_train = split2['train']

    def create_tf_dataset(hf_ds, shuffle=False):
        def generator():
            for sample in hf_ds:
                img = sample['image'].convert("RGB")
                img = np.array(img)
                yield img, sample['label']
        
        output_signature = (
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.int64)
        )
        
        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        
        def process(image, label):
            image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
            label = tf.one_hot(label, NUM_CLASSES)
            return image, label

        dataset = dataset.map(process, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            dataset = dataset.shuffle(1000)
        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return dataset

    return create_tf_dataset(ds_train, shuffle=True), create_tf_dataset(ds_val, shuffle=False), create_tf_dataset(ds_test, shuffle=False)

def main():
    train_ds, val_ds, test_ds = get_datasets()

    print(f"Loading Phase 1 model from {MODEL_PHASE1_PATH}...")
    model = keras.models.load_model(MODEL_PHASE1_PATH)
    
    print("Phase 2: Fine-tuning...")
    
    # Strategy: Unfreeze top 20 layers of the ENTIRE model, skipping BatchNormalization.
    model.trainable = True
    
    for layer in model.layers:
        layer.trainable = False
        
    unfreeze_count = 0
    for layer in reversed(model.layers):
        if unfreeze_count >= 20:
            break
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            unfreeze_count += 1
            
    print(f"Unfrozen {unfreeze_count} layers for fine-tuning.")

    optimizer_ft = keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE_TUNE)
    model.compile(
        optimizer=optimizer_ft,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary(print_fn=lambda x: None)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_FINAL_PATH,
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=4, # Slightly more patience for fine-tuning
            restore_best_weights=True
        ),
        keras.callbacks.CSVLogger(LOG_FILE)
    ]

    print(f"Fine-tuning for {FINE_TUNE_EPOCHS} epochs...")
    model.fit(
        train_ds,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks
    )
    
    print("Evaluating on Test set...")
    loss, acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Loss: {loss:.4f}")
    
    # Simple Plot combining log files would be done by reading CSVs if needed, 
    # but user can just look at logs.

if __name__ == "__main__":
    main()
