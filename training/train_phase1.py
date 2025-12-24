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
INITIAL_EPOCHS = 20
LEARNING_RATE_INITIAL = 1e-2
MODEL_PHASE1_PATH = "models/indoor_outdoor_phase1.keras"
LOG_FILE = "training_log_phase1.csv"

# Ensure reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def configure_gpu():
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
         print("Using existing splits...")
         ds_train = ds['train']
         ds_val = ds['validation']
         ds_test = ds['test']
    else:
        print("Generic splitting...")
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

    return create_tf_dataset(ds_train, shuffle=True), create_tf_dataset(ds_val, shuffle=False)

def main():
    train_ds, val_ds = get_datasets()

    print("Building model...")
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization()(x)
    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    model = keras.Model(inputs=base_model.input, outputs=outputs, name="EfficientNet_Phase1")
    
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE_INITIAL)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PHASE1_PATH,
            save_best_only=True,
            monitor="val_loss", 
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=3, 
            restore_best_weights=True
        ),
        keras.callbacks.CSVLogger(LOG_FILE)
    ]

    print(f"Phase 1: Training top layers for {INITIAL_EPOCHS} epochs...")
    model.fit(
        train_ds,
        epochs=INITIAL_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks
    )
    
if __name__ == "__main__":
    main()
