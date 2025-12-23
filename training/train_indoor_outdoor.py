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
CLASSES = ['Indoor', 'Outdoor'] # Assuming 0: Indoor, 1: Outdoor order, will verify from dataset features if possible
INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 10
LEARNING_RATE_INITIAL = 1e-2
LEARNING_RATE_FINE_TUNE = 1e-5
MODEL_SAVE_PATH = "models/indoor_outdoor_efficientnet_finetuned.keras"
LOG_FILE = "training_log.csv"

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

def main():
    print(f"Loading dataset: {HF_DATASET_ID}...")
    # Load dataset from Hugging Face
    # The dataset is expected to have 'image' and 'label' columns
    ds = load_dataset(HF_DATASET_ID)
    
    # Check dataset structure
    print("Dataset structure:")
    print(ds)
    
    # Combine if splits are not standard or if we want to ensure our own clean split
    # If the dataset already has train/test/val, we can use them, but user requested:
    # "make sure train and test split does not leaked" -> defaulting to re-splitting a single source if needed.
    # Usually simple HF datasets come with 'train'.
    
    if 'train' in ds and 'test' not in ds:
        print("Splitting 'train' into Train/Val/Test...")
        # Split Train -> 90% Train+Val, 10% Test
        split1 = ds['train'].train_test_split(test_size=0.1, seed=42, stratify_by_column="label")
        ds_test = split1['test']
        
        # Split Train+Val -> 89% Train, 11% Val (approx 80/10/10 overall)
        # 0.9 * 0.111 = 0.1
        split2 = split1['train'].train_test_split(test_size=0.1111, seed=42, stratify_by_column="label") 
        ds_train = split2['train']
        ds_val = split2['test']
    elif 'train' in ds and 'validation' in ds and 'test' in ds:
         print("Using existing splits (verifying no overlap)...")
         ds_train = ds['train']
         ds_val = ds['validation']
         ds_test = ds['test']
    else:
        # Fallback generic split
        print("Generic splitting...")
        full_ds = ds['train'] if 'train' in ds else ds[list(ds.keys())[0]]
        split1 = full_ds.train_test_split(test_size=0.2, seed=42)
        ds_test = split1['test']
        split2 = split1['train'].train_test_split(test_size=0.1, seed=42) # 10% of remaining 80%
        ds_val = split2['test']
        ds_train = split2['train']

    print(f"Train size: {len(ds_train)}")
    print(f"Val size: {len(ds_val)}")
    print(f"Test size: {len(ds_test)}")

    # ==========================================
    # Preprocessing
    # ==========================================
    def preprocess_image(examples):
        # Convert PIL images to RGB and resize
        images = [img.convert("RGB").resize((IMG_SIZE, IMG_SIZE)) for img in examples["image"]]
        # Convert to numpy array
        images = [np.array(img) for img in images]
        
        # Labels to integer
        labels = examples["label"]
        
        return {"pixel_values": images, "label": labels}

    # Use TF Data for efficient pipeline
    # We can use the huggingface to_tf_dataset, but manual mapping gives control over resizing if we did it in python.
    # However, doing resize in python via .map might be slow. 
    # Better approach: Use TF mapping.
    
    # Create generator
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
        
        # Transformations
        def process(image, label):
            image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
            # EfficientNet expects [0, 255] inputs as per documentation (it includes normalization layer).
            # So no /255.0 here.
            label = tf.one_hot(label, NUM_CLASSES)
            return image, label

        dataset = dataset.map(process, num_parallel_calls=tf.data.AUTOTUNE)
        
        if shuffle:
            dataset = dataset.shuffle(1000)
            
        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return dataset

    print("Creating TF datasets...")
    train_ds = create_tf_dataset(ds_train, shuffle=True)
    val_ds = create_tf_dataset(ds_val, shuffle=False)
    test_ds = create_tf_dataset(ds_test, shuffle=False)

    # ==========================================
    # Model Definition
    # ==========================================
    print("Building model...")
    # 1. Base Model
    # input_shape=(224, 224, 3)
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze base model
    base_model.trainable = False

    # 2. Rebuild Top
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization()(x)
    
    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    model = keras.Model(inputs=base_model.input, outputs=outputs, name="EfficientNet_FineTune")
    
    # Compile Phase 1
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE_INITIAL)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
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

    # ==========================================
    # Phase 1: Train Top Layers
    # ==========================================
    print(f"Phase 1: Training top layers for {INITIAL_EPOCHS} epochs...")
    history_phase1 = model.fit(
        train_ds,
        epochs=INITIAL_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks
    )

    # ==========================================
    # Phase 2: Fine-Tuning
    # ==========================================
    print("\nPhase 2: Fine-tuning...")
    print("Saving Phase 1 model and refreshing session to avoid MPS graph errors...")
    
    # Save temp model
    temp_model_path = "models/temp_phase1.keras"
    model.save(temp_model_path)
    
    # Clear session to free resources and reset graph state
    tf.keras.backend.clear_session()
    
    # Reload model
    print("Reloading model...")
    model = keras.models.load_model(temp_model_path)
    
    # Unfreeze specific layers
    # "We unfreeze the top 20 layers while leaving BatchNorm layers frozen"
    base_model_layer = model.layers[0] # EfficientNetB0 is the first layer in our Functional model if constructed this way?
    # Wait, in the original script: model = keras.Model(inputs=base_model.input, outputs=outputs)
    # So the layers of 'model' ARE the layers of base_model + top layers. 
    # Actually, EfficientNetB0 returns a Functional model, so 'base_model' is a model distinct from 'model'.
    # But when we did 'x = base_model.output', 'base_model' became part of the graph. 
    # Let's inspect model.layers to be safe. 
    # Typically efficientnet is a big graph of layers.
    
    # Let's rely on the fact that we can iterate model.layers.
    # However, since we reloaded, the 'base_model' variable is gone.
    # We need to find the layers to unfreeze.
    
    # Strategy: Unfreeze top 20 layers of the ENTIRE model, skipping BatchNormalization.
    model.trainable = True
    
    # Freeze all layers first
    for layer in model.layers:
        layer.trainable = False
        
    # Unfreeze top 20 non-BN layers
    # We iterate backwards
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

    # Recompile with lower learning rate
    optimizer_ft = keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE_TUNE)
    model.compile(
        optimizer=optimizer_ft,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary(print_fn=lambda x: None) # Suppress full summary, too long
    
    # Train
    print(f"Fine-tuning for {FINE_TUNE_EPOCHS} more epochs...")
    # Add to total epochs
    total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
    
    # Note: history_phase1 is lost from memory after clear_session unless we saved the data.
    # We can perform a new fit call. 'initial_epoch' is for logging/callbacks continuity.
    
    history_phase2 = model.fit(
        train_ds,
        epochs=total_epochs,
        initial_epoch=INITIAL_EPOCHS, 
        validation_data=val_ds,
        callbacks=callbacks
    )
    
    # ==========================================
    # Evaluation
    # ==========================================
    print("Evaluating on Test set...")
    loss, acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Loss: {loss:.4f}")
    
    # Plotting code (optional save)
    def plot_history(h1, h2):
        acc = h1.history['accuracy'] + h2.history['accuracy']
        val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
        loss = h1.history['loss'] + h2.history['loss']
        val_loss = h1.history['val_loss'] + h2.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.ylim([0.8, 1])
        plt.plot([len(h1.history['accuracy'])-1,len(h1.history['accuracy'])-1],
                  plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.ylim([0, 1.0])
        plt.plot([len(h1.history['loss'])-1,len(h1.history['loss'])-1],
                 plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.savefig('training_history_plot.png')
        print("Saved history plot to training_history_plot.png")

    plot_history(history_phase1, history_phase2)

if __name__ == "__main__":
    main()
