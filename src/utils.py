import os
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_test_data(test_dir):
    """
    Loads the test dataset from the specified directory using TensorFlow's image_dataset_from_directory.
    
    Args:
        test_dir (str): Path to the test directory (containing 'Indoor' and 'Outdoor' subfolders).
        
    Returns:
        tf.data.Dataset: The test dataset.
    """
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary',
        shuffle=False # Identifying correct order is important for evaluation
    )
    return test_ds

def preprocess_image(image_path):
    """
    Preprocesses a single image for inference with the TensorFlow model.
    """
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    return img_array

def preprocess_image_for_hf(image_path_or_array, processor):
    """
    Preprocesses an image for the Hugging Face model using its processor.
    Accepts specific path or numpy array (from cv2/streamlit).
    """
    if isinstance(image_path_or_array, str):
        image = Image.open(image_path_or_array).convert("RGB")
    elif isinstance(image_path_or_array, np.ndarray):
        image = Image.fromarray(image_path_or_array).convert("RGB")
    else:
         image = image_path_or_array.convert("RGB")
         
    inputs = processor(images=image, return_tensors="pt")
    return inputs
