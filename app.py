import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
import tensorflow as tf
from src.utils import preprocess_image, preprocess_image_for_hf
# Optional: Load HF model if needed, or keep it lightweight by loading on demand
from transformers import AutoImageProcessor, SiglipForImageClassification
import torch

# Page Config
st.set_page_config(page_title="Indoor/Outdoor Classifier", layout="wide")

# Paths and Constants
LOCAL_MODEL_PATH = 'models/indoor_outdoor_efficientnet_finetuned.keras'
HF_MODEL_NAME = "prithivMLmods/IndoorOutdoorNet"
CLASSES = ['Indoor', 'Outdoor']

@st.cache_resource
def load_local_model():
    return tf.keras.models.load_model(LOCAL_MODEL_PATH)

@st.cache_resource
def load_hf_model():
    processor = AutoImageProcessor.from_pretrained(HF_MODEL_NAME)
    model = SiglipForImageClassification.from_pretrained(HF_MODEL_NAME)
    return processor, model

def predict_local(model, image_array):
    # Preprocess expects a batch, image_array should be (224, 224, 3)
    # We need to ensure it's resized and preprocessed correctly
    # The utils.preprocess_image takes a PATH. We need a version for ARRAY.
    # Let's adapt here or import if utils supports array.
    # utils.preprocess_image uses load_img.
    
    # Manual preprocessing for array:
    img = cv2.resize(image_array, (224, 224))
    img = img.astype("float32") # / 255.0 ? EfficientNet usually expects 0-255 or specific. 
    # checking notebook: tf.keras.utils.image_dataset_from_directory does NOT rescale by default unless usually specified. 
    # EfficientNet included `preprocess_input` internally or we need to check training code. 
    # Notebook: `x = data_augmentation(inputs)`, `x = base_model(x, training=False)`. 
    # EfficientNetB0 from tf.keras.applications expects 0-255 inputs if include_top=False? 
    # Actually, `tf.keras.applications.EfficientNetB0` expects input to be float32 in [0, 255].
    
    img_expanded = np.expand_dims(img, axis=0)
    
    preds = model.predict(img_expanded, verbose=0)
    # Output structure: [[prob_indoor, prob_outdoor]]
    pred_idx = np.argmax(preds, axis=1)[0]
    pred_prob = preds[0][pred_idx]
    
    label = CLASSES[pred_idx]
    confidence = pred_prob
    return label, confidence

def predict_hf(processor, model, image_array):
    # processor handles resizing and normalization
    inputs = preprocess_image_for_hf(image_array, processor)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
        pred_idx = np.argmax(probs)
        return CLASSES[pred_idx], probs[pred_idx]

# Sidebar
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Go to", ["Image Inference", "Real-time Video", "Video File Inference", "Live Test Mode"])

st.title("Indoor/Outdoor Scene Classification")

# Load Models
with st.spinner("Loading Models..."):
    local_model = load_local_model()
    hf_processor, hf_model = load_hf_model()

if mode == "Image Inference":
    st.header("Image Inference")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        # Convert to numpy for existing functions
        image_np = np.array(image)
        
        col1, col2 = st.columns([0.6, 0.4])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
        with col2:
            st.subheader("Local Model")
            label_local, conf_local = predict_local(local_model, image_np)
            st.metric(label="EfficientNet Prediction", value=label_local, delta=f"{conf_local:.2%}")
            
            st.divider()
            
            st.subheader("HF Model")
            label_hf, conf_hf = predict_hf(hf_processor, hf_model, image_np)
            st.metric(label="SigLip Prediction", value=label_hf, delta=f"{conf_hf:.2%}")

elif mode == "Real-time Video":
    st.header("Real-time Video Inference")
    run = st.checkbox('Run Webcam')
    FRAME_WINDOW = st.image([])
    
    camera = cv2.VideoCapture(0)
    
    while run:
        ret, frame = camera.read()
        if not ret:
            st.write("Failed to capture video")
            break
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference (Using Local Model for speed)
        label, conf = predict_local(local_model, frame)
        
        # Overlay
        color = (0, 255, 0) if label == 'Outdoor' else (0, 0, 255)
        text = f"{label} ({conf:.2%})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        FRAME_WINDOW.image(frame)
        
    camera.release()

elif mode == "Live Test Mode":
    st.header("Live Test Mode")
    st.write("Select the current ground truth and start testing to measure accuracy.")
    
    col1, col2 = st.columns(2)
    with col1:
        ground_truth = st.radio("Current Environment (Ground Truth)", CLASSES)
    with col2:
        start_test = st.button("Start Test")
        stop_test = st.button("Stop Test")
        
    if 'testing' not in st.session_state:
        st.session_state.testing = False
        st.session_state.correct_count = 0
        st.session_state.total_count = 0
        
    if start_test:
        st.session_state.testing = True
        st.session_state.correct_count = 0
        st.session_state.total_count = 0
        
    if stop_test:
        st.session_state.testing = False
        
    FRAME_WINDOW_TEST = st.image([])
    METRIC_PLACEHOLDER = st.empty()
    
    camera = cv2.VideoCapture(0)
    
    while st.session_state.testing:
        ret, frame = camera.read()
        if not ret:
            st.write("Failed to capture video")
            break
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference
        label, conf = predict_local(local_model, frame)
        
        # Update Stats
        st.session_state.total_count += 1
        if label == ground_truth:
            st.session_state.correct_count += 1
            
        accuracy = st.session_state.correct_count / st.session_state.total_count
        
        # Overlay
        color = (0, 255, 0) if label == ground_truth else (255, 0, 0)
        text = f"Pred: {label} | GT: {ground_truth}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        FRAME_WINDOW_TEST.image(frame)
        
        METRIC_PLACEHOLDER.metric("Live Accuracy", f"{accuracy:.2%}", f"{st.session_state.total_count} frames")
        
    camera.release()
    
elif mode == "Video File Inference":
    st.header("Video File Inference")
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    
    if uploaded_video is not None:
        import tempfile
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        st_frame = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Predict
            label, conf = predict_local(local_model, frame)
            
            # Overlay
            color = (0, 255, 0) if label == 'Outdoor' else (0, 0, 255)
            text = f"{label} ({conf:.2%})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            st_frame.image(frame)
            
        cap.release()
