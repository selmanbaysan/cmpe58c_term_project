# Indoor/Outdoor Scene Classification (CMPE58C Term Project)

**Author:** Selman Baysan  
**Course:** CMPE58C: Sp. Tp. Mobile Location Tracking and Motion Sensing  

This project implements a robust deep learning model to classify images and videos as **Indoor** or **Outdoor** environments. It involves fine-tuning an **EfficientNetB0** model on the [IndoorOutdoorNet-20K](https://huggingface.co/datasets/prithivMLmods/IndoorOutdoorNet-20K) dataset, optimized for performance on **Apple Silicon (M1/M2)** using `tensorflow-metal`.

## Features
-   **Fine-Tuned Model**: 99.8% Accuracy efficientNetB0 based classifier.
-   **Streamlit Application**:
    -   **Image Inference**: Upload images for classification.
    -   **Video File Inference**: Process uploaded videos (mp4/mov) frame-by-frame.
    -   **Real-time Webcam**: Live inference using your computer's camera.
    -   **Live Test Mode**: Validate accuracy in real-time against ground truth.
-   **Benchmarking**: Utility to compare Local TF model vs HF PyTorch model speed.

---

## Installation & Setup

### Prerequisites
-   **Operating System**: macOS (Apple Silicon M1/M2/M3 recommended for GPU acceleration).
-   **Python**: 3.10 or 3.11 (Managed via Poetry).
-   **Poetry**: Dependency manager (`pip install poetry`).

### 1. Clone the Repository
```bash
git clone <repository_url>
cd indoor-outdoor-detection
```

### 2. Install Dependencies
This project uses **Poetry** for package management, but due to platform-specific compatibility issues with TensorFlow on macOS, some packages must be installed manually via `pip` inside the poetry environment.

**Step 2.1: Install Base Dependencies**
```bash
poetry install
```

**Step 2.2: Install TensorFlow (Apple Silicon Optimized)**
*Note: These are explicitly installed to avoid Poetry resolution conflicts.*
```bash
poetry run pip install tensorflow-macos==2.15.0 tensorflow-metal==1.1.0
```

**Step 2.3: Install Torch and Protobuf**
*Downgrading protobuf is required to resolve compatibility issues.*
```bash
poetry run pip install torch torchvision protobuf==3.20.3
```

---

## Running the Application

To launch the Streamlit web interface:

```bash
poetry run streamlit run app.py
```
This will open the app in your default browser (usually `http://localhost:8501`).

---

## Evaluation & Benchmarking

### Compare with Hugging Face Model
To evaluate the original pre-trained model against our fine-tuned model:
```bash
poetry run python evaluate_hf_model.py
```

### Save Misclassified Images
To generate a folder of failure cases (`false_predictions/`) for analysis:
```bash
poetry run python save_misclassified.py
```

### Benchmark Inference Speed
To compare the inference latency (CPU vs GPU) of both models:
```bash
poetry run python benchmark_inference.py
```

---

## Training 

If you wish to retrain the model from scratch:

1.  **Phase 1 (Train Top Layers)**:
    ```bash
    poetry run python training/train_phase1.py
    ```
2.  **Phase 2 (Fine-Tune)**:
    ```bash
    poetry run python training/train_phase2_finetune.py
    ```

*Note: Scripts are located in `training/` folder.*

