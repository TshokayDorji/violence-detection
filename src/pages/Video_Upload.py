import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
import sys
from huggingface_hub import hf_hub_download

# ─────────────────────────────────────────────────────────
# Add project root to path
# ─────────────────────────────────────────────────────────
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from model import ViolenceDetector

# ─────────────────────────────────────────────────────────
# Streamlit Page Config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Video Upload",
    page_icon="📂",
    layout="centered"
)

st.title("📂 Video Upload Detection")
st.caption("Upload any video file to analyze for violent content")
st.divider()

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_FRAMES = 16
IMG_SIZE = 224

# ─────────────────────────────────────────────────────────
# Download / Locate Model
# ─────────────────────────────────────────────────────────
@st.cache_resource
def get_model_path():

    models_dir = os.path.join(
        os.path.dirname(__file__),
        "../../models"
    )

    os.makedirs(models_dir, exist_ok=True)

    local_model_path = os.path.join(
        models_dir,
        "best_model.pth"
    )

    # Use local file if available
    if os.path.exists(local_model_path):
        return local_model_path

    # Download from Hugging Face
    downloaded_path = hf_hub_download(
        repo_id="TshokayDorji/violence-detection",
        filename="best_model.pth",
        cache_dir=models_dir
    )

    return downloaded_path


MODEL_PATH = get_model_path()

# Optional Debugging
st.sidebar.write("📁 Model Path:")
st.sidebar.code(MODEL_PATH)

st.sidebar.write("✅ Model Exists:")
st.sidebar.write(os.path.exists(MODEL_PATH))

# ─────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():

    model = ViolenceDetector().to(DEVICE)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


model = load_model()

# ─────────────────────────────────────────────────────────
# Extract Frames
# ─────────────────────────────────────────────────────────
def extract_frames(video_path):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    indices = np.linspace(
        0,
        max(total_frames - 1, 0),
        NUM_FRAMES,
        dtype=int
    )

    frames_raw = []

    frames_tensor = []

    for idx in indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ret, frame = cap.read()

        if not ret:

            frame = np.zeros(
                (IMG_SIZE, IMG_SIZE, 3),
                dtype=np.uint8
            )

        # Display frame
        display_frame = cv2.resize(
            frame,
            (320, 180)
        )

        display_frame = cv2.cvtColor(
            display_frame,
            cv2.COLOR_BGR2RGB
        )

        frames_raw.append(display_frame)

        # Model preprocessing
        f = cv2.resize(
            frame,
            (IMG_SIZE, IMG_SIZE)
        )

        f = cv2.cvtColor(
            f,
            cv2.COLOR_BGR2RGB
        )

        f = f / 255.0

        f = np.transpose(f, (2, 0, 1))

        frames_tensor.append(f)

    cap.release()

    return (
        frames_raw,
        np.array(frames_tensor, dtype=np.float32)
    )

# ─────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────
def predict(model, frames_tensor):

    tensor = torch.tensor(
        frames_tensor
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        prob = model(tensor).item()

    label = (
        "VIOLENT"
        if prob >= 0.5
        else "NON-VIOLENT"
    )

    return label, prob

# ─────────────────────────────────────────────────────────
# File Upload
# ─────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov", "mkv"]
)

# ─────────────────────────────────────────────────────────
# Run Detection
# ─────────────────────────────────────────────────────────
if uploaded:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ) as tmp:

        tmp.write(uploaded.read())

        tmp_path = tmp.name

    # Display uploaded video
    st.video(uploaded)

    # Analyze
    with st.spinner("🔄 Analyzing video..."):

        frames_raw, frames_tensor = extract_frames(
            tmp_path
        )

        label, prob = predict(
            model,
            frames_tensor
        )

    # Remove temp file
    os.unlink(tmp_path)

    pct = prob * 100

    st.divider()

    # Result
    if label == "VIOLENT":

        st.error(
            f"## 🚨 VIOLENT — {pct:.1f}% confidence"
        )

    else:

        st.success(
            f"## ✅ NON-VIOLENT — {pct:.1f}% confidence"
        )

    # Progress Bar
    st.markdown("### 📊 Violence Probability")

    st.progress(float(prob))

    st.caption(
        f"{pct:.1f}% probability of violence detected"
    )

    st.divider()

    # Sampled Frames
    st.markdown("### 🎞️ Sampled Frames")

    cols = st.columns(4)

    for i, frame in enumerate(frames_raw):

        cols[i % 4].image(
            frame,
            use_container_width=True
        )