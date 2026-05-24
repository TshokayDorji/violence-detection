import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ViolenceDetector

import os
from huggingface_hub import hf_hub_download

def get_model_path():
    local_path = os.path.join(os.path.dirname(__file__), '../../models/best_model.pth')
    if os.path.exists(local_path):
        return local_path
    # Download from Hugging Face if not found locally
    os.makedirs(os.path.join(os.path.dirname(__file__), '../../models'), exist_ok=True)
    path = hf_hub_download(
        repo_id="TshokayDorji/violence-detection",
        filename="best_model.pth",
        cache_dir=os.path.join(os.path.dirname(__file__), '../../models')
    )
    return path

MODEL_PATH = get_model_path()

st.set_page_config(page_title="Video Upload", page_icon="📂", layout="centered")
st.title("📂 Video Upload Detection")
st.caption("Upload any video file to analyze for violent content")
st.divider()

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/best_model.pth')
NUM_FRAMES = 16
IMG_SIZE   = 224

@st.cache_resource
def load_model():
    model = ViolenceDetector().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def extract_frames(video_path):
    cap     = cv2.VideoCapture(video_path)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), NUM_FRAMES, dtype=int)
    frames_raw    = []
    frames_tensor = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        frames_raw.append(cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2RGB))
        f = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = f / 255.0
        f = np.transpose(f, (2, 0, 1))
        frames_tensor.append(f)
    cap.release()
    return frames_raw, np.array(frames_tensor, dtype=np.float32)

def predict(model, frames_tensor):
    tensor = torch.tensor(frames_tensor).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = model(tensor).item()
    label = "VIOLENT" if prob >= 0.5 else "NON-VIOLENT"
    return label, prob

model = load_model()

uploaded = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.video(uploaded)

    with st.spinner("🔄 Analyzing video..."):
        frames_raw, frames_tensor = extract_frames(tmp_path)
        label, prob = predict(model, frames_tensor)

    os.unlink(tmp_path)
    pct = prob * 100

    st.divider()

    if label == "VIOLENT":
        st.error(f"## 🚨 VIOLENT  —  {pct:.1f}% confidence")
    else:
        st.success(f"## ✅ NON-VIOLENT  —  {pct:.1f}% confidence")

    st.markdown("**Violence Probability**")
    st.progress(prob)
    st.caption(f"{pct:.1f}% probability of violence detected")

    st.divider()
    st.markdown("**🎞️ Sampled Frames**")
    cols = st.columns(8)
    for i, col in enumerate(cols):
        if i < len(frames_raw):
            col.image(frames_raw[i], use_container_width=True)