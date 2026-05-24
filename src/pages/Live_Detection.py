import streamlit as st
import torch
import cv2
import numpy as np
import os
import sys
from collections import deque
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

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
# Streamlit Config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Live Violence Detection",
    page_icon="📷",
    layout="centered"
)

st.title("📷 Live Violence Detection")
st.caption("Real-time webcam violence detection")
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
# Download Model
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

    # Use local model if exists
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
# Frame Preprocessing
# ─────────────────────────────────────────────────────────
def preprocess_frame(frame):

    frame = cv2.resize(
        frame,
        (IMG_SIZE, IMG_SIZE)
    )

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    frame = frame / 255.0

    frame = np.transpose(
        frame,
        (2, 0, 1)
    )

    return frame.astype(np.float32)

# ─────────────────────────────────────────────────────────
# Video Processor
# ─────────────────────────────────────────────────────────
class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.frame_buffer = deque(
            maxlen=NUM_FRAMES
        )

        self.frame_count = 0

        self.label = "Detecting..."

        self.prob = 0.0

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        # Flip horizontally
        # Removes mirror/opposite effect
        img = cv2.flip(img, 1)

        self.frame_count += 1

        # Preprocess
        processed = preprocess_frame(img)

        self.frame_buffer.append(processed)

        # Prediction
        if (
            len(self.frame_buffer) == NUM_FRAMES
            and self.frame_count % 8 == 0
        ):

            frames = np.array(
                list(self.frame_buffer)
            )

            tensor = torch.tensor(
                frames
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():

                prob = model(tensor).item()

            self.prob = prob

            self.label = (
                "VIOLENT"
                if prob >= 0.5
                else "NON-VIOLENT"
            )

        pct = self.prob * 100

        # Colors
        if self.label == "VIOLENT":

            color = (0, 0, 255)

        else:

            color = (0, 255, 0)

        # Overlay background
        cv2.rectangle(
            img,
            (0, 0),
            (img.shape[1], 60),
            color,
            -1
        )

        # Overlay text
        cv2.putText(
            img,
            f"{self.label} - {pct:.1f}%",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# ─────────────────────────────────────────────────────────
# Webcam Stream
# ─────────────────────────────────────────────────────────
webrtc_streamer(
    key="violence-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)

# ─────────────────────────────────────────────────────────
# Instructions
# ─────────────────────────────────────────────────────────
st.divider()

st.markdown("""
### ✅ Features

- Real-time webcam detection
- Cloud compatible
- No webcam capture issue
- Correct camera orientation
- Violence probability detection
- Hugging Face model loading
""")