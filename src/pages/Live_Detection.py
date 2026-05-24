import streamlit as st
import torch
import cv2
import numpy as np
import os
import sys
import time
from collections import deque

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

st.set_page_config(page_title="Live Detection", page_icon="📷", layout="centered")
st.title("📷 Live Webcam Detection")
st.caption("Real-time violence detection using your webcam")
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

def preprocess_frame(frame):
    f = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    f = f / 255.0
    f = np.transpose(f, (2, 0, 1))
    return f.astype(np.float32)

def predict_sequence(model, frame_buffer):
    frames = np.array(list(frame_buffer))          # (16, C, H, W)
    tensor = torch.tensor(frames).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = model(tensor).item()
    return "VIOLENT" if prob >= 0.5 else "NON-VIOLENT", prob

model = load_model()

# ── Controls ─────────────────────────────────────────────
col1, col2 = st.columns(2)
start_btn  = col1.button("▶️  Start Webcam", use_container_width=True, type="primary")
stop_btn   = col2.button("⏹️  Stop",         use_container_width=True)

st.divider()

# ── Live feed placeholders ────────────────────────────────
feed_placeholder   = st.empty()
result_placeholder = st.empty()
prob_placeholder   = st.empty()

st.divider()
st.markdown("**📊 Detection History (last 10)**")
history_placeholder = st.empty()

# ── Session state ─────────────────────────────────────────
if 'running' not in st.session_state:
    st.session_state.running = False
if 'history' not in st.session_state:
    st.session_state.history = []

if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

# ── Webcam Loop ───────────────────────────────────────────
if st.session_state.running:
    cap          = cv2.VideoCapture(0)
    frame_buffer = deque(maxlen=NUM_FRAMES)
    frame_count  = 0

    if not cap.isOpened():
        st.error("❌ Cannot access webcam. Make sure it is connected and not in use.")
        st.session_state.running = False
    else:
        result_placeholder.info("⏳ Collecting frames... (need 16 frames before first prediction)")

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read from webcam.")
                break

            frame_count += 1

            # Add to buffer
            frame_buffer.append(preprocess_frame(frame))

            # Display live frame
            display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run prediction every 8 frames once buffer is full
            if len(frame_buffer) == NUM_FRAMES and frame_count % 8 == 0:
                label, prob = predict_sequence(model, frame_buffer)
                pct         = prob * 100

                # Overlay on frame
                color = (243, 139, 168) if label == "VIOLENT" else (166, 227, 161)
                cv2.rectangle(display, (0, 0), (display.shape[1], 50), 
                              tuple(int(c) for c in color), -1)
                cv2.putText(display,
                            f"{label}  —  {pct:.1f}%",
                            (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (30, 30, 46), 2)

                # Update result
                if label == "VIOLENT":
                    result_placeholder.error(f"## 🚨 VIOLENT — {pct:.1f}%")
                else:
                    result_placeholder.success(f"## ✅ NON-VIOLENT — {pct:.1f}%")

                prob_placeholder.progress(prob, text=f"Violence probability: {pct:.1f}%")

                # History
                st.session_state.history.append({
                    'time':  time.strftime('%H:%M:%S'),
                    'label': label,
                    'prob':  f"{pct:.1f}%"
                })
                st.session_state.history = st.session_state.history[-10:]

                history_placeholder.table(st.session_state.history)

            feed_placeholder.image(display, channels="RGB", use_container_width=True)

            time.sleep(0.03)  # ~30fps display

        cap.release()
        result_placeholder.info("⏹️ Detection stopped.")

else:
    feed_placeholder.info("👆 Click **Start Webcam** to begin live detection")