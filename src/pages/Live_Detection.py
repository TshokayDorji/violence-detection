import streamlit as st
import torch
import cv2
import numpy as np
import os
import sys
import time
from collections import deque
from huggingface_hub import hf_hub_download

# ─────────────────────────────────────────────────────────
# Add project root to path
# ─────────────────────────────────────────────────────────
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from model import ViolenceDetector

# ─────────────────────────────────────────────────────────
# Streamlit Page Config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Live Detection",
    page_icon="📷",
    layout="centered"
)

st.title("📷 Live Webcam Detection")
st.caption("Real-time violence detection using your webcam")
st.divider()

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_FRAMES = 16
IMG_SIZE = 224

# ─────────────────────────────────────────────────────────
# Download / Load Model
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

    # Use local file if already exists
    if os.path.exists(local_model_path):
        return local_model_path

    # Otherwise download from Hugging Face
    downloaded_path = hf_hub_download(
        repo_id="TshokayDorji/violence-detection",
        filename="best_model.pth",
        cache_dir=models_dir
    )

    return downloaded_path


MODEL_PATH = get_model_path()

# Debugging info (optional)
st.sidebar.write("📁 Model Path:")
st.sidebar.code(MODEL_PATH)

st.sidebar.write("✅ File Exists:")
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
# Frame Preprocessing
# ─────────────────────────────────────────────────────────
def preprocess_frame(frame):

    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = frame / 255.0

    frame = np.transpose(frame, (2, 0, 1))

    return frame.astype(np.float32)


# ─────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────
def predict_sequence(model, frame_buffer):

    frames = np.array(list(frame_buffer))

    tensor = torch.tensor(frames).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = model(tensor).item()

    label = "VIOLENT" if prob >= 0.5 else "NON-VIOLENT"

    return label, prob


# ─────────────────────────────────────────────────────────
# Buttons
# ─────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

start_btn = col1.button(
    "▶️ Start Webcam",
    use_container_width=True,
    type="primary"
)

stop_btn = col2.button(
    "⏹️ Stop",
    use_container_width=True
)

st.divider()

# ─────────────────────────────────────────────────────────
# UI Placeholders
# ─────────────────────────────────────────────────────────
feed_placeholder = st.empty()

result_placeholder = st.empty()

prob_placeholder = st.empty()

st.divider()

st.markdown("### 📊 Detection History")

history_placeholder = st.empty()

# ─────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────
if "running" not in st.session_state:
    st.session_state.running = False

if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────────────────────
# Button Logic
# ─────────────────────────────────────────────────────────
if start_btn:
    st.session_state.running = True

if stop_btn:
    st.session_state.running = False

# ─────────────────────────────────────────────────────────
# Webcam Loop
# ─────────────────────────────────────────────────────────
if st.session_state.running:

    cap = cv2.VideoCapture(0)

    frame_buffer = deque(maxlen=NUM_FRAMES)

    frame_count = 0

    if not cap.isOpened():

        st.error(
            "❌ Cannot access webcam. "
            "Make sure webcam permissions are enabled."
        )

        st.session_state.running = False

    else:

        result_placeholder.info(
            "⏳ Collecting frames... "
            "(Need 16 frames before prediction)"
        )

        while st.session_state.running:

            ret, frame = cap.read()

            if not ret:
                st.error("❌ Failed to read webcam.")
                break

            frame_count += 1

            # Store frame
            frame_buffer.append(
                preprocess_frame(frame)
            )

            # Display image
            display = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Predict every 8 frames
            if (
                len(frame_buffer) == NUM_FRAMES
                and frame_count % 8 == 0
            ):

                label, prob = predict_sequence(
                    model,
                    frame_buffer
                )

                pct = prob * 100

                # Overlay result
                color = (
                    (255, 0, 0)
                    if label == "VIOLENT"
                    else (0, 255, 0)
                )

                cv2.rectangle(
                    display,
                    (0, 0),
                    (display.shape[1], 50),
                    color,
                    -1
                )

                cv2.putText(
                    display,
                    f"{label} - {pct:.1f}%",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                # Update UI
                if label == "VIOLENT":

                    result_placeholder.error(
                        f"## 🚨 VIOLENT - {pct:.1f}%"
                    )

                else:

                    result_placeholder.success(
                        f"## ✅ NON-VIOLENT - {pct:.1f}%"
                    )

                prob_placeholder.progress(
                    float(prob),
                    text=f"Violence Probability: {pct:.1f}%"
                )

                # Save history
                st.session_state.history.append({
                    "Time": time.strftime("%H:%M:%S"),
                    "Result": label,
                    "Probability": f"{pct:.1f}%"
                })

                st.session_state.history = (
                    st.session_state.history[-10:]
                )

                history_placeholder.table(
                    st.session_state.history
                )

            # Show webcam feed
            feed_placeholder.image(
                display,
                channels="RGB",
                use_container_width=True
            )

            time.sleep(0.03)

        cap.release()

        result_placeholder.info(
            "⏹️ Detection stopped."
        )

else:

    feed_placeholder.info(
        "👆 Click 'Start Webcam' to begin detection"
    )