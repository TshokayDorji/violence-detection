import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ViolenceDetector

st.set_page_config(
    page_title="Video Upload",
    layout="centered"
)

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
DEVICE     = torch.device('cpu')
NUM_FRAMES = 16
IMG_SIZE   = 224

# ─────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    from huggingface_hub import hf_hub_download
    model_dir  = os.path.join(os.path.dirname(__file__), '../../models')
    local_path = os.path.join(model_dir, 'best_model.pth')
    if not os.path.exists(local_path):
        os.makedirs(model_dir, exist_ok=True)
        local_path = hf_hub_download(
            repo_id=st.secrets.get(
                "violence-detection", "TshokayDorji/violence-detection"
            ),
            filename="best_model.pth",
            local_dir=model_dir,
            token=st.secrets.get("HF_TOKEN", None)
        )
    m = ViolenceDetector().to(DEVICE)
    m.load_state_dict(torch.load(local_path, map_location=DEVICE))
    m.eval()
    return m

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def extract_frames(video_path):
    cap     = cv2.VideoCapture(video_path)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    indices = np.linspace(0, max(total - 1, 0), NUM_FRAMES, dtype=int)

    frames_display = []
    frames_tensor  = []

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        frames_display.append(
            cv2.cvtColor(
                cv2.resize(frame, (160, 90)),
                cv2.COLOR_BGR2RGB
            )
        )

        f = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = f.astype(np.float32) / 255.0
        f = np.transpose(f, (2, 0, 1))
        frames_tensor.append(f)

    cap.release()
    duration = round(total / fps, 1) if fps > 0 else 0
    return frames_display, np.array(frames_tensor), duration, total, int(fps)


def predict(model, frames_tensor):
    tensor = torch.from_numpy(frames_tensor).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = float(model(tensor).item())
    label = "VIOLENT" if prob >= 0.5 else "NON-VIOLENT"
    return label, prob


# ─────────────────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────────────────
st.title("Video Upload")
st.divider()

model    = load_model()
uploaded = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded is None:
    st.info("Upload a video file above to begin analysis.")
    st.stop()

# save to temp file
with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

st.video(tmp_path)
st.divider()

with st.spinner("Analyzing video..."):
    frames_display, frames_tensor, duration, total_frames, fps = extract_frames(tmp_path)
    label, prob = predict(model, frames_tensor)

os.unlink(tmp_path)

pct  = prob * 100
conf = pct if label == "VIOLENT" else (1 - prob) * 100

# ─────────────────────────────────────────────────────────
# RESULT
# ─────────────────────────────────────────────────────────
if label == "VIOLENT":
    st.error(f"VIOLENT — {conf:.1f}% confidence")
else:
    st.success(f"NON-VIOLENT — {conf:.1f}% confidence")

st.progress(prob, text=f"Violence probability: {pct:.1f}%")

st.divider()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Result",     label)
c2.metric("Confidence", f"{conf:.1f}%")
c3.metric("Duration",   f"{duration}s")
c4.metric("FPS",        fps)

st.divider()

# ─────────────────────────────────────────────────────────
# FRAMES
# ─────────────────────────────────────────────────────────
st.caption("16 frames sampled uniformly from the video")

row1 = st.columns(8)
row2 = st.columns(8)

for i, col in enumerate(row1):
    col.image(frames_display[i],     use_container_width=True)
for i, col in enumerate(row2):
    col.image(frames_display[i + 8], use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────
# INTERPRETATION
# ─────────────────────────────────────────────────────────
st.subheader("Interpretation")

if label == "VIOLENT":
    level = "High" if pct > 80 else "Moderate" if pct > 65 else "Low"
    st.error(f"""
**Violence detected**
- Violence probability: {pct:.2f}%
- Confidence level: {level}
- The model identified motion patterns and visual cues consistent with violent behavior across the sampled frames.
""")
else:
    safe_pct = (1 - prob) * 100
    level    = "High" if safe_pct > 80 else "Moderate" if safe_pct > 65 else "Low"
    st.success(f"""
**No violence detected**
- Non-violence probability: {safe_pct:.2f}%
- Confidence level: {level}
- The model found no significant patterns consistent with violent behavior in the sampled frames.
""")