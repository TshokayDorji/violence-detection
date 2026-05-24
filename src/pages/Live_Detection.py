import streamlit as st
import torch
import cv2
import numpy as np
import os
import sys
import time
import threading
from collections import deque
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ViolenceDetector

st.set_page_config(
    page_title="Live Detection",
    layout="centered"
)

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
DEVICE        = torch.device('cpu')
NUM_FRAMES    = 16
IMG_SIZE      = 224
INFER_EVERY   = 6
SMOOTH_WINDOW = 4
CONFIRM_COUNT = 2
MIN_MOTION    = 8.0
MAX_MOTION    = 95.0

# ─────────────────────────────────────────────────────────
# RTC CONFIGURATION (STUN + TURN for cloud)
# ─────────────────────────────────────────────────────────
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": ["turn:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]
})

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
def compute_motion(gray_prev, gray_curr):
    flow   = cv2.calcOpticalFlowFarneback(
        gray_prev, gray_curr, None,
        pyr_scale=0.5, levels=3, winsize=13,
        iterations=3, poly_n=5, poly_sigma=1.1, flags=0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag) * 100)

def to_tensor(frame_rgb):
    f = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
    f = f.astype(np.float32) / 255.0
    return np.transpose(f, (2, 0, 1))

def infer(model, buffer):
    arr    = np.stack(list(buffer), axis=0)
    tensor = torch.from_numpy(arr).unsqueeze(0)
    with torch.no_grad():
        return float(model(tensor).item())

def draw_overlay(frame_rgb, label, prob, motion):
    out     = frame_rgb.copy()
    h, w    = out.shape[:2]
    is_viol = label == "VIOLENT"

    color = (210, 55, 55) if is_viol else (55, 185, 85)
    cv2.rectangle(out, (0, 0), (w, 6), color, -1)

    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - 36), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

    conf = (1 - prob) * 100 if not is_viol else prob * 100
    text = f"{label}  {conf:.0f}%   motion {motion:.0f}"
    cv2.putText(
        out, text,
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52, (255, 255, 255), 1, cv2.LINE_AA
    )
    return out

# ─────────────────────────────────────────────────────────
# VIDEO PROCESSOR
# ─────────────────────────────────────────────────────────
class ViolenceProcessor(VideoProcessorBase):
    def __init__(self):
        self.model     = load_model()
        self.buf       = deque(maxlen=NUM_FRAMES)
        self.prob_hist = deque(maxlen=SMOOTH_WINDOW)
        self.count     = 0
        self.prev_gray = None
        self.motion    = 0.0
        self.prob      = 0.0
        self.label     = "NON-VIOLENT"
        self.consec    = 0
        self.threshold = 0.75
        # shared result for the sidebar stats
        self._lock     = threading.Lock()
        self.total     = 0
        self.violent   = 0
        self.log       = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")  # H×W×3 RGB

        self.count += 1

        # motion
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        if self.prev_gray is not None:
            self.motion = compute_motion(self.prev_gray, gray)
        self.prev_gray = gray

        self.buf.append(to_tensor(img))

        # inference
        if len(self.buf) == NUM_FRAMES and self.count % INFER_EVERY == 0:
            if MIN_MOTION <= self.motion <= MAX_MOTION:
                raw = infer(self.model, self.buf)
                self.prob_hist.append(raw)
                self.prob = float(np.mean(self.prob_hist))

                if self.prob >= self.threshold:
                    self.consec += 1
                else:
                    self.consec = 0
                    self.label  = "NON-VIOLENT"

                if self.consec >= CONFIRM_COUNT:
                    self.label = "VIOLENT"

                conf = (1 - self.prob) * 100 if self.label == "NON-VIOLENT" \
                       else self.prob * 100

                with self._lock:
                    self.total  += 1
                    self.violent += (1 if self.label == "VIOLENT" else 0)
                    self.log = (self.log + [{
                        "Time":       time.strftime('%H:%M:%S'),
                        "Result":     self.label,
                        "Confidence": f"{conf:.1f}%",
                        "Motion":     f"{self.motion:.1f}",
                    }])[-15:]

            elif self.motion < MIN_MOTION:
                self.consec = 0
                self.label  = "NON-VIOLENT"

        out = draw_overlay(img, self.label, self.prob, self.motion)
        return av.VideoFrame.from_ndarray(out, format="rgb24")

# ─────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────
if "threshold_pct" not in st.session_state:
    st.session_state.threshold_pct = 75

model = load_model()

# ─────────────────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────────────────
st.title("Live Detection")
st.divider()

threshold_pct = st.slider(
    "Detection threshold",
    min_value=0, max_value=100,
    value=st.session_state.threshold_pct, step=1,
    help="Higher = less sensitive to violence.",
    format="%d%%"
)
st.session_state.threshold_pct = threshold_pct
threshold = threshold_pct / 100.0

st.divider()

ctx = webrtc_streamer(
    key="violence-detection",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=ViolenceProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# push current threshold into the processor
if ctx.video_processor:
    ctx.video_processor.threshold = threshold

st.divider()

# ─────────────────────────────────────────────────────────
# LIVE STATS (shown while stream is active)
# ─────────────────────────────────────────────────────────
if ctx.state.playing and ctx.video_processor:
    status_slot = st.empty()
    bar_slot    = st.empty()
    meta_slot   = st.empty()
    st.divider()
    st.caption("Detection log")
    log_slot = st.empty()

    while ctx.state.playing:
        vp   = ctx.video_processor
        prob = vp.prob
        pct  = prob * 100
        m    = vp.motion
        conf = (1 - prob) * 100 if vp.label == "NON-VIOLENT" else pct

        if m < MIN_MOTION:
            status_slot.info("Scene is static — inference paused")
        elif m > MAX_MOTION:
            status_slot.warning("Too much movement — skipping inference")
        elif vp.label == "VIOLENT":
            status_slot.error(f"VIOLENT — {conf:.1f}% confidence")
        else:
            status_slot.success(f"NON-VIOLENT — {conf:.1f}% confidence")

        bar_slot.progress(prob, text=f"Violence probability: {pct:.1f}%")

        with vp._lock:
            total   = vp.total
            violent = vp.violent
            log     = list(vp.log)

        meta_slot.caption(
            f"Total: {total}   "
            f"Violent: {violent}   "
            f"Safe: {total - violent}   "
            f"Motion: {m:.1f}   "
            f"Threshold: {threshold}   "
            f"Confirmed: {vp.consec}/{CONFIRM_COUNT}"
        )

        if log:
            log_slot.dataframe(log, use_container_width=True, hide_index=True)

        time.sleep(0.3)

elif not ctx.state.playing and ctx.video_processor:
    # stream just stopped — show final summary
    vp = ctx.video_processor
    with vp._lock:
        total   = vp.total
        violent = vp.violent
        log     = list(vp.log)

    if total > 0:
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total predictions", total)
        c2.metric("Violent",           violent)
        c3.metric("Safe",              total - violent)
    if log:
        st.caption("Detection log")
        st.dataframe(log, use_container_width=True, hide_index=True)

else:
    st.info("Click **START** above to begin. Your browser will ask for camera permission.")