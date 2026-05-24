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
POLL_MS       = 80

# ─────────────────────────────────────────────────────────
# RTC CONFIGURATION  ← added for cloud webcam support
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
# SHARED STATE
# ─────────────────────────────────────────────────────────
class Detection:
    def __init__(self):
        self._lock     = threading.Lock()
        self.frame     = None
        self.label     = "NON-VIOLENT"
        self.prob      = 0.0
        self.motion    = 0.0
        self.consec    = 0
        self.log       = []
        self.total     = 0
        self.violent   = 0
        self.phase     = "idle"
        self.error_msg = ""
        self.stop      = threading.Event()

    def snapshot(self):
        with self._lock:
            return dict(
                frame   = self.frame,
                label   = self.label,
                prob    = self.prob,
                motion  = self.motion,
                consec  = self.consec,
                log     = list(self.log),
                total   = self.total,
                violent = self.violent,
                phase   = self.phase,
                error   = self.error_msg,
            )

    def write(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def append_log(self, entry):
        with self._lock:
            self.log = (self.log + [entry])[-15:]

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

def to_tensor(frame_bgr):
    f = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE))
    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
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

    # colored strip at top
    color = (210, 55, 55) if is_viol else (55, 185, 85)
    cv2.rectangle(out, (0, 0), (w, 6), color, -1)

    # semi-transparent bottom bar
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - 36), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

    # label text
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
# WEBRTC VIDEO PROCESSOR  ← replaces open_camera + camera_worker
# ─────────────────────────────────────────────────────────
class ViolenceProcessor(VideoProcessorBase):
    def __init__(self):
        self._model    = load_model()
        self._det      = None          # set from outside after creation
        self._buf      = deque(maxlen=NUM_FRAMES)
        self._prob_hist= deque(maxlen=SMOOTH_WINDOW)
        self._count    = 0
        self._prev_gray= None
        self._motion   = 0.0
        self._prob     = 0.0
        self._label    = "NON-VIOLENT"
        self._consec   = 0
        self.threshold = 0.75

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # convert incoming WebRTC frame to BGR (same as cv2 reads)
        img_rgb = frame.to_ndarray(format="rgb24")
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_bgr = cv2.flip(img_bgr, 1)

        self._count += 1

        # motion — identical logic to old camera_worker
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        if self._prev_gray is not None:
            self._motion = compute_motion(self._prev_gray, gray)
        self._prev_gray = gray

        self._buf.append(to_tensor(img_bgr))

        # inference — identical logic to old camera_worker
        if len(self._buf) == NUM_FRAMES and self._count % INFER_EVERY == 0:
            if MIN_MOTION <= self._motion <= MAX_MOTION:
                raw = infer(self._model, self._buf)
                self._prob_hist.append(raw)
                self._prob = float(np.mean(self._prob_hist))

                if self._prob >= self.threshold:
                    self._consec += 1
                else:
                    self._consec = 0
                    self._label  = "NON-VIOLENT"

                if self._consec >= CONFIRM_COUNT:
                    self._label = "VIOLENT"

                conf = (1 - self._prob) * 100 if self._label == "NON-VIOLENT" \
                       else self._prob * 100

                if self._det is not None:
                    self._det.write(
                        prob    = self._prob,
                        label   = self._label,
                        motion  = self._motion,
                        consec  = self._consec,
                        total   = self._det.total + 1,
                        violent = self._det.violent + (1 if self._label == "VIOLENT" else 0),
                        phase   = "running",
                    )
                    self._det.append_log({
                        "Time":       time.strftime('%H:%M:%S'),
                        "Result":     self._label,
                        "Confidence": f"{conf:.1f}%",
                        "Motion":     f"{self._motion:.1f}",
                    })

            elif self._motion < MIN_MOTION:
                self._consec = 0
                self._label  = "NON-VIOLENT"
                if self._det is not None:
                    self._det.write(label="NON-VIOLENT", consec=0,
                                    motion=self._motion)

        # draw overlay and return frame — identical to old draw path
        rgb_out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        display = draw_overlay(rgb_out, self._label, self._prob, self._motion)
        if self._det is not None:
            self._det.write(frame=display)

        return av.VideoFrame.from_ndarray(display, format="rgb24")

# ─────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────
if "det" not in st.session_state:
    st.session_state.det = Detection()

det   = st.session_state.det
model = load_model()

# ─────────────────────────────────────────────────────────
# CAMERA PERMISSION REQUEST
# ─────────────────────────────────────────────────────────
st.components.v1.html("""
<script>
(function() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;
    if (window._camPermissionRequested) return;
    window._camPermissionRequested = true;
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then(function(stream) {
            stream.getTracks().forEach(function(t) { t.stop(); });
        })
        .catch(function(err) {
            console.warn("Camera permission denied or unavailable:", err);
        });
})();
</script>
""", height=0)

# ─────────────────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────────────────
st.title("Live Detection")
st.divider()

threshold_pct = st.slider(
    "Detection threshold",
    min_value=0, max_value=100,
    value=75, step=1,
    help="Higher = less sensitive to violence.",
    format="%d%%"
)
threshold = threshold_pct / 100.0

b1, b2     = st.columns(2)
start_btn  = b1.button("Start", use_container_width=True, type="primary")
stop_btn   = b2.button("Stop",  use_container_width=True)

# ─────────────────────────────────────────────────────────
# WEBRTC STREAMER  ← replaces threading.Thread(camera_worker)
# ─────────────────────────────────────────────────────────
if start_btn and det.phase not in ("warming", "running"):
    det                  = Detection()
    st.session_state.det = det
    det.write(phase="running")

if stop_btn and det.phase in ("warming", "running"):
    det.stop.set()
    det.write(phase="stopped")

ctx = webrtc_streamer(
    key="violence-detection",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=ViolenceProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    desired_playing_state=det.phase == "running",
)

# wire the shared Detection object into the processor
if ctx.video_processor:
    ctx.video_processor._det       = det
    ctx.video_processor.threshold  = threshold

st.divider()

feed_slot   = st.empty()
status_slot = st.empty()
bar_slot    = st.empty()
meta_slot   = st.empty()
st.divider()
st.caption("Detection log")
log_slot = st.empty()

# ─────────────────────────────────────────────────────────
# POLLING LOOP  ← unchanged
# ─────────────────────────────────────────────────────────
if det.phase in ("warming", "running"):
    while True:
        s   = det.snapshot()
        pct = s["prob"] * 100

        # feed
        if s["phase"] == "warming":
            feed_slot.info("Camera warming up — please wait...")
        elif s["frame"] is not None:
            feed_slot.image(
                s["frame"],
                channels="RGB",
                use_container_width=True
            )

        # status
        if s["phase"] == "running":
            m    = s["motion"]
            conf = (1 - s["prob"]) * 100 if s["label"] == "NON-VIOLENT" \
                   else s["prob"] * 100

            if m < MIN_MOTION:
                status_slot.info(
                    f"Scene is static — inference paused"
                )
            elif m > MAX_MOTION:
                status_slot.warning(
                    f"Too much movement — skipping inference"
                )
            elif s["label"] == "VIOLENT":
                status_slot.error(
                    f"VIOLENT — {conf:.1f}% confidence"
                )
            else:
                status_slot.success(
                    f"NON-VIOLENT — {conf:.1f}% confidence"
                )

            bar_slot.progress(
                s["prob"],
                text=f"Violence probability: {pct:.1f}%"
            )

            safe = s["total"] - s["violent"]
            meta_slot.caption(
                f"Total: {s['total']}   "
                f"Violent: {s['violent']}   "
                f"Safe: {safe}   "
                f"Motion: {m:.1f}   "
                f"Threshold: {threshold}   "
                f"Confirmed: {s['consec']}/{CONFIRM_COUNT}"
            )

        if s["log"]:
            log_slot.dataframe(
                s["log"],
                use_container_width=True,
                hide_index=True
            )

        # exit
        if s["phase"] == "error":
            st.error(s["error"])
            break
        if s["phase"] == "stopped":
            status_slot.info("Detection stopped.")
            break
        if stop_btn:
            break

        time.sleep(POLL_MS / 1000)

elif det.phase == "stopped":
    feed_slot.info("Press Start to begin.")
    s = det.snapshot()
    if s["total"] > 0:
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total predictions", s["total"])
        c2.metric("Violent",           s["violent"])
        c3.metric("Safe",              s["total"] - s["violent"])
    if s["log"]:
        log_slot.dataframe(
            s["log"],
            use_container_width=True,
            hide_index=True
        )

elif det.phase == "error":
    st.error(det.error_msg)
    feed_slot.info("Press Start to try again.")

else:
    feed_slot.info("Press Start to begin.")