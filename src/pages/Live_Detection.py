import streamlit as st
import torch
import cv2
import numpy as np
import os
import sys
import time
import threading
from collections import deque

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
def open_camera():
    for idx in range(3):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,   1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
            cap.set(cv2.CAP_PROP_FPS,           30)
            cap.set(cv2.CAP_PROP_AUTOFOCUS,     1)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            for _ in range(25):
                cap.read()
            return cap
        cap.release()
    return None

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
# CAMERA THREAD
# ─────────────────────────────────────────────────────────
def camera_worker(model, threshold, det: Detection):
    det.write(phase="warming")

    cap = open_camera()
    if cap is None:
        det.write(phase="error", error_msg="No webcam found.")
        return

    det.write(phase="running")

    buf       = deque(maxlen=NUM_FRAMES)
    prob_hist = deque(maxlen=SMOOTH_WINDOW)
    count     = 0
    prev_gray = None
    motion    = 0.0
    prob      = 0.0
    label     = "NON-VIOLENT"
    consec    = 0

    try:
        while not det.stop.is_set():
            ret, frame = cap.read()
            if not ret:
                det.write(phase="error", error_msg="Camera read failed.")
                break

            count += 1
            frame  = cv2.flip(frame, 1)

            # motion
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (15, 15), 0)
            if prev_gray is not None:
                motion = compute_motion(prev_gray, gray)
            prev_gray = gray

            buf.append(to_tensor(frame))

            # inference
            if len(buf) == NUM_FRAMES and count % INFER_EVERY == 0:
                if MIN_MOTION <= motion <= MAX_MOTION:
                    raw = infer(model, buf)
                    prob_hist.append(raw)
                    prob = float(np.mean(prob_hist))

                    if prob >= threshold:
                        consec += 1
                    else:
                        consec = 0
                        label  = "NON-VIOLENT"

                    if consec >= CONFIRM_COUNT:
                        label = "VIOLENT"

                    conf = (1 - prob) * 100 if label == "NON-VIOLENT" else prob * 100

                    det.write(
                        prob    = prob,
                        label   = label,
                        motion  = motion,
                        consec  = consec,
                        total   = det.total + 1,
                        violent = det.violent + (1 if label == "VIOLENT" else 0),
                    )
                    det.append_log({
                        "Time":       time.strftime('%H:%M:%S'),
                        "Result":     label,
                        "Confidence": f"{conf:.1f}%",
                        "Motion":     f"{motion:.1f}",
                    })

                elif motion < MIN_MOTION:
                    consec = 0
                    label  = "NON-VIOLENT"
                    det.write(label="NON-VIOLENT", consec=0, motion=motion)

            # build display frame
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display = draw_overlay(rgb, label, prob, motion)
            det.write(frame=display)

            time.sleep(0.001)

    finally:
        cap.release()
        if det.phase == "running":
            det.write(phase="stopped")

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

if start_btn and det.phase not in ("warming", "running"):
    det                  = Detection()
    st.session_state.det = det
    threading.Thread(
        target=camera_worker,
        args=(model, threshold, det),
        daemon=True
    ).start()
    time.sleep(0.3)

if stop_btn and det.phase in ("warming", "running"):
    det.stop.set()

st.divider()

feed_slot   = st.empty()
status_slot = st.empty()
bar_slot    = st.empty()
meta_slot   = st.empty()
st.divider()
st.caption("Detection log")
log_slot = st.empty()

# ─────────────────────────────────────────────────────────
# POLLING LOOP
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