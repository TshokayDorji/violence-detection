import streamlit as st

st.set_page_config(
    page_title="Violence Detection",
    page_icon="🛡️",
    layout="centered"
)

st.title("Violence Detection System")
st.caption("ResNet50 + LSTM — Real Life Violence Situations dataset")
st.divider()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy",  "97.33%")
c2.metric("Precision", "97.78%")
c3.metric("Recall",    "96.35%")
c4.metric("F1-Score",  "97.06%")

st.divider()

st.markdown("""
**How it works**

Each video is broken into 16 frames sampled uniformly. Every frame passes through
ResNet50 to extract a 2048-dimensional spatial feature vector. The 16 vectors are
fed into a 2-layer LSTM which models how the scene changes over time. A final dense
layer outputs a violence probability between 0 and 1. Above 0.5 the clip is violent.
""")

st.divider()

col1, col2 = st.columns(2)

col1.markdown("""
**Model**

| | |
|---|---|
| CNN encoder | ResNet50 pretrained |
| LSTM layers | 2 × 256 hidden units |
| Input | 16 × 224 × 224 |
| Total parameters | ~26.4M |
| Loss | Binary Cross-Entropy |
| Optimizer | Adam lr=0.0001 |
""")

col2.markdown("""
**Training**

| | |
|---|---|
| Dataset | RLVS 2000 videos |
| Train / Val / Test | 70 / 15 / 15% |
| Train videos | 1400 |
| Val videos | 300 |
| Test videos | 300 |
| Best epoch | 37 of 50 |
| Early stopping | patience 10 |
""")

st.divider()

st.markdown("""
**Pipeline**
Video
└─ Extract 16 frames (uniform sampling)
└─ Resize 224×224, normalize [0,1]
└─ ResNet50 → 2048-dim vector per frame
└─ LSTM (2 layers) → temporal modeling
└─ Dense → Sigmoid → probability
├─ ≥ 0.5 → VIOLENT
└─ < 0.5 → NON-VIOLENT
""")

st.divider()

st.markdown("""
**Confusion Matrix — Test Set (300 videos)**

|  | Predicted Non-Violence | Predicted Violence |
|---|---|---|
| Actual Non-Violence | 160 | 3 |
| Actual Violence | 5 | 132 |
""")

st.divider()
st.info("Use the sidebar to navigate to Video Upload or Live Detection.")