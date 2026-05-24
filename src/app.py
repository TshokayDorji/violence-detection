import streamlit as st

st.set_page_config(
    page_title="Violence Detection System",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Violence Detection System")
st.caption("ResNet50 + LSTM Deep Learning Model | Accuracy: 97.33%")
st.divider()

st.markdown("""
## 👋 Welcome

This system uses a **ResNet50 + LSTM** deep learning model to detect violence in videos.

### 📌 Navigation
Use the **sidebar** to switch between pages:

| Page | Description |
|---|---|
| 📂 **Video Upload** | Upload any video and get instant prediction |
| 📷 **Live Detection** | Use your webcam for real-time detection |

---

### 📊 Model Performance
""")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy",  "97.33%")
c2.metric("Precision", "97.78%")
c3.metric("Recall",    "96.35%")
c4.metric("F1-Score",  "97.06%")

st.divider()
st.markdown("""
### 🧠 Architecture
- **CNN Encoder:** ResNet50 (pretrained on ImageNet)
- **Temporal Model:** 2-layer LSTM (256 hidden units)
- **Dataset:** Real Life Violence Situations (2000 videos)
- **Training:** 37 epochs with early stopping
""")

st.info("👈 Select a page from the sidebar to get started.")