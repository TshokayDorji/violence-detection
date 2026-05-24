import torch
import cv2
import numpy as np
import sys
import os
from model import ViolenceDetector

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_video(video_path, num_frames=16, img_size=224):
    model = ViolenceDetector().to(DEVICE)
    model.load_state_dict(torch.load('../models/best_model.pth', map_location=DEVICE))
    model.eval()

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)

    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (img_size, img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        frames.append(frame)
    cap.release()

    frames = torch.tensor(np.array(frames), dtype=torch.float32)
    frames = frames.unsqueeze(0).to(DEVICE)  # (1, T, C, H, W)

    with torch.no_grad():
        prob = model(frames).item()

    label = "🚨 VIOLENT" if prob >= 0.5 else "✅ NON-VIOLENT"
    print(f"\nVideo: {video_path}")
    print(f"Result: {label}")
    print(f"Confidence: {prob*100:.1f}% violence probability")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/video.mp4")
    else:
        predict_video(sys.argv[1])