import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

from dataset import ViolenceDataset
from model import ViolenceDetector

DATA_DIR  = '../data'
MODEL_DIR = '../models'
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate():
    test_ds = ViolenceDataset(DATA_DIR, split='test', num_frames=16)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2)

    model = ViolenceDetector().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pth'), map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for frames, labels in test_loader:
            frames = frames.to(DEVICE)
            outputs = model(frames)
            preds = (outputs >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds,
                                target_names=['NonViolence', 'Violence']))

    print("=== Confusion Matrix ===")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    TP = cm[1][1]; TN = cm[0][0]
    FP = cm[0][1]; FN = cm[1][0]
    accuracy  = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nAccuracy : {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall   : {recall*100:.2f}%")
    print(f"F1-Score : {f1*100:.2f}%")

if __name__ == '__main__':
    evaluate()