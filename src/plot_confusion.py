import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
from dataset import ViolenceDataset
from model import ViolenceDetector

DATA_DIR  = '../data'
MODEL_DIR = '../models'
LOG_DIR   = '../logs'
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_confusion_matrix():
    test_ds     = ViolenceDataset(DATA_DIR, split='test', num_frames=16)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2)

    model = ViolenceDetector(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pth'), map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for frames, labels in test_loader:
            frames  = frames.to(DEVICE)
            outputs = model(frames)
            preds   = (outputs >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm     = confusion_matrix(all_labels, all_preds)
    labels = ['NonViolence', 'Violence']

    # ── values to display (count + percentage) ──
    total      = cm.sum()
    cm_percent = cm / total * 100

    annot = np.array([
        [f"{cm[i][j]}\n({cm_percent[i][j]:.1f}%)" for j in range(2)]
        for i in range(2)
    ])

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=annot,
        fmt='',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        linewidths=1,
        linecolor='white',
        cbar_kws={'label': 'Count'},
        ax=ax
    )

    ax.set_title('Confusion Matrix — Violence Detection\n(ResNet50 + LSTM)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
    ax.set_ylabel('True Label',      fontsize=12, labelpad=10)
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # ── metrics annotation below plot ──
    TP = cm[1][1]; TN = cm[0][0]
    FP = cm[0][1]; FN = cm[1][0]
    acc  = (TP + TN) / total * 100
    prec = TP / (TP + FP) * 100
    rec  = TP / (TP + FN) * 100
    f1   = 2 * prec * rec / (prec + rec)

    stats = f"Accuracy: {acc:.2f}%   Precision: {prec:.2f}%   Recall: {rec:.2f}%   F1-Score: {f1:.2f}%"
    fig.text(0.5, 0.01, stats, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    save_path = os.path.join(LOG_DIR, 'confusion_matrix.png')
    os.makedirs(LOG_DIR, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    plot_confusion_matrix()
