import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import time
from tqdm import tqdm

from dataset import ViolenceDataset
from model import ViolenceDetector

# ─── CONFIG ──────────────────────────────────────────────
DATA_DIR    = '../data'
MODEL_DIR   = '../models'
LOG_DIR     = '../logs'
NUM_FRAMES  = 16
IMG_SIZE    = 224
BATCH_SIZE  = 8        # lower if GPU runs out of memory
EPOCHS      = 50
LR          = 1e-4
PATIENCE    = 10       # early stopping
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ─────────────────────────────────────────────────────────

def train():
    print(f"\n{'='*50}")
    print(f"  Training on: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*50}\n")

    # Datasets
    train_ds = ViolenceDataset(DATA_DIR, split='train', num_frames=NUM_FRAMES, img_size=IMG_SIZE)
    val_ds   = ViolenceDataset(DATA_DIR, split='val',   num_frames=NUM_FRAMES, img_size=IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = ViolenceDetector().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = nn.BCELoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        # ── Training ──
        model.train()
        train_loss = 0.0
        for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            frames = frames.to(DEVICE)
            labels = labels.float().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validation ──
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for frames, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]  "):
                frames = frames.to(DEVICE)
                labels = labels.float().to(DEVICE)

                outputs = model(frames)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = (outputs >= 0.5).long()
                correct += (preds == labels.long()).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total * 100

        scheduler.step(val_loss)

        elapsed = time.time() - start
        print(f"\nEpoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"Time: {elapsed:.1f}s")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pth'))
            print(f"   Best model saved! (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print(f"\n Early stopping triggered at epoch {epoch}")
            break

    # Save history
    np.save(os.path.join(LOG_DIR, 'history.npy'), history)
    print("\n Training complete! Best model saved to models/best_model.pth")

if __name__ == '__main__':
    train()