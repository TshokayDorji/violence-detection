import numpy as np
import matplotlib.pyplot as plt
import os

# Load history
history = np.load("../logs/history.npy", allow_pickle=True).item()

# Plot curves
plt.figure(figsize=(12,5))

# Loss subplot
plt.subplot(1,2,1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()

# Accuracy subplot (only validation accuracy available)
plt.subplot(1,2,2)
plt.plot(history['val_acc'], label='Validation Accuracy', color='orange')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# Save plots
plt.savefig("../logs/training_curves.png")
print("Saved training curves to ../logs/training_curves.png")
