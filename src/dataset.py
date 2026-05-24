import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class ViolenceDataset(Dataset):
    def __init__(self, data_dir, split='train', num_frames=16, img_size=224,
                 train_ratio=0.70, val_ratio=0.15):
        self.num_frames = num_frames
        self.img_size = img_size
        self.samples = []  # (video_path, label)

        classes = {'Violence': 1, 'NonViolence': 0}

        all_samples = []
        for class_name, label in classes.items():
            class_dir = os.path.join(data_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.endswith(('.mp4', '.avi', '.mov')):
                    all_samples.append((os.path.join(class_dir, fname), label))

        # Shuffle with fixed seed for reproducibility
        np.random.seed(42)
        np.random.shuffle(all_samples)

        total = len(all_samples)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))

        if split == 'train':
            self.samples = all_samples[:train_end]
        elif split == 'val':
            self.samples = all_samples[train_end:val_end]
        else:  # test
            self.samples = all_samples[val_end:]

        print(f"[{split.upper()}] {len(self.samples)} videos loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._extract_frames(video_path)
        frames = torch.tensor(frames, dtype=torch.float32)  # (T, C, H, W)
        return frames, label

    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, max(total - 1, 0), self.num_frames, dtype=int)

        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0  # normalize to [0,1]
            frame = np.transpose(frame, (2, 0, 1))  # (C, H, W)
            frames.append(frame)

        cap.release()
        return np.array(frames, dtype=np.float32)  # (T, C, H, W)