import torch
import torch.nn as nn
import torchvision.models as models


class ViolenceDetector(nn.Module):
    """
    ResNet50 + LSTM hybrid model for violence detection in video.
    - ResNet50 extracts spatial features per frame (2048-dim vector)
    - 2-layer LSTM captures temporal relationships across frames
    - Dense classifier outputs violence probability
    """

    def __init__(self, hidden_size=256, num_layers=2, dropout=0.3):
        super(ViolenceDetector, self).__init__()

        # ── CNN Spatial Encoder: ResNet50 ──────────────────────
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # (B, 2048, 1, 1)

        # Freeze early layers — only fine-tune layer3 and layer4
        for name, param in self.cnn.named_parameters():
            if 'layer3' not in name and 'layer4' not in name:
                param.requires_grad = False

        # ── LSTM Temporal Encoder ──────────────────────────────
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # ── Classifier Head ────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)           # (B*T, C, H, W)
        features = self.cnn(x)                 # (B*T, 2048, 1, 1)
        features = features.view(B, T, -1)     # (B, T, 2048)
        lstm_out, _ = self.lstm(features)      # (B, T, 256)
        final = lstm_out[:, -1, :]             # (B, 256)
        out = self.classifier(final)           # (B, 1)
        return out.squeeze(1)                  # (B,)