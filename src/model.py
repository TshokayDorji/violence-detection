import torch
import torch.nn as nn
import torchvision.models as models

class ViolenceDetector(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, dropout=0.3):
        super(ViolenceDetector, self).__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # output: (B, 2048, 1, 1)

        for name, param in self.cnn.named_parameters():
            if 'layer4' not in name and 'layer3' not in name:
                param.requires_grad = False

        # LSTM Temporal Model
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)          
        features = self.cnn(x)                 
        features = features.view(B, T, -1)     

        lstm_out, _ = self.lstm(features)     
        final_hidden = lstm_out[:, -1, :]     
        out = self.classifier(final_hidden)    
        return out.squeeze(1)                  