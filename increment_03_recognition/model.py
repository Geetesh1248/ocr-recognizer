import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextRecognizerV1(nn.Module):
    """Legacy recognition network using GroupNorm CNN + 2-layer BiLSTM."""

    def __init__(self, classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(4, 64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(4, 128), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.sequence = nn.LSTM(128, 128, num_layers=2, bidirectional=True, dropout=.1)
        self.characters = nn.Linear(256, classes)

    def forward(self, images, lengths):
        features = self.cnn(images).mean(dim=2).permute(2, 0, 1)  # time, batch, features
        packed = pack_padded_sequence(features, lengths.cpu(), enforce_sorted=False)
        encoded, _ = self.sequence(packed)
        encoded, _ = pad_packed_sequence(encoded)
        return self.characters(encoded)


class ResBlock(nn.Module):
    """Residual convolutional block with Batch Normalization."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class TextRecognizerV2(nn.Module):
    """Upgraded ResNet-BiLSTM text recognizer with BatchNorm, dropout, and CTC loss.
    
    Architecture:
    - Input: (B, 1, H=32, W)
    - Residual CNN with BatchNorm downsamples 4x horizontally (W -> W//4) and 8x vertically (32 -> 4).
    - Vertical reduction to 1 feature map.
    - 2-layer Bidirectional LSTM (hidden_size=256, 512 total) with dropout=0.3.
    - Linear projection to vocabulary classes (blank=0).
    - Total parameters: ~5.57M (under 10M constraint).
    """

    def __init__(self, classes: int):
        super().__init__()
        self.classes = classes
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Stage 1: 64 -> 128, H: 32 -> 16, W: W -> W//2
            ResBlock(64, 128),
            nn.MaxPool2d(2, 2),
            # Stage 2: 128 -> 256, H: 16 -> 8, W: W//2 -> W//4
            ResBlock(128, 128),
            ResBlock(128, 256),
            nn.MaxPool2d(2, 2),
            # Stage 3: 256 -> 256, H: 8 -> 4, W: W//4 (horizontal factor stays 4)
            ResBlock(256, 256),
            nn.MaxPool2d((2, 1)),
            # Stage 4: Height reduction 4 -> 1
            nn.Conv2d(256, 256, (4, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.sequence = nn.LSTM(256, 256, num_layers=2, bidirectional=True, dropout=0.3)
        self.characters = nn.Linear(512, classes)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized TextRecognizerV2 ({classes} classes, {total_params:,} parameters)", flush=True)

    def forward(self, images: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # images: (B, 1, 32, W)
        features = self.cnn(images)  # (B, 256, 1, W//4)
        features = features.squeeze(2).permute(2, 0, 1)  # (W//4, B, 256)
        packed = pack_padded_sequence(features, lengths.cpu(), enforce_sorted=False)
        encoded, _ = self.sequence(packed)
        encoded, _ = pad_packed_sequence(encoded)
        return self.characters(encoded)  # (T, B, classes)


# Default TextRecognizer is TextRecognizerV2 for modern performance
TextRecognizer = TextRecognizerV2


def build_recognizer(classes: int, version: int = 2) -> nn.Module:
    """Build TextRecognizer instance by architecture version."""
    if version == 1:
        return TextRecognizerV1(classes)
    elif version == 2:
        return TextRecognizerV2(classes)
    else:
        raise ValueError(f"Unknown recognizer version: {version}. Expected 1 or 2.")
