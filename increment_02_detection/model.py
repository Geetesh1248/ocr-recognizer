"""Small encoder/decoder with skip connections. No imported network architecture."""
import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                                    nn.GroupNorm(4, outgoing), nn.ReLU(),
                                    nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
                                    nn.GroupNorm(4, outgoing), nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class TextDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = ConvBlock(3, 16)
        self.encoder2 = ConvBlock(16, 32)
        self.encoder3 = ConvBlock(32, 64)
        self.bottleneck = ConvBlock(64, 128)
        self.decoder3 = ConvBlock(128+64, 64)
        self.decoder2 = ConvBlock(64+32, 32)
        self.decoder1 = ConvBlock(32+16, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, image):
        a = self.encoder1(image)
        b = self.encoder2(F.max_pool2d(a, 2))
        c = self.encoder3(F.max_pool2d(b, 2))
        x = self.bottleneck(F.max_pool2d(c, 2))
        x = self.decoder3(torch.cat([F.interpolate(x, size=c.shape[-2:], mode="bilinear", align_corners=False), c], dim=1))
        x = self.decoder2(torch.cat([F.interpolate(x, size=b.shape[-2:], mode="bilinear", align_corners=False), b], dim=1))
        x = self.decoder1(torch.cat([F.interpolate(x, size=a.shape[-2:], mode="bilinear", align_corners=False), a], dim=1))
        return self.head(x)  # logits, shape B x 1 x H x W
