import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class TinyBaseline(nn.Module):

    def __init__(self):
        super().__init__()

        # 1-channel grayscale input
        self.head = nn.Conv2d(
            1, 64, kernel_size=3, padding=1
        )

        # Learnable restoration blocks
        self.body = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
        )

        # Convert features back to grayscale
        self.tail = nn.Conv2d(
            64, 1, kernel_size=3, padding=1
        )

        # Upscale 128x128 -> 256x256
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

    def forward(self, x):

        # First upscale
        x = self.upsample(x)

        # Extract features
        features = self.head(x)

        # Residual restoration
        features = self.body(features)

        # Generate restored image
        residual = self.tail(features)

        # Global residual connection
        # Helps preserve original image details
        return x + residual