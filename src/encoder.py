# -*- coding: utf-8 -*-
"""encoder.py — CNN encoder f_θ con L2 normalization"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSIEncoder(nn.Module):
    """
    Encoder per CSI (32 antenne, 1024 subportanti, 2 canali Re/Im).
    Input:  (B, 32, 1024, 2)
    Output: (B, latent_dim) — vettori normalizzati L2.
    """

    def __init__(self, latent_dim: int = 8):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(2, 7), stride=(2, 2), padding=(0, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(2, 7), stride=(2, 4), padding=(0, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 32, 1024, 2) — CSI in forma Re/Im
        Returns:
            (B, latent_dim) — vettori normalizzati L2
        """
        # (B, 32, 1024, 2) → (B, 2, 32, 1024)
        x = x.permute(0, 3, 1, 2)

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            x = layer(x)

        x = self.gap(x)  # (B, 128, 1, 1)
        z = self.proj(x)  # (B, latent_dim)
        return F.normalize(z, dim=1)  # L2 normalization sulla sfera unitaria