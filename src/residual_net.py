# -*- coding: utf-8 -*-
"""residual_net.py — Residual net g_φ (MLP) per la componente non lineare"""

import torch
import torch.nn as nn


class ResidualNet(nn.Module):
    """
    Predice il residuo non lineare: Δz = z_{t+1} - W*z_t.
    Input/Output: (B, latent_dim)
    """

    def __init__(self, latent_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → Δz: (B, latent_dim)"""
        return self.net(z)