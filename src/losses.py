# -*- coding: utf-8 -*-
"""losses.py — Funzioni loss per WH-RDCC"""

import torch
import torch.nn.functional as F


def contrastive_loss(z_seq: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    NT-Xent loss (SimCLR-style) per channel charting.
    Positivi = campioni temporalmente vicini (t=0 e t=1 nella stessa sequenza).
    Negativi = tutti gli altri campioni nel batch.

    Args:
        z_seq: (B, L, d) — sequenze di vettori latenti (già normalizzati L2)
        temperature: parametro di temperatura

    Returns:
        scalar loss
    """
    B, L, d = z_seq.shape
    # Prendi t=0 e t=1 come coppia positiva
    z_a = z_seq[:, 0, :]  # (B, d)
    z_p = z_seq[:, 1, :]  # (B, d)

    # Concatena: (2B, d)
    z_all = torch.cat([z_a, z_p], dim=0)

    # Similarity matrix (2B, 2B)
    sim = torch.mm(z_all, z_all.T) / temperature

    # Maschera diagonale
    mask = torch.eye(2 * B, device=z_seq.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Label: per z_a[i] il positivo è z_p[i] = indice i+B
    labels = torch.arange(B, device=z_seq.device)
    labels = torch.cat([labels + B, labels])  # (2B,)

    loss = F.cross_entropy(sim, labels)
    return loss


def wiener_hopf_loss(z_seq: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    L_WH = || z_{t+1} - W*z_t ||^2

    Args:
        z_seq: (B, L, d) — sequenze di vettori latenti
        W: (d, d) — matrice di transizione lineare

    Returns:
        scalar loss
    """
    z_t = z_seq[:, :-1, :]  # (B, L-1, d)
    z_tp1 = z_seq[:, 1:, :]  # (B, L-1, d)
    pred = z_t @ W.T  # (B, L-1, d)
    loss = (z_tp1 - pred).pow(2).mean()
    return loss