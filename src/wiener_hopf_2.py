# -*- coding: utf-8 -*-
"""wiener_hopf.py — Stima closed-form di W* (Wiener-Hopf)"""

import torch


def estimate_wiener_hopf(z_seq: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Stima la matrice W* risolvendo l'equazione di Wiener-Hopf:
        R_zz(1) = W* R_zz(0)

    Args:
        z_seq: (B, L, d) — sequenze di vettori latenti
        eps: regolarizzazione (frazione della traccia)

    Returns:
        W*: (d, d) — matrice di transizione lineare
    """
    B, L, d = z_seq.shape
    z_t = z_seq[:, :-1, :].reshape(-1, d)  # (B*(L-1), d)
    z_tp1 = z_seq[:, 1:, :].reshape(-1, d)

    N = z_t.shape[0]
    Rzz0 = (z_t.T @ z_t) / N  # (d, d)
    Rzz1 = (z_tp1.T @ z_t) / N  # (d, d)

    reg = eps * torch.trace(Rzz0) / d
    Rzz0_reg = Rzz0 + reg * torch.eye(d, device=z_seq.device)

    # Risolvi: Rzz0_reg^T * W^T = Rzz1^T
    W = torch.linalg.lstsq(Rzz0_reg.T, Rzz1.T).solution.T
    return W