# -*- coding: utf-8 -*-
"""evaluate.py — Metriche di valutazione per WH-RDCC"""

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import lstsq
from scipy.spatial.distance import cdist
from sklearn.manifold import trustworthiness
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_latents(
    encoder: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Estrae tutti i vettori latenti (primo campione di ogni sequenza)."""
    encoder.eval()
    zs, ps = [], []
    for csi_b, pos_b in loader:
        zs.append(encoder(csi_b[:, 0].to(device)).cpu().numpy())
        ps.append(pos_b[:, 0].numpy())
    return np.concatenate(zs), np.concatenate(ps)


def affine_align(z_src: np.ndarray, p_src: np.ndarray, z_tgt: np.ndarray) -> np.ndarray:
    """
    Allineamento affine z → posizione (per MAE).
    Trova A,b tali che z_src @ A.T + b ≈ p_src, applica a z_tgt.
    """
    N = len(z_src)
    Z = np.hstack([z_src, np.ones((N, 1))])  # (N, d+1)
    AB, _, _, _ = lstsq(Z, p_src, rcond=None)  # (d+1, 2)
    A, b = AB[:-1], AB[-1]
    return z_tgt @ A + b


def compute_metrics(
    z_train: np.ndarray,
    p_train: np.ndarray,
    z_test: np.ndarray,
    p_test: np.ndarray,
    k: int = 10,
) -> dict:
    """
    Calcola metriche standard di channel charting.

    Returns:
        dict con: mae_train, mae_test, trustworthiness, continuity, kruskal_stress
    """
    # Allineamento affine
    p_pred_train = affine_align(z_train, p_train, z_train)
    p_pred_test = affine_align(z_train, p_train, z_test)

    mae_train = np.mean(np.linalg.norm(p_pred_train - p_train, axis=1))
    mae_test = np.mean(np.linalg.norm(p_pred_test - p_test, axis=1))

    # Trustworthiness e Continuity
    tw = trustworthiness(p_train, z_train, n_neighbors=k)
    ct = trustworthiness(z_train, p_train, n_neighbors=k)

    # Kruskal Stress
    D_pos = cdist(p_train, p_train)
    D_lat = cdist(z_train, z_train)
    D_pos_n = D_pos / D_pos.max()
    D_lat_n = D_lat / D_lat.max()
    ks = np.sqrt(np.sum((D_pos_n - D_lat_n) ** 2) / np.sum(D_pos_n**2))

    return {
        "mae_train": float(mae_train),
        "mae_test": float(mae_test),
        "trustworthiness": float(tw),
        "continuity": float(ct),
        "kruskal_stress": float(ks),
    }


def compute_wiener_hopf_metrics(
    encoder: nn.Module,
    residual: nn.Module,
    W: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Calcola metriche dinamiche WH:
        - pred_error: ||z_{t+1} - W*z_t||
        - residual_norm: ||g_φ(z)||        - spectral_radius: ρ(W*)
        - effective_rank: numero di autovalori > 0.1
    """
    encoder.eval()
    residual.eval()

    with torch.no_grad():
        all_z_seq = []
        for csi_b, _ in loader:
            B, L, a, s, c = csi_b.shape
            z = encoder(csi_b.view(B * L, a, s, c).to(device))
            all_z_seq.append(z.view(B, L, -1))
        all_z_seq = torch.cat(all_z_seq, dim=0)  # (N_seq, L, d)

        # Errore di predizione lineare
        z_t = all_z_seq[:, :-1, :]
        z_tp1 = all_z_seq[:, 1:, :]
        pred = z_t @ W.T
        pred_err = (z_tp1 - pred).pow(2).sum(-1).sqrt()
        pred_err_mean = pred_err.mean().item()
        pred_err_std = pred_err.std().item()

        # Norma del residuo g_phi
        all_z_flat = all_z_seq.view(-1, all_z_seq.shape[-1])
        g_norms = residual(all_z_flat).norm(dim=-1).mean().item()

    # Spettro di W*
    eigvals = torch.linalg.eigvals(W).cpu().numpy()
    mags = np.abs(eigvals)
    spectral_radius = float(mags.max())
    effective_rank = int((mags > 0.1).sum())

    return {
        "pred_error_mean": pred_err_mean,
        "pred_error_std": pred_err_std,
        "residual_norm": g_norms,
        "spectral_radius": spectral_radius,
        "effective_rank": effective_rank,
    }