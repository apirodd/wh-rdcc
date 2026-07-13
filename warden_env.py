"""
warden_env.py
=============

Core module for WARDEN (Wiener-Adaptive Residual Dynamic ENcoder),
a Channel Charting architecture that integrates an analytically optimal
Wiener-Hopf transition matrix as a structured prior for latent-space
dynamics.

Paper:  A. Piroddi, T. Kanakis, M. Opoku Agyeman,
        "WARDEN: Wiener-Adaptive Residual Dynamic ENcoder for Channel
        Charting in Future Cellular Networks."

This module contains the building blocks shared across all experiments:
    - CSIEncoder        : CNN encoder f_theta (antenna-count agnostic)
    - ResidualNet       : residual MLP g_phi
    - DICHASUSDataset   : temporal-sequence dataset over CSI .npy files
    - estimate_wiener_hopf : closed-form MMSE transition matrix W*
    - contrastive_loss  : NT-Xent charting loss (AMP-safe)
    - extract_latents / extract_all_z_seq : evaluation helpers
    - affine_align      : latent -> physical affine anchoring

Usage:
    from warden_env import *
    # then build loaders, call estimate_wiener_hopf(...), etc.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from numpy.linalg import lstsq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 8


# ---------------------------------------------------------------------------
# Encoder f_theta
# ---------------------------------------------------------------------------
class CSIEncoder(nn.Module):
    """CNN encoder mapping a CSI snapshot to an l2-normalised latent vector.

    Input : (B, A, K, 2) real-valued CSI (A antennas, K subcarriers, re/im).
    Output: (B, d) l2-normalised latent.

    Global average pooling makes the encoder agnostic to the antenna count A,
    so the same architecture handles 24- and 32-antenna arrays unchanged.
    """

    def __init__(self, latent_dim: int = 8):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 16, (1, 7), (1, 2), (0, 3)),
            nn.BatchNorm2d(16), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (1, 7), (1, 2), (0, 3)),
            nn.BatchNorm2d(32), nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 7), (1, 2), (0, 3)),
            nn.BatchNorm2d(64), nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, (2, 7), (2, 2), (0, 3)),
            nn.BatchNorm2d(64), nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, (2, 7), (2, 4), (0, 3)),
            nn.BatchNorm2d(128), nn.ReLU())
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)                       # (B,2,A,K)
        for layer in (self.conv1, self.conv2, self.conv3,
                      self.conv4, self.conv5):
            x = layer(x)
        return F.normalize(self.proj(self.gap(x)), dim=1)


# ---------------------------------------------------------------------------
# Residual network g_phi
# ---------------------------------------------------------------------------
class ResidualNet(nn.Module):
    """Residual MLP predicting the non-linear correction to the linear
    Wiener-Hopf prediction: g_phi(z_t) ~ z_{t+1} - W* z_t."""

    def __init__(self, latent_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, latent_dim))

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DICHASUSDataset(Dataset):
    """Temporal-sequence dataset over pre-processed CSI stored as .npy.

    Expects three arrays produced by the preprocessing scripts:
        csi_path  : (N, A, K, 2) float32, per-sample normalised
        pos_path  : (N, 2) float32 ground-truth positions
        time_path : (N,) float64 timestamps [s]

    A sequence of length `seq_len` is valid only if all consecutive time
    gaps are positive and below `max_gap_ms`, which excludes discontinuities
    (e.g. joins between measurement files or docking pauses).
    """

    def __init__(self, csi_path, pos_path, time_path,
                 seq_len=16, stride=4, max_gap_ms=750):
        self.csi = np.load(csi_path, mmap_mode="r")
        self.positions = np.load(pos_path)
        self.timestamps = np.load(time_path)
        self.seq_len = seq_len
        self.valid = []
        for i in range(0, len(self.timestamps) - seq_len + 1, stride):
            gaps = np.diff(self.timestamps[i:i + seq_len]) * 1000
            if np.all(gaps <= max_gap_ms) and np.all(gaps > 0):
                self.valid.append(i)

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        s = self.valid[idx]
        return (torch.from_numpy(self.csi[s:s + self.seq_len].copy()),
                torch.from_numpy(self.positions[s:s + self.seq_len]))


# ---------------------------------------------------------------------------
# Wiener-Hopf transition matrix (closed form)
# ---------------------------------------------------------------------------
def estimate_wiener_hopf(z_seq, eps=1e-4):
    """MMSE-optimal linear transition matrix W* for a latent sequence.

    W* = R_zz(1) (R_zz(0) + eps I)^-1, with Tikhonov regularisation
    eps scaled by tr(R_zz(0))/d. Solved as a linear system for stability.

    Args:
        z_seq: (B, L, d) tensor of latent sequences.
    Returns:
        (d, d) tensor W*.
    """
    B, L, d = z_seq.shape
    z_t = z_seq[:, :-1, :].reshape(-1, d)
    z_tp1 = z_seq[:, 1:, :].reshape(-1, d)
    N = z_t.shape[0]
    Rzz0 = (z_t.T @ z_t) / N
    Rzz1 = (z_tp1.T @ z_t) / N
    reg = eps * torch.trace(Rzz0) / d
    return torch.linalg.lstsq(
        Rzz0 + reg * torch.eye(d, device=z_seq.device), Rzz1.T).solution.T


# ---------------------------------------------------------------------------
# Contrastive charting loss (NT-Xent), AMP-safe
# ---------------------------------------------------------------------------
def contrastive_loss(z_seq, pos_seq=None, temp=0.07):
    """NT-Xent loss using the first two frames of each sequence as an
    anchor/positive pair; all other in-batch samples are negatives.

    The diagonal is masked with the dtype minimum (not a hard -1e9) so the
    loss is safe under automatic mixed precision (fp16).
    """
    B, L, d = z_seq.shape
    z_a = z_seq[:, 0, :]
    z_p = z_seq[:, 1, :]
    z_all = torch.cat([z_a, z_p], dim=0)
    sim = torch.mm(z_all, z_all.T) / temp
    mask = torch.eye(2 * B, device=z_seq.device).bool()
    sim.masked_fill_(mask, torch.finfo(sim.dtype).min)
    labels = torch.arange(B, device=z_seq.device)
    labels = torch.cat([labels + B, labels])
    return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_latents(encoder, loader, device=device):
    """Return (Z, P): per-sequence anchor-frame latents and positions."""
    encoder.eval()
    zs, ps = [], []
    for csi_b, pos_b in loader:
        zs.append(encoder(csi_b[:, 0].to(device)).cpu().numpy())
        ps.append(pos_b[:, 0].numpy())
    return np.concatenate(zs), np.concatenate(ps)


@torch.no_grad()
def extract_all_z_seq(encoder, loader, device=device):
    """Return (B, L, d) latents for every sequence (used to estimate W*)."""
    encoder.eval()
    out = []
    for csi_b, _ in loader:
        B, L, a, s, c = csi_b.shape
        z = encoder(csi_b.view(B * L, a, s, c).to(device))
        out.append(z.view(B, L, -1))
    return torch.cat(out, dim=0)


def affine_align(z_src, p_src, z_tgt):
    """Fit an affine map (latent -> physical) on the source set by least
    squares and apply it to the target latents."""
    Z = np.hstack([z_src, np.ones((len(z_src), 1))])
    AB, _, _, _ = lstsq(Z, p_src, rcond=None)
    return z_tgt @ AB[:-1] + AB[-1]


def effective_rank(z):
    """Entropy-based effective rank of the centred latents' singular-value
    spectrum; detects dimensional collapse (returns a value in [1, d])."""
    z = z - z.mean(0)
    s = np.linalg.svd(z, compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))
