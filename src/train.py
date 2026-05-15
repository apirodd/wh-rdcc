# -*- coding: utf-8 -*-
"""train.py — Two-phase training loop per WH-RDCC"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .encoder import CSIEncoder
from .residual_net import ResidualNet
from .wiener_hopf import estimate_wiener_hopf
from .losses import contrastive_loss, wiener_hopf_loss


def extract_all_z_seq(encoder: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Estrae tutte le sequenze latenti dal dataset (per stima W*)."""
    encoder.eval()
    out = []
    with torch.no_grad():
        for csi_b, _ in loader:
            B, L, a, s, c = csi_b.shape
            z = encoder(csi_b.view(B * L, a, s, c).to(device))
            out.append(z.view(B, L, -1))
    return torch.cat(out, dim=0)


def train_wh_rdcc(
    train_loader: DataLoader,
    latent_dim: int = 8,
    lambda_wh: float = 5.0,
    bootstrap_epochs: int = 40,
    joint_epochs: int = 100,
    wh_update_every: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = None,
) -> tuple:
    """
    Two-phase training per WH-RDCC.

    Phase 1 (bootstrap): solo L_CC per E0 epoche.
    Phase 2 (joint): L_CC + λ L_WH per E1 epoche.

    Returns:
        encoder, residual, W, history
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = CSIEncoder(latent_dim).to(device)
    residual = ResidualNet(latent_dim).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(residual.parameters()), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=bootstrap_epochs + joint_epochs, eta_min=1e-5
    )

    W = torch.eye(latent_dim, device=device)
    history = {"cc": [], "wh": [], "phase": []}

    # ==================== FASE 1: Bootstrap ====================
    print(f"=== FASE 1: Bootstrap ({bootstrap_epochs} epoche) ===")
    for ep in range(1, bootstrap_epochs + 1):
        encoder.train()
        total_cc = 0.0
        n_batches = 0

        for csi_b, pos_b in train_loader:
            B, L, a, s, c = csi_b.shape
            z = encoder(csi_b.view(B * L, a, s, c).to(device)).view(B, L, -1)
            loss = contrastive_loss(z)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

            total_cc += loss.item()
            n_batches += 1

        scheduler.step()
        avg_cc = total_cc / n_batches
        history["cc"].append(avg_cc)
        history["wh"].append(0.0)
        history["phase"].append("bootstrap")

        if ep % 10 == 0:
            print(f"  Ep {ep:3d}/{bootstrap_epochs}  L_CC={avg_cc:.4f}")

    # Stima W* iniziale
    print("Stima W* iniziale...")
    with torch.no_grad():
        all_z = extract_all_z_seq(encoder, train_loader, device)
        W = estimate_wiener_hopf(all_z)
    sr = torch.abs(torch.linalg.eigvals(W)).max().item()
    print(f"  Spectral radius W*: {sr:.4f}")

    # ==================== FASE 2: Joint ====================
    print(f"\n=== FASE 2: Joint ({joint_epochs} epoche, λ={lambda_wh}) ===")
    for ep in range(1, joint_epochs + 1):
        encoder.train()
        residual.train()
        total_cc = 0.0
        total_wh = 0.0
        n_batches = 0

        for csi_b, pos_b in train_loader:
            B, L, a, s, c = csi_b.shape
            z_flat = encoder(csi_b.view(B * L, a, s, c).to(device))
            z_seq = z_flat.view(B, L, -1)

            l_cc = contrastive_loss(z_seq)
            l_wh = wiener_hopf_loss(z_seq, W)
            loss = l_cc + lambda_wh * l_wh

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(residual.parameters()), 1.0)
            optimizer.step()

            total_cc += l_cc.item()
            total_wh += l_wh.item()
            n_batches += 1

        scheduler.step()
        avg_cc = total_cc / n_batches
        avg_wh = total_wh / n_batches
        history["cc"].append(avg_cc)
        history["wh"].append(avg_wh)
        history["phase"].append("joint")

        # Aggiornamento periodico di W*
        if ep % wh_update_every == 0:
            with torch.no_grad():
                all_z = extract_all_z_seq(encoder, train_loader, device)
                W = estimate_wiener_hopf(all_z)
            sr = torch.abs(torch.linalg.eigvals(W)).max().item()

        if ep % 10 == 0:
            print(f"  Ep {ep:3d}/{joint_epochs}  L_CC={avg_cc:.4f}  L_WH={avg_wh:.4f}  sr={sr:.4f}")

    return encoder, residual, W, history