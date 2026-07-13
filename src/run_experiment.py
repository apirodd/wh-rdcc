"""
run_experiment.py
=================

Single WARDEN training run (two-stage: bootstrap + joint) with automatic
mixed precision, on one dataset for a given lambda and seed. Returns the
evaluation metrics used throughout the paper.

Example:
    python -m src.run_experiment --data_dir data/cf0x --lam 20 --seed 0

The data_dir must contain train/test_{csi,positions,timestamps}.npy as
produced by the preprocessing scripts.
"""

import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import trustworthiness

from warden_env import (
    CSIEncoder, ResidualNet, DICHASUSDataset, LATENT_DIM, device,
    estimate_wiener_hopf, contrastive_loss, extract_latents,
    extract_all_z_seq, affine_align, effective_rank,
)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_warden(data_dir, lam, seed=0, d=LATENT_DIM, seq_len=16,
                 stride=4, max_gap_ms=750, delta=5,
                 bootstrap_ep=40, joint_ep=100, batch_size=32):
    """Two-stage WARDEN training. Returns a metrics dict."""
    set_seed(seed)
    torch.backends.cudnn.benchmark = True

    tr = DICHASUSDataset(f"{data_dir}/train_csi.npy",
                         f"{data_dir}/train_positions.npy",
                         f"{data_dir}/train_timestamps.npy",
                         seq_len=seq_len, stride=stride, max_gap_ms=max_gap_ms)
    te = DICHASUSDataset(f"{data_dir}/test_csi.npy",
                         f"{data_dir}/test_positions.npy",
                         f"{data_dir}/test_timestamps.npy",
                         seq_len=seq_len, stride=8, max_gap_ms=max_gap_ms)
    g = torch.Generator(); g.manual_seed(seed)
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True,
                    num_workers=2, pin_memory=True, generator=g)
    el = DataLoader(te, batch_size=batch_size, shuffle=False,
                    num_workers=2, pin_memory=True)

    enc = CSIEncoder(d).to(device)
    res = ResidualNet(d).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(res.parameters()),
                           lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=bootstrap_ep + joint_ep, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    W = torch.eye(d, device=device)

    # ---- Phase 1: bootstrap (charting loss only) --------------------------
    for _ in range(bootstrap_ep):
        enc.train()
        for csi_b, pos_b in tl:
            B, L, a, s, c = csi_b.shape
            with torch.amp.autocast("cuda"):
                z = enc(csi_b.view(B * L, a, s, c).to(device)).view(B, L, -1)
                loss = contrastive_loss(z, pos_b.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()
        sch.step()

    with torch.no_grad():
        W = estimate_wiener_hopf(extract_all_z_seq(enc, tl).float())

    # ---- Phase 2: joint (charting + Wiener-Hopf residual loss) ------------
    for ep in range(1, joint_ep + 1):
        enc.train()
        for csi_b, pos_b in tl:
            B, L, a, s, c = csi_b.shape
            with torch.amp.autocast("cuda"):
                z_seq = enc(csi_b.view(B * L, a, s, c).to(device)).view(B, L, -1)
                l_cc = contrastive_loss(z_seq, pos_b.to(device))
                z_t, z_tp1 = z_seq[:, :-1, :], z_seq[:, 1:, :]
                l_wh = (z_tp1 - z_t @ W.T.to(z_seq.dtype)
                        - res(z_t.float()).to(z_seq.dtype)).pow(2).mean()
                loss = l_cc + lam * l_wh
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()
        sch.step()
        if ep % delta == 0:
            with torch.no_grad():
                W = estimate_wiener_hopf(extract_all_z_seq(enc, tl).float())

    # ---- Evaluation -------------------------------------------------------
    z_tr, p_tr = extract_latents(enc, tl)
    z_te, p_te = extract_latents(enc, el)
    p_pred = affine_align(z_tr, p_tr, z_te)
    mae = float(np.mean(np.linalg.norm(p_pred - p_te, axis=1)))
    tw = float(trustworthiness(p_tr, z_tr, n_neighbors=10))
    ct = float(trustworthiness(z_tr, p_tr, n_neighbors=10))
    with torch.no_grad():
        all_z = extract_all_z_seq(enc, tl).float()
        z_t, z_tp1 = all_z[:, :-1, :], all_z[:, 1:, :]
        pe = float((z_tp1 - z_t @ W.T).pow(2).sum(-1).sqrt().mean())
    sr = float(torch.abs(torch.linalg.eigvals(W)).max())
    erank = effective_rank(z_tr)

    return {"mae": mae, "tw": tw, "ct": ct, "pe": pe,
            "sr": sr, "erank": erank}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--lam", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=LATENT_DIM)
    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max_gap_ms", type=int, default=750)
    ap.add_argument("--delta", type=int, default=5)
    args = ap.parse_args()

    metrics = train_warden(
        args.data_dir, args.lam, seed=args.seed, d=args.d,
        seq_len=args.seq_len, stride=args.stride,
        max_gap_ms=args.max_gap_ms, delta=args.delta)
    print(f"lambda={args.lam} seed={args.seed}  "
          + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
