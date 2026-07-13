"""
phase5_overhead.py
==================

Quantifies the computational overhead of WARDEN relative to the static
baseline: Wiener-Hopf solve time, per-epoch loss overhead, full W*
re-estimation cost, and per-frame inference time. Saves to
results/phase5_overhead.json.
"""

import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from warden_env import (
    CSIEncoder, ResidualNet, DICHASUSDataset, LATENT_DIM, device,
    estimate_wiener_hopf, contrastive_loss, extract_all_z_seq,
)

DATA_ROOT = os.environ.get("WARDEN_DATA", "data")
CF0X = f"{DATA_ROOT}/cf0x"
RESULTS_FILE = "results/phase5_overhead.json"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    os.makedirs("results", exist_ok=True)
    torch.backends.cudnn.benchmark = True

    tr = DICHASUSDataset(f"{CF0X}/train_csi.npy",
                         f"{CF0X}/train_positions.npy",
                         f"{CF0X}/train_timestamps.npy",
                         seq_len=16, stride=4, max_gap_ms=750)
    tl = DataLoader(tr, batch_size=32, shuffle=True,
                    num_workers=2, pin_memory=True)

    enc = CSIEncoder(LATENT_DIM).to(device)
    res = ResidualNet(LATENT_DIM).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(res.parameters()),
                           lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    W = torch.eye(LATENT_DIM, device=device)

    # Warm-up
    for i, (csi_b, pos_b) in enumerate(tl):
        if i >= 3:
            break
        B, L, a, s, c = csi_b.shape
        with torch.amp.autocast("cuda"):
            z = enc(csi_b.view(B * L, a, s, c).to(device)).view(B, L, -1)
            loss = contrastive_loss(z, pos_b.to(device))
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); opt.zero_grad()

    T = {}

    # (1) W* re-estimation cost
    sync(); t0 = time.time()
    with torch.no_grad():
        all_z = extract_all_z_seq(enc, tl).float()
    sync()
    t_extract = time.time() - t0
    t0 = time.time()
    with torch.no_grad():
        for _ in range(100):
            W = estimate_wiener_hopf(all_z)
    sync()
    t_solve = (time.time() - t0) / 100
    T["wh_solve_ms"] = round(t_solve * 1000, 3)
    T["wh_reestimation_s"] = round(t_extract + t_solve, 3)

    # (2) Per-epoch time: baseline vs WARDEN
    def time_epoch(with_wh):
        sync(); t0 = time.time()
        enc.train()
        for csi_b, pos_b in tl:
            B, L, a, s, c = csi_b.shape
            with torch.amp.autocast("cuda"):
                z_seq = enc(csi_b.view(B * L, a, s, c).to(device)).view(B, L, -1)
                loss = contrastive_loss(z_seq, pos_b.to(device))
                if with_wh:
                    z_t, z_tp1 = z_seq[:, :-1, :], z_seq[:, 1:, :]
                    loss = loss + 20.0 * (
                        z_tp1 - z_t @ W.T.to(z_seq.dtype)
                        - res(z_t.float()).to(z_seq.dtype)).pow(2).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()
        sync()
        return time.time() - t0

    t_base, t_ward = [], []
    for _ in range(3):
        t_base.append(time_epoch(False))
        t_ward.append(time_epoch(True))
    T["epoch_baseline_s"] = round(float(np.mean(t_base)), 2)
    T["epoch_warden_s"] = round(float(np.mean(t_ward)), 2)
    T["loss_overhead_pct"] = round(
        100 * (np.mean(t_ward) - np.mean(t_base)) / np.mean(t_base), 1)

    # (3) Per-frame inference
    enc.eval()
    x1 = torch.randn(1, 32, 1024, 2).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = enc(x1)
        sync(); t0 = time.time()
        for _ in range(200):
            _ = enc(x1)
        sync()
    T["inference_per_frame_ms"] = round((time.time() - t0) / 200 * 1000, 2)

    T["params_encoder"] = sum(p.numel() for p in enc.parameters())
    T["params_residual"] = sum(p.numel() for p in res.parameters())
    if torch.cuda.is_available():
        T["gpu"] = torch.cuda.get_device_name(0)

    with open(RESULTS_FILE, "w") as f:
        json.dump(T, f, indent=2)
    print(json.dumps(T, indent=2))


if __name__ == "__main__":
    main()
