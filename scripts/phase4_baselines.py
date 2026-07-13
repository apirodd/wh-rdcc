"""
phase4_baselines.py
===================

Learned temporal-mechanism baselines under a unified protocol on cf0x,
sharing the same CNN backbone, optimiser, schedule, and data pipeline as
WARDEN so that only the temporal mechanism differs:

    - JEPA-CC : online encoder + EMA target encoder + MLP predictor,
                next-step latent prediction with stop-gradient.
    - LSTM-CC : per-frame CNN + LSTM producing per-timestep latents,
                trained with the same NT-Xent charting loss.

Reports an effective-rank diagnostic to detect dimensional collapse.
Saves to results/phase4_baselines.json.
"""

import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.manifold import trustworthiness

from warden_env import (
    CSIEncoder, DICHASUSDataset, LATENT_DIM, device,
    estimate_wiener_hopf, contrastive_loss,
    extract_latents, extract_all_z_seq, affine_align, effective_rank,
)

DATA_ROOT = os.environ.get("WARDEN_DATA", "data")
CF0X = f"{DATA_ROOT}/cf0x"
RESULTS_FILE = "results/phase4_baselines.json"
SEEDS = [0, 1, 2]
EPOCHS = 140


def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)


def make_loaders(seed):
    tr = DICHASUSDataset(f"{CF0X}/train_csi.npy",
                         f"{CF0X}/train_positions.npy",
                         f"{CF0X}/train_timestamps.npy",
                         seq_len=16, stride=4, max_gap_ms=750)
    te = DICHASUSDataset(f"{CF0X}/test_csi.npy",
                         f"{CF0X}/test_positions.npy",
                         f"{CF0X}/test_timestamps.npy",
                         seq_len=16, stride=8, max_gap_ms=750)
    g = torch.Generator(); g.manual_seed(seed)
    tl = DataLoader(tr, batch_size=32, shuffle=True,
                    num_workers=2, pin_memory=True, generator=g)
    el = DataLoader(te, batch_size=32, shuffle=False,
                    num_workers=2, pin_memory=True)
    return tl, el


def evaluate(enc, tl, el):
    z_tr, p_tr = extract_latents(enc, tl)
    z_te, p_te = extract_latents(enc, el)
    p_pred = affine_align(z_tr, p_tr, z_te)
    mae = float(np.mean(np.linalg.norm(p_pred - p_te, axis=1)))
    tw = float(trustworthiness(p_tr, z_tr, n_neighbors=10))
    ct = float(trustworthiness(z_tr, p_tr, n_neighbors=10))
    erank = effective_rank(z_tr)
    with torch.no_grad():
        all_z = extract_all_z_seq(enc, tl).float()
        W = estimate_wiener_hopf(all_z)
        z_t, z_tp1 = all_z[:, :-1, :], all_z[:, 1:, :]
        pe = float((z_tp1 - z_t @ W.T).pow(2).sum(-1).sqrt().mean())
        sr = float(torch.abs(torch.linalg.eigvals(W)).max())
    return {"mae": mae, "tw": tw, "ct": ct, "pe": pe, "sr": sr,
            "erank": erank}


# --------------------------------------------------------------------------
# JEPA-CC
# --------------------------------------------------------------------------
class Predictor(nn.Module):
    def __init__(self, d=8, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d))

    def forward(self, z):
        return self.net(z)


def run_jepa(seed, ema_m=0.996):
    set_seed(seed)
    tl, el = make_loaders(seed)
    enc = CSIEncoder(LATENT_DIM).to(device)
    enc_ema = copy.deepcopy(enc)
    for p in enc_ema.parameters():
        p.requires_grad = False
    pred = Predictor(LATENT_DIM).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()),
                           lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS,
                                                     eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    for _ in range(EPOCHS):
        enc.train()
        for csi_b, _ in tl:
            B, L, a, s, c = csi_b.shape
            x = csi_b.view(B * L, a, s, c).to(device)
            with torch.amp.autocast("cuda"):
                z_on = enc(x).view(B, L, -1)
                with torch.no_grad():
                    z_tg = enc_ema(x).view(B, L, -1)
                z_hat = pred(z_on[:, :-1, :])
                loss = (z_hat - z_tg[:, 1:, :].detach()).pow(2).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(pred.parameters()), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()
            with torch.no_grad():
                for p_on, p_tg in zip(enc.parameters(), enc_ema.parameters()):
                    p_tg.mul_(ema_m).add_(p_on, alpha=1 - ema_m)
        sch.step()
    return evaluate(enc, tl, el)


# --------------------------------------------------------------------------
# LSTM-CC
# --------------------------------------------------------------------------
class LSTMEncoder(nn.Module):
    """Per-frame CNN trunk + LSTM producing per-timestep latents."""

    def __init__(self, d=8, feat=128, hidden=64):
        super().__init__()
        base = CSIEncoder(d)
        self.conv = nn.Sequential(base.conv1, base.conv2, base.conv3,
                                  base.conv4, base.conv5, base.gap,
                                  nn.Flatten())
        self.lstm = nn.LSTM(feat, hidden, batch_first=True)
        self.to_z = nn.Linear(hidden, d)

    def forward(self, x_seq):
        B, L, a, s, c = x_seq.shape
        f = self.conv(x_seq.view(B * L, a, s, c).permute(0, 3, 1, 2))
        f = f.view(B, L, -1)
        h, _ = self.lstm(f)
        return F.normalize(self.to_z(h), dim=-1)


def run_lstm_cc(seed):
    set_seed(seed)
    tl, el = make_loaders(seed)
    model = LSTMEncoder(LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS,
                                                     eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    for _ in range(EPOCHS):
        model.train()
        for csi_b, pos_b in tl:
            with torch.amp.autocast("cuda"):
                z_seq = model(csi_b.to(device))
                loss = contrastive_loss(z_seq, pos_b.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()
        sch.step()

    class EncAdapter(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m

        def forward(self, x):
            return self.m(x.unsqueeze(1))[:, 0]

        def eval(self):
            self.m.eval(); return self

    return evaluate(EncAdapter(model), tl, el)


def main():
    os.makedirs("results", exist_ok=True)
    methods = {"jepa": run_jepa, "lstm_cc": run_lstm_cc}
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} runs already done")

    for mname, fn in methods.items():
        for seed in SEEDS:
            key = f"{mname}_seed{seed}"
            if key in results:
                print(f"SKIP {key}")
                continue
            print(f"RUN  {key} ...", flush=True)
            results[key] = fn(seed)
            r = results[key]
            print(f"     MAE={r['mae']:.3f}  TW={r['tw']:.4f}  "
                  f"erank={r['erank']:.2f}")
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nDone: {len(results)} runs")


if __name__ == "__main__":
    main()
