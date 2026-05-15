#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run.py — Script principale per eseguire WH-RDCC su DICHASUS"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import dei moduli locali
from src.preprocessing import download_all, preprocess_all, assemble_train_test
from src.dataset import DICHASUSDataset
from src.train import train_wh_rdcc
from src.evaluate import extract_latents, affine_align, compute_metrics, compute_wiener_hopf_metrics

# Configurazione
DATA_OUT = "/content/drive/MyDrive/WH_RDCC/data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Parametri (ottimali da ablation)
CONFIG = {
    "seq_len": 16,
    "stride": 4,
    "max_gap_ms": 750,
    "latent_dim": 8,
    "lambda_wh": 5.0,  # ottimo da ablation
    "bootstrap_epochs": 40,
    "joint_epochs": 100,
    "wh_update_every": 5,
    "batch_size": 32,
    "lr": 1e-3,
}


def main():
    # 1. Preprocessing (solo prima volta)
    if not os.path.exists(f"{DATA_OUT}/train_csi.npy"):
        print("Esecuzione preprocessing...")
        download_all()
        preprocess_all()
        assemble_train_test()

    # 2. Carica dataset
    print("\nCaricamento dataset...")
    train_ds = DICHASUSDataset(
        f"{DATA_OUT}/train_csi.npy",
        f"{DATA_OUT}/train_positions.npy",
        f"{DATA_OUT}/train_timestamps.npy",
        seq_len=CONFIG["seq_len"],
        stride=CONFIG["stride"],
        max_gap_ms=CONFIG["max_gap_ms"],
    )
    test_ds = DICHASUSDataset(
        f"{DATA_OUT}/test_csi.npy",
        f"{DATA_OUT}/test_positions.npy",
        f"{DATA_OUT}/test_timestamps.npy",
        seq_len=CONFIG["seq_len"],
        stride=CONFIG["stride"],
        max_gap_ms=CONFIG["max_gap_ms"],
    )

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True
    )

    # 3. Training
    print("\nAvvio training WH-RDCC...")
    encoder, residual, W, history = train_wh_rdcc(
        train_loader,
        latent_dim=CONFIG["latent_dim"],
        lambda_wh=CONFIG["lambda_wh"],
        bootstrap_epochs=CONFIG["bootstrap_epochs"],
        joint_epochs=CONFIG["joint_epochs"],
        wh_update_every=CONFIG["wh_update_every"],
        lr=CONFIG["lr"],
        device=DEVICE,
    )

    # 4. Salvataggio modello
    torch.save(
        {"encoder": encoder.state_dict(), "residual": residual.state_dict(), "W": W, "history": history},
        f"{DATA_OUT}/wh_rdcc_model_final.pt",
    )
    print("Modello salvato.")

    # 5. Valutazione
    print("\nValutazione...")
    z_train, p_train = extract_latents(encoder, train_loader, DEVICE)
    z_test, p_test = extract_latents(encoder, test_loader, DEVICE)

    metrics = compute_metrics(z_train, p_train, z_test, p_test)
    wh_metrics = compute_wiener_hopf_metrics(encoder, residual, W, train_loader, DEVICE)

    print("\n=== RISULTATI FINALI ===")
    print(f"MAE test            : {metrics['mae_test']:.3f} m")
    print(f"Trustworthiness     : {metrics['trustworthiness']:.4f}")
    print(f"Continuity          : {metrics['continuity']:.4f}")
    print(f"Kruskal Stress      : {metrics['kruskal_stress']:.4f}")
    print(f"Spectral radius W*  : {wh_metrics['spectral_radius']:.4f}")
    print(f"Effective rank W*   : {wh_metrics['effective_rank']}")
    print(f"Pred error (linear) : {wh_metrics['pred_error_mean']:.4f} ± {wh_metrics['pred_error_std']:.4f}")
    print(f"Residual norm g_phi : {wh_metrics['residual_norm']:.4f}")


if __name__ == "__main__":
    main()