"""
phase3_ablation.py
==================

Hyperparameter ablation on cf0x (λ=20, seed 0): latent dimension d,
sequence length L, and Wiener-Hopf re-estimation interval Δ. One factor is
varied at a time around the default configuration (d=8, L=16, Δ=5).
Saves to results/phase3_ablation.json.
"""

import json
import os

from src.run_experiment import train_warden

DATA_ROOT = os.environ.get("WARDEN_DATA", "data")
CF0X = f"{DATA_ROOT}/cf0x"
RESULTS_FILE = "results/phase3_ablation.json"

LAM, SEED, GAP = 20.0, 0, 750

# Default (d=8, L=16, Δ=5) is covered by the d8 row.
CONFIGS = (
    [{"d": d, "L": 16, "delta": 5, "tag": f"d{d}"} for d in (2, 4, 8, 16)]
    + [{"d": 8, "L": L, "delta": 5, "tag": f"L{L}"} for L in (8, 32)]
    + [{"d": 8, "L": 16, "delta": dl, "tag": f"delta{dl}"} for dl in (1, 10, 20)]
)


def main():
    os.makedirs("results", exist_ok=True)
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} configs already done")

    for cfg in CONFIGS:
        key = cfg["tag"]
        if key in results:
            print(f"SKIP {key}")
            continue
        # stride proportional to L keeps the number of sequences comparable
        stride = max(2, cfg["L"] // 4)
        print(f"RUN  {key} (d={cfg['d']}, L={cfg['L']}, "
              f"Delta={cfg['delta']}) ...", flush=True)
        results[key] = train_warden(
            CF0X, LAM, seed=SEED, d=cfg["d"], seq_len=cfg["L"],
            stride=stride, max_gap_ms=GAP, delta=cfg["delta"])
        results[key].update(cfg)
        r = results[key]
        print(f"     MAE={r['mae']:.3f}  TW={r['tw']:.4f}  "
              f"erank={r['erank']:.2f}")
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone: {len(results)} configs")


if __name__ == "__main__":
    main()
