"""
phase2_statistics.py
====================

Multi-seed, multi-dataset statistical runs reproducing the main results
table of the paper. Saves incrementally to results/phase2_results.json and
skips already-completed runs (safe to interrupt and resume).

Configure DATA_ROOT to point at the directory containing the preprocessed
dataset folders (cf0x, cf12, 015x, deepmimo).
"""

import json
import os

from src.run_experiment import train_warden

DATA_ROOT = os.environ.get("WARDEN_DATA", "data")
RESULTS_FILE = "results/phase2_results.json"

# Per-dataset sequence-construction parameters (stride, max gap [ms]).
DATASETS = {
    "cf0x":     {"dir": f"{DATA_ROOT}/cf0x",     "stride": 4, "gap": 750},
    "cf12":     {"dir": f"{DATA_ROOT}/cf12",     "stride": 2, "gap": 1000},
    "015x":     {"dir": f"{DATA_ROOT}/015x",     "stride": 2, "gap": 1000},
    "deepmimo": {"dir": f"{DATA_ROOT}/deepmimo", "stride": 2, "gap": 300},
}
LAMBDAS = [0.0, 5.0, 20.0]
SEEDS = {"cf0x": [0, 1, 2, 3, 4], "cf12": [0, 1, 2],
         "015x": [0, 1, 2], "deepmimo": [0, 1, 2]}


def main():
    os.makedirs("results", exist_ok=True)
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} runs already done")

    for name, cfg in DATASETS.items():
        for lam in LAMBDAS:
            for seed in SEEDS[name]:
                key = f"{name}_lam{lam}_seed{seed}"
                if key in results:
                    print(f"SKIP {key}")
                    continue
                print(f"RUN  {key} ...", flush=True)
                results[key] = train_warden(
                    cfg["dir"], lam, seed=seed,
                    stride=cfg["stride"], max_gap_ms=cfg["gap"])
                r = results[key]
                print(f"     MAE={r['mae']:.3f}  TW={r['tw']:.4f}  "
                      f"PE={r['pe']:.4f}")
                with open(RESULTS_FILE, "w") as f:
                    json.dump(results, f, indent=2)

    print(f"\nDone: {len(results)} runs")


if __name__ == "__main__":
    main()
