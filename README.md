# WARDEN: Wiener-Adaptive Residual Dynamic ENcoder

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Official implementation of:

> **WARDEN: Wiener-Adaptive Residual Dynamic ENcoder for Channel Charting in Future Cellular Networks**
> Andrea Piroddi, Triantafyllos Kanakis, Michael Opoku Agyeman

WARDEN is a self-supervised Channel Charting architecture that integrates an
**analytically optimal Wiener-Hopf transition matrix** as a structured prior
for latent-space dynamics. Instead of learning the latent transition model
entirely from data (as in JEPA- or LSTM-based temporal Channel Charting),
WARDEN decomposes the prediction problem into a closed-form linear component
and a learned non-linear residual:

```
z_{t+1} ≈ W* z_t + g_φ(z_t)
         └──┬──┘   └───┬───┘
       Wiener-Hopf   residual MLP
       (closed form)  (learned)
```

The linear operator `W*` is the minimum mean-square-error one-step predictor,
recomputed in closed form from the second-order statistics of the latent
sequence; the residual network `g_φ` only has to capture the genuinely
non-linear part of the dynamics.

---

## Key results

Validated on four environments (three real-world massive MIMO datasets and one
synthetic ray-tracing scenario) with **identical hyperparameters** and multiple
random seeds. Trustworthiness (TW), Continuity (CT), and linear prediction
error (PE) improve **monotonically** with the regularisation strength `λ` on
all four environments; localisation MAE improves on all three real datasets.

| Dataset | λ | MAE [m] | TW | CT | PE |
|---|---|---|---|---|---|
| cf0x (distributed, factory) | 0 | 3.101±0.141 | 0.858 | 0.923 | 0.243 |
| | 20 | **2.748±0.180** | **0.917** | **0.925** | **0.157** |
| cf12 (distributed, 24 ant.) | 0 | 4.064±0.103 | 0.883 | 0.959 | 0.400 |
| | 20 | **3.618±0.288** | **0.980** | **0.992** | **0.210** |
| 015x (co-located, lab) | 0 | 1.632±0.043 | 0.828 | 0.882 | 0.531 |
| | 20 | **1.460±0.045** | **0.933** | **0.960** | **0.296** |
| DeepMIMO (outdoor, synthetic) | 0 | 114.7±1.4 | 0.885 | 0.975 | 0.177 |
| | 20 | 112.6±5.1 | **0.996** | **0.999** | **0.069** |

Under a unified protocol against learned temporal mechanisms (LSTM-CC,
JEPA-CC), WARDEN shows the **lowest seed variance** and is the **only** method
that improves the latent dynamics without degrading any geometric metric,
while adding **no inference overhead**.

---

## Repository structure

```
wh-rdcc/
├── warden_env.py                  # core: encoder, W*, loss, evaluation
├── src/
│   ├── preprocessing_dichasus.py  # DICHASUS TFRecords -> .npy
│   ├── preprocessing_deepmimo.py  # DeepMIMO scenario -> .npy
│   └── run_experiment.py          # single WARDEN training run
├── scripts/
│   ├── phase2_statistics.py       # multi-seed statistical runs
│   ├── phase3_ablation.py         # ablation on d, L, Δ
│   ├── phase4_baselines.py        # LSTM-CC and JEPA-CC baselines
│   └── phase5_overhead.py         # computational-overhead timing
├── notebooks/
│   └── 01_full_pipeline.ipynb     # end-to-end demo (Colab-ready)
├── results/                       # JSON metrics reproducing the paper tables
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/apirodd/wh-rdcc.git
cd wh-rdcc
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended (experiments in the paper use a single
NVIDIA T4). Automatic mixed precision is enabled by default.

---

## Data preparation

### DICHASUS (real datasets)

The DICHASUS measurements are publicly available from the University of
Stuttgart. Download the TFRecords and the reference-transmitter offset files
for the scenarios of interest (`cf0x`, `cf1x`, `015x`) and run:

```bash
python -m src.preprocessing_dichasus --scenario cf0x --out data/cf0x
```

This produces, in `data/cf0x/`:
```
train_csi.npy         (N, A, 1024, 2)  float32, per-sample normalised
train_positions.npy   (N, 2)           float32
train_timestamps.npy  (N,)             float64
test_*.npy            (same layout)
```

### DeepMIMO (synthetic scenario)

Fully reproducible from a fixed seed (no manual download of intermediate
files needed):

```bash
python -m src.preprocessing_deepmimo --scenario asu_campus_3p5 --out data/deepmimo
```

---

## Running experiments

### Single run

```bash
# WARDEN at λ=20 on cf0x, seed 0
python -m src.run_experiment --data_dir data/cf0x --lam 20 --seed 0

# Static baseline (λ=0)
python -m src.run_experiment --data_dir data/cf0x --lam 0 --seed 0
```

### Reproduce the paper tables

```bash
python -m scripts.phase2_statistics   # multi-dataset, multi-seed  -> results/phase2_results.json
python -m scripts.phase3_ablation     # d / L / Δ ablation         -> results/phase3_ablation.json
python -m scripts.phase4_baselines    # LSTM-CC, JEPA-CC           -> results/phase4_baselines.json
python -m scripts.phase5_overhead     # timing                     -> results/phase5_overhead.json
```

Each batch script saves incrementally and skips completed runs, so it can be
interrupted and resumed safely.

---

## Method summary

Training proceeds in two stages:

1. **Bootstrap** (`E0` epochs): train the encoder with the NT-Xent contrastive
   charting loss only, then estimate `W*` in closed form.
2. **Joint** (`E1` epochs): optimise the encoder and residual network with
   `L = L_CC + λ · L_WH`, re-estimating `W*` every `Δ` epochs.

The eigenspectrum of `W*` provides an interpretable, supervision-free diagnostic
of user-equipment kinematics: for quasi-uniform motion, its eigenvalues cluster
near the unit circle.

Default configuration (used across all datasets, no per-dataset tuning):
`d = 8`, `L = 16`, `Δ = 5`, Adam (lr `1e-3`, wd `1e-4`), cosine annealing,
batch size 32, 140 epochs (`E0 = 40`, `E1 = 100`).

---

## Citation

```bibtex
@article{piroddi2025warden,
  author  = {Piroddi, Andrea and Kanakis, Triantafyllos
             and Opoku Agyeman, Michael},
  title   = {{WARDEN}: {Wiener-Adaptive} Residual Dynamic {ENcoder}
             for Channel Charting in Future Cellular Networks},
  journal = {IEEE Access},
  year    = {2025}
}
```

---

## Acknowledgements

This work was carried out as part of a PhD by Published Work at the University
of Northampton. The authors thank the Institute of Telecommunications (INÜ) at
the University of Stuttgart for the publicly available DICHASUS dataset, and
the DeepMIMO project for the synthetic ray-tracing scenarios.

## License

MIT — see [LICENSE](LICENSE). The DICHASUS dataset is licensed under CC BY 4.0.
