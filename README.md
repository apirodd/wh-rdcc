# WH-RDCC: Wiener-Hopf Residual Dynamic Channel Charting

[![IEEE WCL](https://img.shields.io/badge/IEEE-WCL%202025-blue)](https://ieeexplore.ieee.org)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Official implementation of the paper:

> **Wiener-Hopf Residual Dynamic Channel Charting for 6G Semantic Localization**  
> Andrea Piroddi, Triantafyllos Kanakis, Michael Opoku Ogyeman  
> IEEE Wireless Communications Letters, 2025  
> [[Paper]](#) · [[arXiv]](#) · [[DICHASUS Dataset]](https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-cf0x/)

---

## Overview

WH-RDCC is a self-supervised Channel Charting architecture that integrates an
analytically optimal **Wiener-Hopf transition matrix** as a structured prior for
latent-space dynamics.

The key idea: instead of learning latent dynamics entirely from data (as in JEPA or
LSTM-AE approaches), we decompose the prediction problem into:

1. **Optimal linear component** — computed in closed form via the Wiener-Hopf solution
2. **Non-linear residual** — learned by a lightweight MLP

This decomposition improves both the geometric fidelity of the channel chart and
the physical interpretability of the latent space.

### Key Results (DICHASUS cf0x, d=8)

| Method | MAE [m] ↓ | TW ↑ | CT ↑ | Pred Error ↓ |
|--------|-----------|------|------|-------------|
| CC Baseline | 3.094 | 0.844 | 0.917 | 0.241 |
| WH-RDCC λ=5 | **2.916** | 0.886 | **0.922** | 0.204 |
| WH-RDCC λ=20 | 3.120 | **0.900** | 0.905 | **0.151** |

- TW improves **monotonically** with λ (+6.6% at λ=20)
- MAE reduced by **5.8%** at λ=5
- Linear prediction error reduced by **37%** at λ=20
- Spectral radius ρ(W*) ∈ [0.976, 0.994] — physically consistent with quasi-conservative UE dynamics

---

## Architecture

```
CSI_t  →  ┐
           ├→  Encoder f_θ (CNN)  →  z_t  →  ┬→  L_CC (NT-Xent)
CSI_t+1 → ┘                        z_t+1 →  ┤
                                              └→  L_WH = ‖z_t+1 - W*z_t - g_φ(z_t)‖²
                                    W* ← Wiener-Hopf({z_t})  (every Δ epochs)
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/wh-rdcc.git
cd wh-rdcc
pip install -r requirements.txt
```

### Requirements

```
torch>=2.2.0
torchvision>=0.17.0
numpy>=1.26.0
scipy>=1.12.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
seaborn>=0.13.0
pyyaml>=6.0.1
tqdm>=4.66.0
tensorflow>=2.15.0   # for DICHASUS TFRecord parsing only
```

---

## Data Preparation

Download the DICHASUS cf0x dataset from the
[University of Stuttgart DaRUS repository](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-2854):

```bash
mkdir -p data_raw
cd data_raw

# Download via DaRUS API
BASE=https://darus.uni-stuttgart.de/api/access/datafile
wget -O dichasus-cf02.tfrecords "$BASE/:persistentId?persistentId=doi:10.18419/DARUS-2854/14"
wget -O dichasus-cf03.tfrecords "$BASE/:persistentId?persistentId=doi:10.18419/DARUS-2854/15"
wget -O dichasus-cf04.tfrecords "$BASE/:persistentId?persistentId=doi:10.18419/DARUS-2854/16"
```

Also download the offset calibration files and spec.json (see `scripts/download_dichasus.sh`).

Then run preprocessing:

```bash
python src/preprocessing.py
```

This produces:
```
data/
  train_csi.npy        (N_train, 32, 1024, 2)  float32
  train_positions.npy  (N_train, 2)             float32
  train_timestamps.npy (N_train,)               float64
  test_csi.npy         (N_test,  32, 1024, 2)  float32
  test_positions.npy   (N_test,  2)             float32
  test_timestamps.npy  (N_test,)                float64
```

---

## Training

```bash
# Train WH-RDCC with default config (λ=5)
python src/train.py --config configs/default.yaml

# Train with specific λ
python src/train.py --lambda_wh 20.0

# Train baseline (no WH regularisation)
python src/train.py --lambda_wh 0.0 --run_name baseline
```

### Configuration

Key hyperparameters in `configs/default.yaml`:

```yaml
latent_dim: 8          # d — ablation: 2, 4, 8, 16
lambda_wh: 5.0         # regularisation weight
bootstrap_epochs: 40   # Phase 1 epochs
joint_epochs: 100      # Phase 2 epochs
wh_update_every: 5     # W* re-estimation frequency
batch_size: 32
learning_rate: 5e-4
seq_len: 16            # temporal sequence length
max_gap_ms: 750        # maximum inter-sample gap
```

---

## Evaluation

```bash
python src/evaluate.py --checkpoint runs/wh_rdcc_lambda5/model.pt
```

Outputs:
- MAE (after affine alignment)
- Trustworthiness and Continuity (k=10)
- Kruskal Stress
- Prediction error PE = E[‖z_{t+1} - W*z_t‖]
- Spectral radius ρ(W*)
- Effective rank of W*
- Residual norm ‖g_φ(z_t)‖

---

## Reproducing Paper Results

```bash
# Full ablation study (λ ∈ {0, 0.1, 0.5, 1, 5, 10, 20})
python scripts/run_ablation.py

# Generate all paper figures
python scripts/generate_figures.py

# Results are saved to results/ablation_results.json
```

---

## Project Structure

```
wh-rdcc/
├── src/
│   ├── preprocessing.py   # DICHASUS TFRecord → NumPy
│   ├── dataset.py         # PyTorch Dataset (temporal sequences)
│   ├── encoder.py         # CNN encoder f_θ
│   ├── wiener_hopf.py     # W* estimation (closed-form)
│   ├── residual_net.py    # MLP residual g_φ
│   ├── losses.py          # NT-Xent + WH loss
│   ├── train.py           # Two-phase training loop
│   └── evaluate.py        # Metrics: TW, CT, MAE, PE, ρ(W*)
├── scripts/
│   ├── download_dichasus.sh
│   ├── run_ablation.py
│   └── generate_figures.py
├── configs/
│   └── default.yaml
├── notebooks/
│   └── 01_colab_full_pipeline.ipynb  # End-to-end on Google Colab
├── figures/               # Paper figures (PDF + PNG)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{piroddi2025whrdcc,
  author  = {Piroddi, Andrea and Kanakis, Triantafyllos, Opoku Ogyeman Michael},
  title   = {Wiener-Hopf Residual Dynamic Channel Charting
             for {6G} Semantic Localization},
  journal = {IEEE Wireless Communications Letters},
  year    = {2025},
  note    = {to appear}
}
```

---

## Acknowledgements

This work was carried out as part of a PhD by Published Work at the
University of Northampton (CAST Centre), supervised by Dr. Aldo Kanakis.
The authors thank the Institute of Telecommunications (INÜ) at the
University of Stuttgart for making the DICHASUS dataset publicly available.

Experiments were conducted on Google Colab with NVIDIA Tesla T4 GPU.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The DICHASUS dataset is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
