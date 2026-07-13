"""
preprocessing_deepmimo.py
=========================

Generate a synthetic Channel Charting dataset from a DeepMIMO ray-tracing
scenario. UE mobility is synthesised via random-waypoint trajectories over
the valid (non-blocked) user grid; the CSI of the nearest grid point is
sampled at a fixed rate. Fully reproducible from a fixed seed.

The base station uses an 8x4 planar array (32 antennas) with 1024 OFDM
subcarriers over 50 MHz, matching the antenna/subcarrier dimensions of the
DICHASUS datasets so the same encoder applies unchanged.

Example:
    python -m src.preprocessing_deepmimo \
        --scenario asu_campus_3p5 --out data/deepmimo --seed 42
"""

import argparse
import os

import numpy as np
from scipy.spatial import cKDTree


def main():
    import deepmimo as dm

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="asu_campus_3p5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_traj", type=int, default=10)
    ap.add_argument("--speed", type=float, default=1.5, help="m/s")
    ap.add_argument("--fs", type=float, default=4.0, help="Hz")
    ap.add_argument("--dur", type=float, default=150.0, help="s per trajectory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dm.download(args.scenario)
    dataset = dm.load(args.scenario)

    # valid (non-blocked) grid points
    valid_idx = np.where(dataset.los >= 0)[0]
    valid_pos = dataset.rx_pos[valid_idx][:, :2]
    tree = cKDTree(valid_pos)

    rng = np.random.default_rng(args.seed)
    step = args.speed / args.fs
    positions, grid_index, timestamps = [], [], []
    t = 0.0
    for _ in range(args.n_traj):
        cur = valid_pos[rng.integers(len(valid_pos))].astype(float)
        wp = valid_pos[rng.integers(len(valid_pos))].astype(float)
        for _ in range(int(args.dur * args.fs)):
            if np.linalg.norm(wp - cur) < step:
                wp = valid_pos[rng.integers(len(valid_pos))].astype(float)
            cur = cur + (wp - cur) / (np.linalg.norm(wp - cur) + 1e-9) * step
            _, nn = tree.query(cur)
            positions.append(valid_pos[nn])
            grid_index.append(valid_idx[nn])
            timestamps.append(t)
            t += 1.0 / args.fs
        t += 10.0                              # gap between trajectories

    positions = np.array(positions, dtype=np.float32)
    grid_index = np.array(grid_index)
    timestamps = np.array(timestamps)
    unique_idx = np.unique(grid_index)

    # compute channels only for visited grid points
    ds_sub = dataset.trim(idxs=unique_idx)
    ch_params = dm.ChannelParameters()
    ch_params.bs_antenna.shape = [8, 4]
    ch_params.ue_antenna.shape = [1, 1]
    ch_params.ofdm.subcarriers = 1024
    ch_params.ofdm.bandwidth = 50e6
    ch_params.ofdm.selected_subcarriers = np.arange(1024)
    ds_sub.compute_channels(ch_params)
    ch = np.squeeze(np.asarray(ds_sub.channel))          # (U, 32, 1024)

    idx_map = {g: i for i, g in enumerate(unique_idx)}
    rows = np.array([idx_map[g] for g in grid_index])
    csi_seq = ch[rows]                                   # (N, 32, 1024)

    csi_all = np.stack([csi_seq.real, csi_seq.imag], -1).astype(np.float32)
    mean = csi_all.mean(axis=(1, 2, 3), keepdims=True)
    std = csi_all.std(axis=(1, 2, 3), keepdims=True) + 1e-8
    csi_all = (csi_all - mean) / std

    k = int(len(csi_all) * 0.8)
    for prefix, sl in (("train", slice(None, k)), ("test", slice(k, None))):
        np.save(f"{args.out}/{prefix}_csi.npy", csi_all[sl])
        np.save(f"{args.out}/{prefix}_positions.npy", positions[sl])
        np.save(f"{args.out}/{prefix}_timestamps.npy", timestamps[sl])
        print(f"{prefix}: {csi_all[sl].shape}")

    print(f"Total {len(csi_all)} samples "
          f"({len(unique_idx)} unique grid points), saved to {args.out}")


if __name__ == "__main__":
    main()
