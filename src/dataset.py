# -*- coding: utf-8 -*-
"""dataset.py — PyTorch Dataset per sequenze temporali di CSI"""

import numpy as np
import torch
from torch.utils.data import Dataset


class DICHASUSDataset(Dataset):
    """
    Dataset per WH-RDCC.
    Restituisce sequenze di CSI e posizioni con gap temporale ≤ max_gap_ms.
    """

    def __init__(
        self,
        csi_path: str,
        pos_path: str,
        time_path: str,
        seq_len: int = 16,
        stride: int = 4,
        max_gap_ms: float = 750.0,
    ):
        self.csi = np.load(csi_path, mmap_mode="r")
        self.positions = np.load(pos_path)
        self.timestamps = np.load(time_path)
        self.seq_len = seq_len

        # Costruisce lista di indici di sequenze valide
        self.valid = []
        for i in range(0, len(self.timestamps) - seq_len + 1, stride):
            gaps = np.diff(self.timestamps[i : i + seq_len]) * 1000
            if np.all(gaps <= max_gap_ms) and np.all(gaps > 0):
                self.valid.append(i)

        print(f"  Sequenze valide: {len(self.valid)}")

    def __len__(self) -> int:
        return len(self.valid)

    def __getitem__(self, idx: int):
        s = self.valid[idx]
        e = s + self.seq_len
        csi = torch.from_numpy(self.csi[s:e].copy())  # (L, 32, 1024, 2)
        pos = torch.from_numpy(self.positions[s:e])  # (L, 2)
        return csi, pos