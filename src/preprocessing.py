# -*- coding: utf-8 -*-
"""preprocessing.py — DICHASUS TFRecord → NumPy chunks

Scarica e preprocessa i dataset DICHASUS cf02, cf03, cf04.
I file tfrecords (~20 GB totali) vengono letti, calibrati e salvati
come chunk di NumPy array su Google Drive.
"""

import os
import json
import subprocess
import numpy as np
import tensorflow as tf

# ============================================================================
# Configurazione percorsi
# ============================================================================
PROJECT_DIR = "/content/drive/MyDrive/WH_RDCC"
DATA_RAW = f"{PROJECT_DIR}/data_raw"
DATA_OUT = f"{PROJECT_DIR}/data"

os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_OUT, exist_ok=True)

# ============================================================================
# Spec DICHASUS (ricostruita dalla documentazione ufficiale)
# ============================================================================
SPEC = {
    "bandwidth": 50000000.0,
    "fc": 1272000000.0,
    "antennas": [
        {"location": [2.365, -5.5, 2.5], "assignments": [[24, 25, 26, 27], [28, 29, 30, 31]]},
        {"location": [-6.0, -5.5, 2.5], "assignments": [[16, 17, 18, 19], [20, 21, 22, 23]]},
        {"location": [-6.0, -13.0, 2.5], "assignments": [[8, 9, 10, 11], [12, 13, 14, 15]]},
        {"location": [2.365, -13.0, 2.5], "assignments": [[0, 1, 2, 3], [4, 5, 6, 7]]},
    ],
}

antenna_count = sum(sum(len(row) for row in ap["assignments"]) for ap in SPEC["antennas"])
antenna_assignments = [ap["assignments"] for ap in SPEC["antennas"]]
AP_pos = np.array([ap["location"] for ap in SPEC["antennas"]])

# ============================================================================
# Download con resume
# ============================================================================
BASE_API = "https://darus.uni-stuttgart.de/api/access/datafile"

FILES = {
    "dichasus-cf02.tfrecords": "/:persistentId?persistentId=doi:10.18419/DARUS-2854/14",
    "dichasus-cf03.tfrecords": "/:persistentId?persistentId=doi:10.18419/DARUS-2854/15",
    "dichasus-cf04.tfrecords": "/:persistentId?persistentId=doi:10.18419/DARUS-2854/16",
    "spec.json": "/:persistentId?persistentId=doi:10.18419/DARUS-2854/12",
    "reftx-offsets-dichasus-cf0x.json": "/:persistentId?persistentId=doi:10.18419/DARUS-2854/13",
}

EXPECTED_GB = {
    "dichasus-cf02.tfrecords": 4.5,
    "dichasus-cf03.tfrecords": 5.5,
    "dichasus-cf04.tfrecords": 10.3,
    "spec.json": 0.0,
    "reftx-offsets-dichasus-cf0x.json": 0.0,
}


def download_with_resume(url, dest, expected_gb):
    """Download con resume per file grandi."""
    existing = os.path.getsize(dest) if os.path.exists(dest) else 0
    expected_bytes = expected_gb * 1e9

    if existing >= expected_bytes * 0.95:
        print(f"  OK: {os.path.basename(dest)} ({existing/1e9:.2f} GB)")
        return

    headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
    mode = "ab" if existing > 0 else "wb"

    with requests.get(url, stream=True, timeout=120, headers=headers) as r:
        if r.status_code == 416:
            print(f"  File già completo")
            return
        r.raise_for_status()

        downloaded = existing
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=16 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                print(f"\r    {downloaded/1e9:.2f} GB", end="", flush=True)
    print(f"\n  Completato: {os.path.getsize(dest)/1e9:.2f} GB")


def download_all():
    """Scarica tutti i file DICHASUS."""
    import requests

    print("Download file DICHASUS...")
    for fname, path in FILES.items():
        dest = f"{DATA_RAW}/{fname}"
        url = f"{BASE_API}{path}"
        download_with_resume(url, dest, EXPECTED_GB[fname])

    # Salva spec.json
    with open(f"{DATA_RAW}/spec.json", "w") as f:
        json.dump(SPEC, f, indent=2)
    print("spec.json salvato.")


# ============================================================================
# Loader e calibratore
# ============================================================================
def load_calibrate(tfrecords_path, offsets_path):
    """Crea un dataset TFRecord calibrato e riordinato per antenne."""

    with open(offsets_path) as f:
        offsets = json.load(f)

    def parse(proto):
        record = tf.io.parse_single_example(
            proto,
            {
                "csi": tf.io.FixedLenFeature([], tf.string, default_value=""),
                "pos-tachy": tf.io.FixedLenFeature([], tf.string, default_value=""),
                "time": tf.io.FixedLenFeature([], tf.float32, default_value=0),
            },
        )
        csi = tf.ensure_shape(
            tf.io.parse_tensor(record["csi"], out_type=tf.float32), (antenna_count, 1024, 2)
        )
        csi = tf.complex(csi[:, :, 0], csi[:, :, 1])
        csi = tf.signal.fftshift(csi, axes=1)
        pos = tf.ensure_shape(tf.io.parse_tensor(record["pos-tachy"], out_type=tf.float64), (3,))
        return csi, pos, record["time"]

    def calibrate(csi, pos, time):
        sto = tf.tensordot(
            tf.constant(offsets["sto"]),
            2 * np.pi * tf.range(tf.shape(csi)[1], dtype=tf.float32) / tf.cast(tf.shape(csi)[1], tf.float32),
            axes=0,
        )
        cpo = tf.tensordot(tf.constant(offsets["cpo"]), tf.ones(tf.shape(csi)[1], dtype=tf.float32), axes=0)
        csi = tf.multiply(csi, tf.exp(tf.complex(0.0, sto + cpo)))
        return csi, pos, time

    def reorder(csi, pos, time):
        csi = tf.stack([[tf.gather(csi, idx) for idx in arr] for arr in antenna_assignments])
        return csi, pos, time

    ds = tf.data.TFRecordDataset(tfrecords_path)
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(calibrate, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(reorder, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


# ============================================================================
# Preprocessing → chunk NumPy
# ============================================================================
def process_tfrecord_to_chunks(tfrecords_path, offsets_path, chunk_size=500, every_n=5):
    """Processa un file TFRecord e salva chunk di NumPy."""
    import tensorflow as tf

    fname = os.path.basename(tfrecords_path).replace(".tfrecords", "")
    print(f"\n=== {fname} ===")

    ds = load_calibrate(tfrecords_path, offsets_path)
    ds = ds.enumerate().filter(lambda i, _: i % every_n == 0).map(lambda i, v: v)

    csi_chunk, pos_chunk, time_chunk = [], [], []
    chunk_idx = 0
    total = 0

    for csi, pos, time in ds:
        csi_flat = tf.reshape(csi, (antenna_count, 1024))
        csi_ri = tf.stack([tf.math.real(csi_flat), tf.math.imag(csi_flat)], axis=-1)

        csi_chunk.append(csi_ri.numpy().astype(np.float32))
        pos_chunk.append(pos.numpy()[:2].astype(np.float32))
        time_chunk.append(float(time.numpy()))

        if len(csi_chunk) >= chunk_size:
            out_prefix = f"{DATA_OUT}/{fname}_chunk{chunk_idx:04d}"
            np.save(f"{out_prefix}_csi.npy", np.stack(csi_chunk))
            np.save(f"{out_prefix}_pos.npy", np.stack(pos_chunk))
            np.save(f"{out_prefix}_time.npy", np.array(time_chunk))
            total += len(csi_chunk)
            print(f"  Chunk {chunk_idx:04d} salvato — totale: {total}", end="\r")
            csi_chunk, pos_chunk, time_chunk = [], [], []
            chunk_idx += 1

    if csi_chunk:
        out_prefix = f"{DATA_OUT}/{fname}_chunk{chunk_idx:04d}"
        np.save(f"{out_prefix}_csi.npy", np.stack(csi_chunk))
        np.save(f"{out_prefix}_pos.npy", np.stack(pos_chunk))
        np.save(f"{out_prefix}_time.npy", np.array(time_chunk))
        total += len(csi_chunk)

    print(f"\n  {fname}: {total} campioni in {chunk_idx+1} chunk")
    return total


def preprocess_all():
    """Processa tutti e tre i dataset DICHASUS."""
    PATHS = [
        {"tfrecords": f"{DATA_RAW}/dichasus-cf02.tfrecords", "offsets": f"{DATA_RAW}/reftx-offsets-dichasus-cf02.json"},
        {"tfrecords": f"{DATA_RAW}/dichasus-cf03.tfrecords", "offsets": f"{DATA_RAW}/reftx-offsets-dichasus-cf03.json"},
        {"tfrecords": f"{DATA_RAW}/dichasus-cf04.tfrecords", "offsets": f"{DATA_RAW}/reftx-offsets-dichasus-cf04.json"},
    ]

    for p in PATHS:
        if os.path.exists(p["tfrecords"]):
            process_tfrecord_to_chunks(p["tfrecords"], p["offsets"])
        else:
            print(f"File non trovato: {p['tfrecords']}")


# ============================================================================
# Assembly finale (concatena chunk e split train/test)
# ============================================================================
def assemble_train_test(train_ratio=0.8):
    """Concatena tutti i chunk, ordina per timestamp, split train/test."""
    chunk_bases = sorted(
        set(f.replace("_csi.npy", "") for f in os.listdir(DATA_OUT) if "_chunk" in f and "_csi.npy" in f)
    )
    print(f"Chunk totali: {len(chunk_bases)}")

    csi_list, pos_list, time_list = [], [], []
    for base in chunk_bases:
        csi_list.append(np.load(f"{DATA_OUT}/{base}_csi.npy"))
        pos_list.append(np.load(f"{DATA_OUT}/{base}_pos.npy"))
        time_list.append(np.load(f"{DATA_OUT}/{base}_time.npy"))

    print("Assemblo e ordino...")
    csi_all = np.concatenate(csi_list, axis=0)
    pos_all = np.concatenate(pos_list, axis=0)
    time_all = np.concatenate(time_list, axis=0)

    idx = np.argsort(time_all)
    csi_all = csi_all[idx]
    pos_all = pos_all[idx]
    time_all = time_all[idx]

    k = int(len(csi_all) * train_ratio)
    for prefix, sl in [("train", slice(None, k)), ("test", slice(k, None))]:
        np.save(f"{DATA_OUT}/{prefix}_csi.npy", csi_all[sl])
        np.save(f"{DATA_OUT}/{prefix}_positions.npy", pos_all[sl])
        np.save(f"{DATA_OUT}/{prefix}_timestamps.npy", time_all[sl])
        print(f"{prefix}: {csi_all[sl].shape}")

    # Salva posizioni AP
    np.save(f"{DATA_OUT}/AP_pos.npy", AP_pos.astype(np.float32))
    print("Assembly completato.")


if __name__ == "__main__":
    download_all()
    preprocess_all()
    assemble_train_test()