"""
preprocessing_dichasus.py
=========================

Convert DICHASUS TFRecords into the .npy layout consumed by
DICHASUSDataset. Handles the three scenarios used in the paper, each with
its own antenna count and calibration-offset file:

    cf0x : 32 antennas (4 x 2x4 distributed arrays), meander trajectories
    cf12 : 24 active antennas (cf1x series), circular trajectories
    015x : 32 antennas (co-located array), serpentine trajectories

Download the TFRecords, spec.json, and the reftx-offsets JSON from the
DICHASUS website / DaRUS repository before running. Calibration applies the
per-antenna sampling-time (STO) and carrier-phase (CPO) offsets; note that
these constant offsets are absorbed by the encoder during training and are
applied here only for consistency with the classical CC pipeline.

Example:
    python -m src.preprocessing_dichasus \
        --tfrecords data_raw/cf0x/dichasus-cf02.tfrecords \
                    data_raw/cf0x/dichasus-cf03.tfrecords \
                    data_raw/cf0x/dichasus-cf04.tfrecords \
        --offsets   data_raw/cf0x/reftx-offsets-dichasus-cf02.json \
                    data_raw/cf0x/reftx-offsets-dichasus-cf03.json \
                    data_raw/cf0x/reftx-offsets-dichasus-cf04.json \
        --antennas 32 --out data/cf0x
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf


def build_dataset(tfrecord_path, offsets, antennas):
    def parse(proto):
        rec = tf.io.parse_single_example(proto, {
            "csi": tf.io.FixedLenFeature([], tf.string, default_value=""),
            "pos-tachy": tf.io.FixedLenFeature([], tf.string,
                                               default_value=""),
            "time": tf.io.FixedLenFeature([], tf.float32, default_value=0),
        })
        csi = tf.ensure_shape(
            tf.io.parse_tensor(rec["csi"], out_type=tf.float32),
            (antennas, 1024, 2))
        csi = tf.complex(csi[:, :, 0], csi[:, :, 1])
        csi = tf.signal.fftshift(csi, axes=1)
        pos = tf.ensure_shape(
            tf.io.parse_tensor(rec["pos-tachy"], out_type=tf.float64), (3,))
        return csi, pos, rec["time"]

    def calibrate(csi, pos, time):
        k = tf.range(tf.shape(csi)[1], dtype=tf.float32)
        sto = tf.tensordot(
            tf.constant(offsets["sto"]),
            2 * np.pi * k / tf.cast(tf.shape(csi)[1], tf.float32), axes=0)
        cpo = tf.tensordot(
            tf.constant(offsets["cpo"]),
            tf.ones(tf.shape(csi)[1], dtype=tf.float32), axes=0)
        csi = tf.multiply(csi, tf.exp(tf.complex(0.0, sto + cpo)))
        return csi, pos, time

    ds = tf.data.TFRecordDataset(tfrecord_path)
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(calibrate, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrecords", nargs="+", required=True)
    ap.add_argument("--offsets", nargs="+", required=True,
                    help="one offsets JSON per TFRecord (same order)")
    ap.add_argument("--antennas", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every_n", type=int, default=5,
                    help="temporal subsampling factor")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    csi_all, pos_all, time_all = [], [], []
    for tfr, off_path in zip(args.tfrecords, args.offsets):
        with open(off_path) as f:
            offsets = json.load(f)
        ds = build_dataset(tfr, offsets, args.antennas)
        ds = (ds.enumerate()
                .filter(lambda i, _: i % args.every_n == 0)
                .map(lambda i, v: v))
        for csi, pos, time in ds:
            c = csi.numpy()
            csi_all.append(np.stack([c.real, c.imag], -1).astype(np.float32))
            pos_all.append(pos.numpy()[:2].astype(np.float32))
            time_all.append(float(time.numpy()))

    csi_all = np.stack(csi_all)
    pos_all = np.stack(pos_all)
    time_all = np.array(time_all)

    order = np.argsort(time_all)
    csi_all, pos_all, time_all = csi_all[order], pos_all[order], time_all[order]

    # per-sample normalisation
    mean = csi_all.mean(axis=(1, 2, 3), keepdims=True)
    std = csi_all.std(axis=(1, 2, 3), keepdims=True) + 1e-8
    csi_all = (csi_all - mean) / std

    k = int(len(csi_all) * 0.8)
    for prefix, sl in (("train", slice(None, k)), ("test", slice(k, None))):
        np.save(f"{args.out}/{prefix}_csi.npy", csi_all[sl])
        np.save(f"{args.out}/{prefix}_positions.npy", pos_all[sl])
        np.save(f"{args.out}/{prefix}_timestamps.npy", time_all[sl])
        print(f"{prefix}: {csi_all[sl].shape}")

    dt = np.diff(time_all)
    print(f"Total {len(csi_all)} samples, "
          f"~{1 / dt.mean():.2f} Hz, saved to {args.out}")


if __name__ == "__main__":
    main()
