#!/usr/bin/env python
"""Phase 1 of the B-mode Cls information-loss Fisher study: reduce the fiducial tfrecords to the
per-perturbation *mean* binned Cls (the finite-difference Jacobian source).

Why this exists
---------------
`fiducial_cls.h5` already holds the fiducial-cosmology Cls of every realization (the sample
COVARIANCE, `cls/binned` + `cls/bmode_binned`). It does NOT hold the +/- perturbation Cls -- those
live per-example in the tfrecords as `cl_{label}` / `cl_bmode_{label}` features (raw, unbinned). The
Jacobian d(Cl)/d(theta) is the finite difference of the perturbation MEANS over all realizations, so
here we stream every fiducial tfrecord, average each perturbation's raw Cls over all
(example, noise) realizations, bin it with the exact same binning `merge()` used, and dump the
result to a small intermediate h5 that phase 2 (analyze.py) turns into a Fisher forecast.

Consistency with the covariance
--------------------------------
- The h5 `cls`/`cls_bmode` covariance columns are the raw get_cls lexicographic order (NO
  get_cross_bin_indices gather -- see tfrecords.parse_inverse_fiducial_cls). We therefore parse the
  perturbation `cl_{label}` the same way (plain reshape, no gather) so the Jacobian columns line up
  with the covariance columns one-to-one.
- Binning: E-block via power_spectra.bin_according_to_config (fixed 30-1500 grid, [None] smoothing),
  B-block via smooth_and_bin_cls(with_cross=False, [None]*42, fixed_binning) -- identical to the two
  binning calls in run_fiducial_postprocessing.merge().
- Binning is linear in Cl, so bin(mean_over_realizations(raw)) == mean(bin(raw)); we bin the mean.

Usage:  python reduce.py --config <cfg> --dir_in <fiducial tfrecord dir> --out <inputs.h5>
"""
import argparse
import glob
import os

import numpy as np
import h5py
import tensorflow as tf

from msfm.utils import files, parameters, power_spectra, logger

LOGGER = logger.get_logger(__file__)


def build_parse_fn(labels, e_only=False):
    """Parse only the Cls features (E, and B unless e_only) of every perturbation label, summed over
    the noise axis.

    Skipping the map features (kg_/dg_/sn_/pn_, ~7 MB each) keeps the host transfer tiny; the 2.5 GB
    files are still read from disk in full (TFRecord is sequential) but nothing heavy is decoded.
    Returns, per example: {label: (n_cls, 36)} for E and {label: (n_cls, 42)} for B, each already
    reduce_sum'd over the n_noise realizations, plus the scalar noise count. With e_only=True the
    tfrecord has no `cl_bmode_*`/`n_z_cross_bmode` (b_mode_cls was off) so only E is parsed -- used
    for the ell_min=30 baseline tree, whose Jacobian is needed for the l_min ablation.
    """
    feats = {
        "n_noise": tf.io.FixedLenFeature([], tf.int64),
        "n_cls": tf.io.FixedLenFeature([], tf.int64),
        "n_z_cross": tf.io.FixedLenFeature([], tf.int64),
    }
    if not e_only:
        feats["n_z_cross_bmode"] = tf.io.FixedLenFeature([], tf.int64)
    for lab in labels:
        feats[f"cl_{lab}"] = tf.io.FixedLenFeature([], tf.string)
        if not e_only:
            feats[f"cl_bmode_{lab}"] = tf.io.FixedLenFeature([], tf.string)

    def parse_fn(serialized):
        d = tf.io.parse_single_example(serialized, feats)
        n_noise = d["n_noise"]
        n_cls = d["n_cls"]
        n_zc = d["n_z_cross"]
        n_zc_b = None if e_only else d["n_z_cross_bmode"]
        out = {}
        for lab in labels:
            e = tf.io.parse_tensor(d[f"cl_{lab}"], out_type=tf.float32)
            e = tf.reshape(e, (n_noise, n_cls, n_zc))
            out[f"E::{lab}"] = tf.reduce_sum(e, axis=0)  # (n_cls, 36)
            if not e_only:
                b = tf.io.parse_tensor(d[f"cl_bmode_{lab}"], out_type=tf.float32)
                b = tf.reshape(b, (n_noise, n_cls, n_zc_b))
                out[f"B::{lab}"] = tf.reduce_sum(b, axis=0)  # (n_cls, 42)
        out["n_noise"] = n_noise
        return out

    return parse_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dir_in", required=True, help="fiducial tfrecord directory")
    ap.add_argument("--out", required=True, help="output intermediate h5")
    ap.add_argument("--cycle_length", type=int, default=32, help="parallel file reads")
    ap.add_argument("--e_only", action="store_true",
                    help="tfrecords have no cl_bmode field (b_mode_cls off) -- parse only E. Used "
                         "for the ell_min=30 baseline tree in the l_min ablation.")
    ap.add_argument("-v", "--verbosity", default="info")
    args = ap.parse_args()
    logger.set_all_loggers_level(args.verbosity)

    conf = files.load_config(args.config)
    params = list(parameters.get_parameters(conf=conf))
    labels = parameters.get_fiducial_perturbation_labels(params)  # [fiducial, P0_m, P0_p, ...]
    offsets = parameters.get_fiducial_perturbations(conf=conf)
    fiducials = parameters.get_fiducials(conf=conf)
    LOGGER.info(f"{len(params)} params, {len(labels)} perturbation labels")

    tfr = sorted(glob.glob(os.path.join(args.dir_in, "*.tfrecord")))
    assert tfr, f"no tfrecords in {args.dir_in}"
    LOGGER.info(f"streaming {len(tfr)} tfrecords from {args.dir_in}")

    ds = tf.data.Dataset.from_tensor_slices(tfr)
    ds = ds.interleave(
        tf.data.TFRecordDataset,
        cycle_length=args.cycle_length,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    ds = ds.map(build_parse_fn(labels, e_only=args.e_only), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # accumulate sum over all (example, noise) of the raw Cls, per label
    sum_E = {lab: None for lab in labels}
    sum_B = {lab: None for lab in labels}
    total_count = 0  # total realizations = sum of n_noise over examples
    n_examples = 0

    for ex in ds:
        for lab in labels:
            e = ex[f"E::{lab}"].numpy()
            sum_E[lab] = e if sum_E[lab] is None else sum_E[lab] + e
            if not args.e_only:
                b = ex[f"B::{lab}"].numpy()
                sum_B[lab] = b if sum_B[lab] is None else sum_B[lab] + b
        total_count += int(ex["n_noise"].numpy())
        n_examples += 1
        if n_examples % 200 == 0:
            LOGGER.info(f"  processed {n_examples} examples ({total_count} realizations)")

    LOGGER.info(f"done streaming: {n_examples} examples, {total_count} realizations")
    assert n_examples > 0

    # mean raw Cls per label, then bin exactly as merge() does
    n_z_stores = 0
    if conf["analysis"]["modelling"]["lensing"]["store"]:
        n_z_stores += len(conf["survey"]["metacal"]["z_bins"])
    if conf["analysis"]["modelling"]["clustering"]["store"]:
        n_z_stores += len(conf["survey"]["maglim"]["z_bins"])

    mean_raw_E = np.stack([sum_E[lab] / total_count for lab in labels], axis=0)  # (n_lab, n_cls, 36)

    # E-block: identical call to merge() -> bin_according_to_config (fixed 30-1500, [None] smoothing)
    binned_E, bin_edges_E = power_spectra.bin_according_to_config(mean_raw_E, conf)

    if not args.e_only:
        mean_raw_B = np.stack([sum_B[lab] / total_count for lab in labels], axis=0)  # (n_lab,n_cls,42)
        # B-block: identical call to merge() -> smooth_and_bin_cls(with_cross=False, [None]*n_b)
        n_b = mean_raw_B.shape[-1]
        binned_B, bin_edges_B = power_spectra.smooth_and_bin_cls(
            mean_raw_B,
            l_mins_smoothing=[None] * n_b,
            l_maxs_smoothing=[None] * n_b,
            n_bins=conf["analysis"]["power_spectra"]["n_bins"],
            with_cross=False,
            fixed_binning=True,
            l_min_binning=conf["analysis"]["power_spectra"]["l_min"],
            l_max_binning=conf["analysis"]["power_spectra"]["l_max"],
        )
        LOGGER.info(f"binned E {binned_E.shape}, binned B {binned_B.shape}")
    else:
        LOGGER.info(f"binned E {binned_E.shape} (e_only, no B)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("pert_binned_E", data=binned_E)   # (n_lab, n_bins, 36)
        f.create_dataset("bin_edges_E", data=bin_edges_E)
        if not args.e_only:
            f.create_dataset("pert_binned_B", data=binned_B)   # (n_lab, n_bins, 42)
            f.create_dataset("bin_edges_B", data=bin_edges_B)
        f.create_dataset("offsets", data=np.asarray(offsets, dtype=np.float64))
        f.create_dataset("fiducials", data=np.asarray(fiducials, dtype=np.float64))
        f.create_dataset("params", data=np.array(params, dtype="S"))
        f.create_dataset("labels", data=np.array(labels, dtype="S"))
        f.attrs["n_examples"] = n_examples
        f.attrs["n_realizations"] = total_count
        f.attrs["config"] = os.path.abspath(args.config)
    LOGGER.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
