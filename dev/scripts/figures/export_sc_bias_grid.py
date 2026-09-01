# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created August 2026
Author: Arne Thomsen (with Claude)

Exports the grid of fitted source-clustering biases b_g,s -- one value per CosmoGrid grid
cosmology and metacal tomographic bin, for both the "clean" and "contam" arms of
notebooks/sc_bias_fit_count.ipynb -- together with each cosmology's S8, for the paper_2 figure
showing the effect of the DES Y3 imaging-systematics correction on the fit.

The bias table is the tracked deliverable of that notebook, data/desy3_metacal_bias.h5 -- one file
with a "clean" (no correction) and a "contam" (contaminated with the ISD imprint of
files/lss_systematics.tex before the fit) HDF5 group. See project_isd_systematics_bias_fit and
project_source_clustering_bias_fit in the workspace memory for why the correction exists.

Cheap: two ~4 MB h5 tables and one metainfo file, no maps. A login node is fine.

    ~/dlss/torch_env/bin/python3 dev/scripts/figures/export_sc_bias_grid.py
"""

import argparse
import os
import subprocess

import h5py
import numpy as np

from msfm.utils import cosmogrid, files, source_clustering_bias as scb

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(REPO_DIR, "configs/v18/default.yaml"))
    parser.add_argument(
        "--bias_file",
        default=os.path.join(REPO_DIR, "data/desy3_metacal_bias.h5"),
        help="both arms, one HDF5 group each",
    )
    parser.add_argument("--output", default=None, help="path of the .h5 to write; defaults to the plotting cache")
    return parser.parse_args()


def default_output():
    return os.path.abspath(
        os.path.join(REPO_DIR, "../deep_lss_paper/paper_2_plotting/cache", "sc_bias_systematics_grid.h5")
    )


def git_hash(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def read_bias_table(path, group):
    """One arm (HDF5 group) of notebooks/sc_bias_fit_count.ipynb's combined bias table.

    The biases are one dataset per cosmology, not a stacked array (unlike `loss`/`stat`), so `keys`
    gives the row order to read them in.
    """
    with h5py.File(path, "r") as f:
        g = f[group]
        keys = [k.decode() if isinstance(k, bytes) else str(k) for k in g["keys"][:]]
        biases = np.array([g[k][:] for k in keys])
        label = str(g.attrs["systematics_label"])
    return keys, biases, label


def main():
    args = parse_args()
    output = args.output or default_output()

    conf = files.load_config(args.config)
    z_bins = conf["survey"]["metacal"]["z_bins"]

    print(f"reading {args.bias_file}", flush=True)
    keys_clean, bias_clean, label_clean = read_bias_table(args.bias_file, "clean")
    keys_contam, bias_contam, label_contam = read_bias_table(args.bias_file, "contam")
    assert keys_clean == keys_contam, "the two arms must be fit against the same cosmology grid"
    keys = keys_clean
    print(
        f"{len(keys)} cosmologies, systematics {label_clean!r} vs {label_contam!r}, " f"biases {bias_clean.shape}",
        flush=True,
    )

    # S8 per grid cosmology, keyed the same way the fit tables are -- as in
    # sc_bias_fit_count.ipynb, which itself follows notebooks/metacal_bias.ipynb
    meta_info_file = os.path.join(REPO_DIR, conf["files"]["meta_info"])
    params_grid = cosmogrid.get_cosmo_params_info(meta_info_file, simset="grid")
    key_to_row = {scb.cosmo_key(p): i for i, p in enumerate(params_grid["path_par"])}

    i_grid = np.array([k != "fiducial" for k in keys])
    rows = [key_to_row[k] for k in np.array(keys)[i_grid]]
    Om = params_grid["Om"][rows].astype(float)
    s8 = params_grid["s8"][rows].astype(float)
    S8 = s8 * np.sqrt(Om / 0.3)
    print(
        f"{i_grid.sum()} grid cosmologies (fiducial excluded), "
        f"sigma_8 in [{s8.min():.3f}, {s8.max():.3f}], S8 in [{S8.min():.3f}, {S8.max():.3f}]",
        flush=True,
    )

    # ------------------------------------------------------------------------------------ write
    ds = {"compression": "gzip", "compression_opts": 4}
    os.makedirs(os.path.dirname(output), exist_ok=True)
    print(f"writing {output}", flush=True)

    with h5py.File(output, "w") as f:
        f.attrs["description"] = (
            "the fitted source-clustering bias b_g,s of every CosmoGrid grid cosmology and metacal "
            "tomographic bin, clean vs contaminated with the DES Y3 imaging-systematics correction, "
            "against each cosmology's sigma_8 and S8. Produced by "
            "msfm/dev/scripts/figures/export_sc_bias_grid.py from the tracked bias table it names below"
        )
        f.attrs["bias_file"] = os.path.abspath(args.bias_file)
        f.attrs["systematics_label_clean"] = label_clean
        f.attrs["systematics_label_contam"] = label_contam
        f.attrs["msfm_git_sha"] = git_hash(REPO_DIR)
        f.attrs["n_bins"] = len(z_bins)
        f.attrs["bin_labels"] = z_bins

        f.create_dataset("S8", data=S8.astype(np.float64), **ds)
        f.create_dataset("sigma8", data=s8.astype(np.float64), **ds)
        b = f.create_group("bias")
        b.attrs["description"] = "b_g,s of the grid cosmologies (fiducial excluded), row-matched to S8"
        b.create_dataset("clean", data=bias_clean[i_grid].astype(np.float64), **ds)
        b.create_dataset("contam", data=bias_contam[i_grid].astype(np.float64), **ds)

    print("done", flush=True)


if __name__ == "__main__":
    main()
