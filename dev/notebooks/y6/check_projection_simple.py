"""Inspect one DES Y6 projected_probes_maps file. Needs numpy and h5py."""

import h5py
import numpy as np

path = "/Users/arne/data/temp/projected_probes_maps_v11dmb.h5"

# values in every file of the buggy version (row 0 of desy6_source_redshift.h5)
old_m = [0.000527, 0.012040, 0.014410, 0.006542]
old_zmean = [0.3945, 0.5200, 0.8295, 1.1371]

with h5py.File(path, "r") as f:
    print("maps:")
    for probe in f["map"]:
        for b in f["map"][probe]:
            npix = f["map"][probe][b].shape[0]
            print(f"  map/{probe}/{b}: nside {int(np.sqrt(npix / 12))}")

    print("\nnz bins:", list(f["nz"]))

    print("\nsource bins:   shearm (old)        mean z (old)")
    for i in range(4):
        nz = f[f"nz/metacal{i + 1}"]
        z, n = nz[:, 0], nz[:, 1]
        m, zmean = nz.attrs["shearm"], (z * n).sum() / n.sum()
        print(f"  metacal{i + 1}:   {m:+.6f} ({old_m[i]:+.6f})   {zmean:.4f} ({old_zmean[i]:.4f})")
