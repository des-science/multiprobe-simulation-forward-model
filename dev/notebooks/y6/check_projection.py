"""Check DES Y6 projected_probes_maps files: nside, stored bins, and shearm + n(z) against the realization set.

Needs only numpy and h5py. Pass files, or a grid dir to sample --n random cosmologies (two perms each):

    python check_projection.py <grid_dir> --nz-set <.../desy6_source_redshift.h5>
"""

import argparse
import glob
import hashlib
import os
import random

import h5py
import numpy as np

FNAME = "projected_probes_maps_v11dmb.h5"
SRC = ["metacal1", "metacal2", "metacal3", "metacal4"]


def sample_files(grid_dir, n, rng):
    cosmo_dirs = sorted(glob.glob(os.path.join(grid_dir, "cosmo_*")))
    files = []
    for cosmo_dir in rng.sample(cosmo_dirs, min(n, len(cosmo_dirs))):
        perm_dirs = sorted(glob.glob(os.path.join(cosmo_dir, "perm_*")))
        files += [os.path.join(p, FNAME) for p in rng.sample(perm_dirs, min(2, len(perm_dirs)))]
    return files


def read(path):
    with h5py.File(path, "r") as f:
        npix = {f"{g}/{b}": f["map"][g][b].shape[0] for g in f["map"] for b in f["map"][g]}
        nz = {b: f["nz"][b][:] for b in f["nz"]}
        m = {b: float(f["nz"][b].attrs["shearm"]) for b in f["nz"]}
    return npix, nz, m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="h5 files and/or grid dirs")
    parser.add_argument("--n", type=int, default=5, help="cosmologies sampled per grid dir")
    parser.add_argument("--nz-set", help="desy6_source_redshift.h5 (M_BIN*, NZ_BIN*)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    files = []
    for p in args.paths:
        files += sample_files(p, args.n, rng) if os.path.isdir(p) else [p]

    nz_set = h5py.File(args.nz_set, "r") if args.nz_set else None
    m_set = np.stack([nz_set[f"M_BIN{i}"][:] for i in range(1, 5)], axis=1) if nz_set else None

    m_seen, nz_seen, rows_seen = set(), set(), []
    for path in files:
        npix, nz, m = read(path)
        nside = sorted({int(round(np.sqrt(n / 12))) for n in npix.values()})
        bins = sorted(nz)
        m_src = np.array([m[b] for b in SRC])
        nz_md5 = hashlib.md5(b"".join(nz[b].tobytes() for b in bins)).hexdigest()[:10]
        m_seen.add(tuple(m_src))
        nz_seen.add(nz_md5)

        print(path)
        print(f"  nside {nside}, {len(npix)} maps, nz bins {bins}")
        print(f"  shearm {SRC}: {np.array2string(m_src, precision=5, sign='+')}")
        print(f"  shearm other bins: { {b: v for b, v in m.items() if b not in SRC} }")
        print(f"  n(z) md5 {nz_md5}")
        if nz_set:
            rows = np.flatnonzero(np.all(np.isclose(m_set, m_src, rtol=0, atol=1e-12), axis=1))
            nz_ok = [
                all(
                    np.allclose(nz[b][:, 1], nz_set[f"NZ_BIN{i + 1}"][r], rtol=1e-6, atol=0) for i, b in enumerate(SRC)
                )
                for r in rows
            ]
            rows_seen += list(rows)
            print(f"  realization rows matching shearm: {rows.tolist()}, n(z) equals that row: {nz_ok}")

    print(f"\n{len(files)} files: {len(m_seen)} distinct shearm sets, {len(nz_seen)} distinct n(z)")
    if nz_set:
        print(f"matched rows: {sorted(set(int(r) for r in rows_seen))}")


if __name__ == "__main__":
    main()
