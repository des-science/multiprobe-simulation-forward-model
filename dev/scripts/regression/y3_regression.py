"""Bit-level regression test of the DES Y3 (v18) forward model, for refactors that must not change Y3 output.

Three stages, each a production entry point on frozen inputs:
    obs       observation.forward_model_observation_map on the cached DES Y3 catalog maps (data/cache), all four
              variants deep_lss evaluates (shear rotation on/off x maglim systematics on/off)
    single    run_single_postprocessing on the fiducial perm_0000, all 4 patches (the mock observation path)
    grid      run_grid_postprocessing --debug on grid cosmology 0, its perm_0000-0003 symlinked to the fiducial
              permutations (the Y3 grid maps are gone from scratch; meaningless physically, fine for a regression)

The fiducial app is not covered: only cosmo_fiducial survives, and its --no_derivatives mode is broken at HEAD.

Every array of every output is hashed into <out>/manifest.json. Usage, CPU only, on a compute node:
    python y3_regression.py run --out DIR [--stages obs,single,grid,fiducial]
    python y3_regression.py compare DIR_A DIR_B
    python y3_regression.py baseline --out ROOT      # run ROOT/a, run ROOT/b, compare: is the baseline stable?
"""

import argparse
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import h5py  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
CONFIG = os.path.join(REPO, "configs/v18/default.yaml")
FIDUCIAL_DIR = "/capstor/store/cscs/swissai/a0158/athomsen/deep_lss/data/fiducial/cosmo_fiducial"
STAGES = ("obs", "single", "grid")
N_GRID_PERMS = 4  # run_grid_postprocessing --debug stops after perm 3


# outputs -> arrays ###################################################################################################


def _tfrecord_arrays(path):
    import tensorflow as tf
    from tensorflow.core.framework import tensor_pb2

    for i, raw in enumerate(tf.data.TFRecordDataset(path).as_numpy_iterator()):
        example = tf.train.Example.FromString(raw)
        for key in sorted(example.features.feature):
            feature = example.features.feature[key]
            kind = feature.WhichOneof("kind")
            if kind == "bytes_list":
                for j, value in enumerate(feature.bytes_list.value):
                    yield f"{i}/{key}/{j}", tf.make_ndarray(tensor_pb2.TensorProto.FromString(value))
            else:
                yield f"{i}/{key}", np.asarray(getattr(feature, kind).value)


def _h5_arrays(path):
    with h5py.File(path, "r") as f:
        names = []
        f.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
        for name in sorted(names):
            yield name, f[name][()]


def _npz_arrays(path):
    with np.load(path) as f:
        for name in sorted(f.files):
            yield name, f[name]


READERS = {".tfrecord": _tfrecord_arrays, ".h5": _h5_arrays, ".npz": _npz_arrays}


def iter_arrays(out_dir):
    """(key, array) for every array of every output under out_dir, keyed by path relative to out_dir."""
    for path in sorted(glob.glob(os.path.join(out_dir, "**", "*"), recursive=True)):
        reader = READERS.get(os.path.splitext(path)[1])
        if reader is None or "/inputs/" in path:
            continue
        rel = os.path.relpath(path, out_dir)
        for key, arr in reader(path):
            yield f"{rel}::{key}", np.asarray(arr)


def _digest(arr):
    arr = np.ascontiguousarray(arr)
    return {
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


# stages ##############################################################################################################


def _drain(gen):
    for _ in gen:
        pass


def stage_obs(out):
    from msfm.utils import catalog, files, observation

    for name in ("desy3_metacal_gamma_e1p_e2m", "desy3_metacal_gamma_no_psi_rot_e1p_e2m", "desy3_maglim_count"):
        cache = os.path.join(REPO, "data/cache", f"{name}.npy")
        assert os.path.exists(cache), f"{cache} missing; the Y3 catalogs are gone, so it cannot be rebuilt"

    conf = files.load_config(CONFIG)
    wl = {
        "": catalog.build_metacal_map_from_cat(conf)[0],
        "_no_psi_rot": catalog.build_metacal_map_from_cat(conf, apply_shear_rotation=False, debug=False)[0],
    }
    gc = catalog.build_maglim_map_from_cat(conf)

    arrays = {}
    for rot, wl_map in wl.items():
        for apply_sys in (True, False):
            # forward_model_observation_map masks its inputs in place
            dv, cls, _ = observation.forward_model_observation_map(
                wl_gamma_map=wl_map.copy(),
                gc_count_map=gc.copy(),
                conf=conf,
                apply_norm=True,
                with_padding=True,
                nest_in=False,
                apply_maglim_sys_map=apply_sys,
            )
            label = f"DESy3{'' if apply_sys else '_no_sys'}{rot}"
            arrays[f"{label}__dv"], arrays[f"{label}__cls"] = dv, cls
    os.makedirs(os.path.join(out, "obs"), exist_ok=True)
    np.savez(os.path.join(out, "obs", "obs.npz"), **arrays)


def stage_single(out):
    from msfm.apps import run_single_postprocessing as app

    argv = [
        f"--dir_in={FIDUCIAL_DIR}",
        f"--dir_out={os.path.join(out, 'single')}",
        f"--msfm_config={CONFIG}",
        "--with_lensing",
        "--with_clustering",
        "--max_sleep=0",
        "--verbosity=warning",
    ]
    _drain(app.main([0], argv))


def _input_tree(root):
    """Symlink tree in the CosmoGrid layout: grid cosmology 0 -> fiducial perms, fiducial -> the store copy."""
    from msfm.utils import cosmogrid, files

    conf = files.load_config(CONFIG)
    meta = os.path.join(REPO, conf["files"]["meta_info"])
    grid_dir = cosmogrid.get_cosmo_params_info(meta, "grid")["path_par"][0].decode()
    fid_dir = cosmogrid.get_cosmo_params_info(meta, "fiducial")["path_par"][0].decode()
    assert fid_dir.rstrip("/") == "cosmo_fiducial", fid_dir

    tree = os.path.join(root, "inputs")
    cosmo = os.path.join(tree, "grid", grid_dir)
    os.makedirs(cosmo, exist_ok=True)
    for i in range(N_GRID_PERMS):
        link = os.path.join(cosmo, f"perm_{i:04d}")
        if not os.path.islink(link):
            os.symlink(os.path.join(FIDUCIAL_DIR, f"perm_{i:04d}"), link)
    os.makedirs(os.path.join(tree, "fiducial"), exist_ok=True)
    link = os.path.join(tree, "fiducial", fid_dir.rstrip("/"))
    if not os.path.islink(link):
        os.symlink(FIDUCIAL_DIR, link)
    return tree


def stage_grid(out):
    from msfm.apps import run_grid_postprocessing as app

    argv = [
        f"--dir_in={_input_tree(os.path.dirname(out))}",
        f"--dir_out={os.path.join(out, 'grid')}",
        f"--config={CONFIG}",
        "--n_files=2500",
        "--debug",
        "--max_sleep=0",
        "--verbosity=warning",
    ]
    _drain(app.main([0], argv))


# commands ############################################################################################################


def _provenance():
    def git(*cmd):
        try:
            return subprocess.check_output(["git", "-C", REPO, *cmd], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    import healpy
    import tensorflow as tf

    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": git("status", "--porcelain", "--untracked-files=no"),
        "host": platform.node(),
        "n_cpus": len(os.sched_getaffinity(0)),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "versions": {"numpy": np.__version__, "healpy": healpy.__version__, "tensorflow": tf.__version__},
        "config": CONFIG,
    }


def run(out, stages):
    """Fresh output dir per run: --debug reloads any program_state*.pkl it finds in dir_out."""
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f"{out} is not empty; the apps would replay a previous run's state")
    os.makedirs(out, exist_ok=True)

    timings = {}
    for stage in stages:
        t0 = time.time()
        globals()[f"stage_{stage}"](out)
        timings[stage] = round(time.time() - t0, 1)
        print(f"[y3_regression] {stage} done in {timings[stage]} s", flush=True)

    manifest = {
        "provenance": _provenance(),
        "stages": list(stages),
        "timings_s": timings,
        "arrays": {key: _digest(arr) for key, arr in iter_arrays(out)},
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[y3_regression] {len(manifest['arrays'])} arrays hashed into {out}/manifest.json", flush=True)


def compare(dir_a, dir_b):
    """Exit 0 iff both manifests hold the same keys with identical hashes. Reports max|diff| where outputs exist."""
    man = [json.load(open(os.path.join(d, "manifest.json")))["arrays"] for d in (dir_a, dir_b)]
    only = [sorted(set(man[0]) - set(man[1])), sorted(set(man[1]) - set(man[0]))]
    differ = sorted(k for k in set(man[0]) & set(man[1]) if man[0][k] != man[1][k])

    print(f"[y3_regression] {len(man[0])} vs {len(man[1])} arrays, {len(differ)} differ")
    for side, keys in zip("AB", only):
        for k in keys[:20]:
            print(f"  only in {side}: {k}")

    if differ:
        b_arrays = dict(iter_arrays(dir_b)) if os.path.isdir(dir_b) else {}
        for key, a in iter_arrays(dir_a):
            if key not in differ:
                continue
            b = b_arrays.get(key)
            if b is None or a.shape != b.shape or a.dtype.kind not in "fiu":
                print(f"  DIFF {key}: {man[0][key]['shape']} vs {man[1][key]['shape']}")
                continue
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            scale = np.max(np.abs(a)) or 1.0
            print(f"  DIFF {key}: max|d| {d.max():.3e} (rel {d.max() / scale:.3e}), {np.count_nonzero(d)} elements")

    ok = not (differ or only[0] or only[1])
    print(f"[y3_regression] {'IDENTICAL' if ok else 'DIFFERENT'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in ("run", "baseline"):
        p = sub.add_parser(cmd)
        p.add_argument("--out", required=True)
        p.add_argument("--stages", default=",".join(STAGES))
    p = sub.add_parser("compare")
    p.add_argument("dir_a")
    p.add_argument("dir_b")
    args = parser.parse_args()

    if args.cmd == "compare":
        sys.exit(0 if compare(args.dir_a, args.dir_b) else 1)

    stages = [s for s in args.stages.split(",") if s]
    assert set(stages) <= set(STAGES), f"unknown stages {set(stages) - set(STAGES)}"
    if args.cmd == "run":
        run(args.out, stages)
    else:
        run(os.path.join(args.out, "a"), stages)
        run(os.path.join(args.out, "b"), stages)
        sys.exit(0 if compare(os.path.join(args.out, "a"), os.path.join(args.out, "b")) else 1)


if __name__ == "__main__":
    main()
