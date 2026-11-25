"""Validation of a grid .tfrecord set written by msfm/apps/run_grid_postprocessing.py, for any survey config.

An esub-style app: main(indices, argv) checks one file per index (one file = one grid cosmology), merge(indices, argv)
checks the whole set. Both run through pipelines/common/run_packed.py, so a cluster's packed executor runs them like the
grid app itself.

main, stage tfrecords (read with tfrecords.parse_inverse_grid, the reader downstream uses):
    records     n_perms * n_patches records, i_signal 0, 1, ... in order, one i_sobol equal to the metainfo's
    schema      the feature keys; n_pix (the pixel file's data vector), n_z_metacal, n_z_maglim, n_noise, n_params
                (parameters.get_parameters) and n_z_cross
    labels      the cosmology equals the metainfo row; the astrophysical parameters equal the app's Latin hypercube draw
                (seeded with the cosmology index) and lie inside the priors
    maps        finite, the padding exactly 0, the footprint filled, no two noise realizations identical
    noise       per record and bin, the variance over the noise realizations, averaged over the footprint. For the lens
                counts it is also divided by the Poisson expectation, n_gal * A_pix times the fraction of white-noise
                variance in l_min <= l <= l_max ((l_max + 1)^2 - l_min^2 of 12 n_side^2 modes)
    signal      per record and bin, the footprint variance of the mean over the realizations minus noise / n_noise, and
                the footprint std of every realization
    grid_cls    (--grid_cls) the labels, i_sobol and raw Cls equal the cosmology's row of the merged Cl file, found by
                i_sobol since the merge orders the files randomly
main, stage projection (--dir_in, the cosmology's n_perms projected files):
    files       present, no dataset ending past the end of the file, one map nside and layout
    m           lensing.check_projection, and shearm through lensing.read_m_bias (within 5 sigma of the prior) if the
                config takes m from the projection
    n(z)        the mean z of every nz/<bin>
merge:
    completeness, the per-file failures, i_sobol and the astrophysical labels distinct across files, one Cl length,
                every row of the merged Cl file matched by one file
    noise       the lensing noise variance per bin with its robust outliers, and the lens Poisson ratio per bin
    labels/maps per file, each lens bin's signal variance correlates best with its own bg^2 label, so a label/channel
                permutation fails here; the lensing signal variance rises with S8 across files
    m and n(z)  m differs between the perms of every cosmology and no two files share an m vector; m against the
                config's prior; the mean z across perms and against the config's n(z)
    reproduce   (--reproduce_dir, the output of a --debug rerun of the grid app) every rerun record is byte-identical to
                the production record at the same position

main writes <dir_out>/scan/index_XXXX.npz and records failures instead of raising. merge writes scan.h5, report.txt
(PASS, WARN, FAIL or INFO per check), noise.png and signal.png, and exits 1 on any FAIL.

Usage, CPU, tf_env (pipelines/daint/desy6/submit.sh validate submits both):
    python pipelines/common/run_packed.py --app dev/scripts/validation/grid_tfrecords.py --function main --tasks 0 \\
        --config <yaml> --dir_tfrecords <dir> --dir_out <dir> [--dir_in <projection root>] [--grid_cls <h5>]
    python pipelines/common/run_packed.py --app dev/scripts/validation/grid_tfrecords.py --function merge --tasks 0 \\
        <the same arguments> [--reproduce_dir <dir>] [--partial]
"""

import argparse
import glob
import os
import re
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# the tasks read the merged Cl file concurrently
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
TAG = "[grid_tfrecords]"
STAGES = ("projection", "tfrecords")
SIZE_KEYS = ("n_pix", "n_params", "n_z_metacal", "n_z_maglim", "n_noise", "n_z_cross")
FAILURE = ("kind", "count", "detail")

# merge tolerances
NOISE_OUTLIER_Z = 5.0
POISSON_TOLERANCE = 0.02
BG_MIN_CORRELATION = 0.9
S8_MIN_SPEARMAN = 0.8
REPRODUCE_GLOB = "*.tfrecord"


def setup(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--dir_tfrecords", required=True, help="the grid .tfrecords")
    p.add_argument("--dir_out", required=True, help="scan/, scan.h5, report.txt and the figures")
    p.add_argument("--dir_in", help="projection root holding grid/, enables stage projection")
    p.add_argument("--grid_cls", help="the merged grid_cls.h5, compared with the .tfrecords")
    p.add_argument("--reproduce_dir", help="merge: the .tfrecords of a --debug rerun of the grid app")
    p.add_argument("--stages", help="comma-separated subset of projection,tfrecords, default all that apply")
    p.add_argument("--file_suffix", default="", help="as for the grid app")
    p.add_argument("--cosmogrid_version", default="1.1", help="as for the grid app")
    p.add_argument("--partial", action="store_true", help="merge: missing files are a WARN, for test runs")
    args, _ = p.parse_known_args(argv)

    stages = args.stages.split(",") if args.stages else ["projection"] * bool(args.dir_in) + ["tfrecords"]
    assert set(stages) <= set(STAGES), f"unknown stages {set(stages) - set(STAGES)}"
    assert "projection" not in stages or args.dir_in, "stage projection needs --dir_in"
    args.stages = stages
    return args


class Context:
    """What the checks compare against: the config, the metainfo and the pixel file."""

    def __init__(self, args):
        from msfm.utils import cosmogrid, filenames, files, parameters

        conf = files.load_config(args.config)
        modelling = conf["analysis"]["modelling"]
        assert modelling["lensing"]["store"] and modelling["clustering"]["store"], "expects lensing and clustering"
        assert not modelling["store_cross_maps"], "expects no cross maps"
        self.conf = conf

        grid = conf["analysis"]["grid"]
        self.n_cosmos, self.n_perms, self.n_noise = (
            grid["n_cosmos"],
            grid["n_perms_per_cosmo"],
            grid["n_noise_per_signal"],
        )
        self.n_signal = self.n_perms * conf["analysis"]["n_patches"]
        self.bins_wl, self.bins_gc = conf["survey"]["metacal"]["z_bins"], conf["survey"]["maglim"]["z_bins"]
        self.n_wl, self.n_gc = len(self.bins_wl), len(self.bins_gc)
        self.channels = self.bins_wl + self.bins_gc
        n = len(self.channels)
        self.n_pairs = n * (n + 1) // 2

        self.info = cosmogrid.get_cosmo_params_info(os.path.join(REPO, conf["files"]["meta_info"]), "grid")
        self.params = list(parameters.get_parameters(conf=conf))
        self.cosmo_params = parameters.get_sobol_parameters(conf)
        self.astro_params = parameters.get_latin_hypercube_parameters(conf)
        self.astro_priors = parameters.get_prior_intervals(self.astro_params, conf=conf)
        self.bg_index = [self.params.index(p) for p in conf["analysis"]["params"]["bg"]["linear"]]

        masks = files.get_tomo_dv_masks(conf)
        self.masks = [m.astype(bool) for m in masks["metacal"].T] + [m.astype(bool) for m in masks["maglim"].T]
        self.n_pix = masks["metacal"].shape[0]
        # Poisson variance of the lens counts per pixel, times the fraction of white noise that the hard scale cuts
        # keep: the multipoles l_min <= l <= l_max of the 12 n_side^2 a white-noise map carries
        n_side = conf["analysis"]["n_side"]
        n_pix_sky = 12 * n_side**2
        cuts = conf["analysis"]["scale_cuts"]["clustering"]
        l_max = np.array(3 * n_side - 1 if cuts["l_max"] is None else cuts["l_max"])
        band = ((l_max + 1) ** 2 - np.array(cuts["l_min"]) ** 2) / n_pix_sky
        pix_area = 4 * np.pi * (180 / np.pi) ** 2 / n_pix_sky
        self.poisson_var_gc = np.array(conf["survey"]["maglim"]["n_gal"]) * pix_area * band

        tag = conf["survey"]["name"] + args.file_suffix
        baryonified = modelling["baryonified"]
        self.tfrecord = lambda d, i: filenames.get_filename_tfrecords(d, i, tag, "grid", with_bary=baryonified)
        self.projected = filenames.get_filename_full_maps(with_bary=baryonified, version=args.cosmogrid_version)

        noise = range(self.n_noise)
        self.keys = {"cosmo", "n_params", "i_sobol", "i_signal", "cls", "n_noise", "n_cls", "n_z_cross", "n_pix"}
        self.keys |= {"n_z_metacal", "n_z_maglim"} | {f"kg_{i}" for i in noise} | {f"dg_{i}" for i in noise}


class Failures:
    """Failure kinds of one file, each with its count and first detail."""

    def __init__(self):
        self.kinds = {}

    def __call__(self, kind, detail):
        count, first = self.kinds.get(kind, (0, detail))
        self.kinds[kind] = (count + 1, first)

    def arrays(self):
        kinds = sorted(self.kinds)
        return {
            "failure_kind": np.array(kinds, dtype=str),
            "failure_count": np.array([self.kinds[k][0] for k in kinds], dtype=int),
            "failure_detail": np.array([self.kinds[k][1] for k in kinds], dtype=str),
        }


def scan_file(dir_out, index):
    return os.path.join(dir_out, "scan", f"index_{index:04d}.npz")


# main ################################################################################################################


def _check_record(ctx, raw, r, out, fail):
    """Checks one serialized record and fills row r of out. Returns its Cls, or None if it cannot be parsed."""
    import tensorflow as tf

    from msfm.utils import tfrecords

    feature = tf.train.Example.FromString(raw.numpy()).features.feature
    keys = set(feature)
    if keys != ctx.keys:
        fail("feature keys", f"record {r}: missing {sorted(ctx.keys - keys)}, extra {sorted(keys - ctx.keys)}")
        return None
    sizes = {key: feature[key].int64_list.value[0] for key in SIZE_KEYS + ("n_cls",)}
    expected = dict(zip(SIZE_KEYS, (ctx.n_pix, len(ctx.params), ctx.n_wl, ctx.n_gc, ctx.n_noise, ctx.n_pairs)))
    wrong = {key: (sizes[key], value) for key, value in expected.items() if sizes[key] != value}
    for key, (got, value) in wrong.items():
        fail(f"schema {key}", f"record {r}: {got}, expected {value}")
    if wrong:
        return None
    out["n_cls"][r] = sizes["n_cls"]

    inv = tfrecords.parse_inverse_grid(raw, range(ctx.n_noise))
    out["cosmo"][r] = inv["cosmo"].numpy()
    out["i_sobol"][r] = int(inv["i_sobol"])
    out["i_signal"][r] = int(inv["i_signal"])
    # (n_noise, n_pix, n_channels)
    maps = np.concatenate(
        [np.stack([inv[f"{probe}_{i}"].numpy() for i in range(ctx.n_noise)]) for probe in ("kg", "dg")], axis=-1
    )
    cls = np.stack([inv[f"cl_{i}"].numpy() for i in range(ctx.n_noise)])

    if not (np.all(np.isfinite(maps)) and np.all(np.isfinite(cls)) and np.all(np.isfinite(out["cosmo"][r]))):
        fail("not finite", f"record {r}")
    for c, mask in enumerate(ctx.masks):
        x = maps[:, :, c]
        if np.any(x[:, ~mask]):
            fail("padding not 0", f"record {r}, {ctx.channels[c]}")
        footprint = x[:, mask]
        if np.count_nonzero(footprint) < 0.5 * footprint.size:
            fail("footprint empty", f"record {r}, {ctx.channels[c]}")
        noise = footprint.var(axis=0, ddof=1).mean()
        out["noise_var"][r, c] = noise
        out["signal_var"][r, c] = footprint.mean(axis=0).var() - noise / ctx.n_noise
        out["std"][r, :, c] = footprint.std(axis=1)
    for i in range(ctx.n_noise):
        for j in range(i + 1, ctx.n_noise):
            if np.array_equal(maps[i], maps[j]):
                fail("identical noise realizations", f"record {r}: {i} and {j}")
    return cls


def _check_labels(ctx, index, out, fail):
    from scipy.stats import qmc

    from msfm.utils import parameters

    n_cosmo = len(ctx.cosmo_params)
    cosmo = np.array([ctx.info[p][index] for p in ctx.cosmo_params], dtype=np.float32)
    if not np.all(out["cosmo"][:, :n_cosmo] == cosmo):
        fail("cosmology labels", f"max |label - metainfo| {np.abs(out['cosmo'][:, :n_cosmo] - cosmo).max():.3g}")

    # the grid app's draw, one cosmology per file
    lo, hi = ctx.astro_priors[:, 0], ctx.astro_priors[:, 1]
    sampler = qmc.LatinHypercube(d=len(ctx.astro_params), seed=index)
    expected = parameters.unit_to_prior(sampler.random(ctx.n_signal), ctx.astro_params, ctx.conf).astype(np.float32)
    astro = out["cosmo"][:, n_cosmo:]
    if not np.array_equal(astro, expected):
        fail("astro labels", f"not the Latin hypercube draw, max |diff| {np.abs(astro - expected).max():.3g}")
    if not np.all((astro >= lo) & (astro <= hi)):
        fail("astro labels", "outside the prior")


def _check_grid_cls(ctx, args, out, cls, fail):
    """The merged file's row of this cosmology holds its records, the noise realizations concatenated. The merge lists
    the files in shuffled order (Dataset.list_files), so the row is found by i_sobol. Returns the row."""
    cls = np.stack(cls)
    with h5py.File(args.grid_cls, "r") as f:
        rows = np.flatnonzero(f["i_sobol"][:, 0] == out["i_sobol"][0])
        if len(rows) != 1:
            fail("grid_cls row", f"{len(rows)} rows with i_sobol {out['i_sobol'][0]}")
            return -1
        k = int(rows[0])
        raw = f["cls/raw"][k]
        row = {key: f[key][k] for key in ("cosmo", "i_sobol", "i_signal")}
    if not np.array_equal(raw, np.concatenate([cls[:, i] for i in range(ctx.n_noise)])):
        fail("grid_cls Cls", f"row {k} differs from the .tfrecord")
    for key in ("i_sobol", "i_signal"):
        if not np.array_equal(row[key], np.tile(out[key], ctx.n_noise)):
            fail(f"grid_cls {key}", f"row {k} differs from the .tfrecord")
    if not np.array_equal(row["cosmo"], np.tile(out["cosmo"], (ctx.n_noise, 1))):
        fail("grid_cls labels", f"row {k} differs from the .tfrecord")
    return k


def scan_tfrecords(ctx, args, index, fail):
    import tensorflow as tf

    n, n_ch = ctx.n_signal, len(ctx.channels)
    out = {
        "cosmo": np.full((n, len(ctx.params)), np.nan, dtype=np.float32),
        "i_sobol": np.full(n, -1),
        "i_signal": np.full(n, -1),
        "n_cls": np.full(n, -1),
        "noise_var": np.full((n, n_ch), np.nan),
        "signal_var": np.full((n, n_ch), np.nan),
        "std": np.full((n, ctx.n_noise, n_ch), np.nan),
        "grid_cls_row": np.array(-1),
    }
    path = ctx.tfrecord(args.dir_tfrecords, index)
    if not os.path.isfile(path):
        fail("tfrecord missing", path)
        return out

    cls, n_records = [], 0
    try:
        for raw in tf.data.TFRecordDataset(path):
            if n_records < n:
                cls.append(_check_record(ctx, raw, n_records, out, fail))
            n_records += 1
    except tf.errors.OpError as e:
        fail("unreadable", f"after record {n_records}: {type(e).__name__}: {e.message[:300]}")

    if n_records != n:
        fail("record count", f"{n_records}, expected {n}")
    if len(np.unique(out["n_cls"])) != 1:
        fail("Cl length", f"varies between records: {np.unique(out['n_cls'])}")
    if not np.array_equal(out["i_signal"], np.arange(n)):
        fail("i_signal", f"not 0..{n - 1} in order: {out['i_signal'][:8]}...")
        return out
    sobol, expected = np.unique(out["i_sobol"]), ctx.info["sobol_index"][index]
    if not (len(sobol) == 1 and sobol[0] == expected):
        fail("i_sobol", f"{sobol[:5]}, metainfo {expected}")
    _check_labels(ctx, index, out, fail)
    if args.grid_cls and all(c is not None for c in cls):
        out["grid_cls_row"] = np.array(_check_grid_cls(ctx, args, out, cls, fail))
    return out


def scan_projection(ctx, args, index, fail):
    from msfm.utils import lensing

    bins = ctx.bins_wl + ctx.bins_gc
    out = {
        "projection_m": np.full((ctx.n_perms, ctx.n_wl), np.nan),
        "projection_mean_z": np.full((ctx.n_perms, len(bins)), np.nan),
        "projection_nside": np.full(ctx.n_perms, -1),
    }
    layouts = [""] * ctx.n_perms
    m_from_projection = lensing.m_bias_from_projection(ctx.conf)
    path_par = ctx.info["path_par"][index].decode().strip("/")

    for p in range(ctx.n_perms):
        path = os.path.join(args.dir_in, "grid", path_par, f"perm_{p:04d}", ctx.projected)
        if not os.path.isfile(path):
            fail("projected file missing", path)
            continue
        try:
            size = os.path.getsize(path)
            n_pix, ends = set(), []
            with h5py.File(path, "r") as f:
                for group in f["map"]:
                    for name in f["map"][group]:
                        dataset = f["map"][group][name]
                        n_pix.add(dataset.shape[0])
                        offset = dataset.id.get_offset()
                        if offset is not None:
                            ends.append(offset + dataset.id.get_storage_size())
                layouts[p] = ";".join(f"{g}:{'+'.join(sorted(f['map'][g]))}" for g in sorted(f["map"]))
                for k, z_bin in enumerate(bins):
                    nz = f["nz"][z_bin][:]
                    out["projection_mean_z"][p, k] = _mean_z(nz[:, 0], nz[:, 1])
            if ends and max(ends) > size:
                fail("projected file truncated", path)
            if len(n_pix) != 1:
                fail("projected maps of several sizes", f"{path}: {sorted(n_pix)}")
            out["projection_nside"][p] = int(round(np.sqrt(min(n_pix) / 12)))
            lensing.check_projection(ctx.conf, path)
            if m_from_projection:
                out["projection_m"][p] = lensing.read_m_bias(ctx.conf, path)
        except AssertionError as e:
            fail("projection does not match the config", str(e)[:300])
        except (OSError, KeyError) as e:
            fail("projected file unreadable", f"{path}: {type(e).__name__}: {e}")

    out["projection_layout"] = np.array(layouts, dtype=str)
    return out


def _mean_z(z, nz):
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return trapezoid(z * nz, z) / trapezoid(nz, z)


def main(indices, argv):
    args = setup(argv)
    ctx = Context(args)
    os.makedirs(os.path.join(args.dir_out, "scan"), exist_ok=True)

    for index in indices:
        fail = Failures()
        out = {"index": np.array(index)}
        if "projection" in args.stages:
            out.update(scan_projection(ctx, args, index, fail))
        if "tfrecords" in args.stages:
            out.update(scan_tfrecords(ctx, args, index, fail))
        out.update(fail.arrays())

        # written under a name the done-marker glob does not match, then renamed
        path = scan_file(args.dir_out, index)
        tmp = os.path.join(os.path.dirname(path), "tmp_" + os.path.basename(path))
        np.savez(tmp, **out)
        os.replace(tmp, path)

        print(f"{TAG} index {index}: {len(fail.kinds)} failure kinds, stages {','.join(args.stages)}", flush=True)
        for kind, (count, detail) in sorted(fail.kinds.items()):
            print(f"{TAG}   {kind} x{count}: {detail}", flush=True)
        yield index


# merge ###############################################################################################################


class Report:
    def __init__(self):
        self.lines, self.n_fail = [], 0

    def __call__(self, status, check, detail):
        assert status in ("PASS", "WARN", "FAIL", "INFO"), status
        self.n_fail += status == "FAIL"
        self.lines.append(f"{status:4s}  {check}: {detail}")
        print(f"{TAG} {self.lines[-1]}", flush=True)


def _per_bin(values, labels, fmt="{:.4g}"):
    return ", ".join(f"{label} {fmt.format(v)}" for label, v in zip(labels, values))


def load_scans(ctx, dir_out):
    """The per-file arrays stacked along a leading file axis, over the files that were scanned."""
    scans = [
        dict(np.load(scan_file(dir_out, i))) for i in range(ctx.n_cosmos) if os.path.isfile(scan_file(dir_out, i))
    ]
    assert scans, f"no scan in {dir_out}/scan"
    failures = [(int(s["index"]), k, c, d) for s in scans for k, c, d in zip(*(s[f"failure_{x}"] for x in FAILURE))]
    stacked = {key: np.stack([s[key] for s in scans]) for key in scans[0] if not key.startswith("failure_")}
    return stacked, failures


def write_scan(path, scans, failures):
    with h5py.File(path, "w") as f:
        for key, arr in scans.items():
            f.create_dataset(key, data=arr.astype("S") if arr.dtype.kind == "U" else arr)
        f.create_dataset("failure_index", data=np.array([x[0] for x in failures], dtype=int))
        for i, name in enumerate(FAILURE, start=1):
            column = np.array([x[i] for x in failures])
            f.create_dataset(f"failure_{name}", data=column.astype("S") if column.dtype.kind == "U" else column)


def check_files(ctx, args, scans, failures, report):
    present = scans["index"]
    missing = sorted(set(range(ctx.n_cosmos)) - set(present.tolist()))
    status = "PASS" if not missing else "WARN" if args.partial else "FAIL"
    report(
        status,
        "completeness",
        f"{len(present)} of {ctx.n_cosmos} files" + (f", missing {missing[:10]}" * bool(missing)),
    )

    if not failures:
        report("PASS", "per-file checks", f"no failure in {len(present)} files, stages {','.join(args.stages)}")
    by_kind = {}
    for index, kind, count, detail in failures:
        by_kind.setdefault(kind, []).append((index, count, detail))
    for kind, rows in sorted(by_kind.items()):
        index, count, detail = rows[0]
        report("FAIL", f"per-file {kind}", f"{len(rows)} files, e.g. index {index} (x{count}): {detail}")


def check_tfrecords(ctx, scans, report):
    from scipy.stats import spearmanr

    n_cosmo, n_wl = len(ctx.cosmo_params), ctx.n_wl
    ok = np.all(scans["i_signal"] == np.arange(ctx.n_signal), axis=1)

    sobol = scans["i_sobol"][ok, 0]
    report("PASS" if len(np.unique(sobol)) == len(sobol) else "FAIL", "i_sobol", f"{len(np.unique(sobol))} distinct")
    astro = scans["cosmo"][ok, 0, n_cosmo:]
    n_unique = len(np.unique(astro, axis=0))
    report("PASS" if n_unique == len(astro) else "FAIL", "astro labels", f"{n_unique} distinct across files")
    n_cls = np.unique(scans["n_cls"][scans["n_cls"] >= 0])
    report("PASS" if len(n_cls) == 1 else "FAIL", "Cl length", f"{n_cls}")
    rows = scans["grid_cls_row"][scans["grid_cls_row"] >= 0]
    if len(rows):
        report(
            "PASS" if len(np.unique(rows)) == len(rows) else "FAIL",
            "grid_cls rows",
            f"{len(rows)} files matched to {len(np.unique(rows))} distinct rows of the merged Cl file",
        )

    # lensing noise: per file, the mean over its records
    noise = np.nanmean(scans["noise_var"][:, :, :n_wl], axis=1)
    median = np.nanmedian(noise, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(noise - median), axis=0)
    z = np.abs(noise - median) / mad
    outliers = np.unique(np.nonzero(z > NOISE_OUTLIER_Z)[0])
    spread = [f"{lo:.3f}-{hi:.3f}" for lo, hi in (np.nanpercentile(noise, [1, 99], axis=0) / median).T]
    report(
        "WARN" if len(outliers) else "PASS",
        "lensing noise variance",
        f"median {_per_bin(median, ctx.bins_wl)}; 1-99% / median {_per_bin(spread, ctx.bins_wl, '{}')}; "
        f"{len(outliers)} files beyond {NOISE_OUTLIER_Z:g} robust sigma"
        + (f", e.g. index {scans['index'][outliers[:10]].tolist()}" * bool(len(outliers))),
    )

    poisson = scans["noise_var"][:, :, n_wl:] / ctx.poisson_var_gc
    p_median = np.nanmedian(poisson, axis=(0, 1))
    p_lo, p_hi = np.nanmin(poisson, axis=(0, 1)), np.nanmax(poisson, axis=(0, 1))
    report(
        "WARN" if np.any(np.abs(p_median - 1) > POISSON_TOLERANCE) else "PASS",
        "lens Poisson ratio",
        f"median {_per_bin(p_median, ctx.bins_gc)}; range {p_lo.min():.4f}-{p_hi.max():.4f}",
    )

    # each lens bin's signal variance against every bg^2 label, per file over its records
    signal = scans["signal_var"][ok][:, :, n_wl:]
    bg2 = scans["cosmo"][ok][:, :, ctx.bg_index] ** 2
    corr = _corr(signal, bg2)
    diag = np.diagonal(corr, axis1=1, axis2=2)
    off = corr + np.where(np.eye(ctx.n_gc, dtype=bool), -np.inf, 0)
    bad = np.any(np.argmax(corr, axis=2) != np.arange(ctx.n_gc), axis=1) | np.any(diag < BG_MIN_CORRELATION, axis=1)
    report(
        "FAIL" if np.any(bad) else "PASS",
        "lens signal vs bg labels",
        f"own bg^2: min correlation {_per_bin(diag.min(0), ctx.bins_gc, '{:.3f}')}; "
        f"largest with another bin {off.max():.3f}; {bad.sum()} files where a bin does not match its own label"
        + (f", e.g. index {scans['index'][ok][bad][:10].tolist()}" * bool(np.any(bad))),
    )

    s8 = scans["cosmo"][ok, 0, ctx.params.index("s8")] * np.sqrt(scans["cosmo"][ok, 0, ctx.params.index("Om")] / 0.3)
    wl_signal = scans["signal_var"][ok][:, :, :n_wl].mean(axis=1)
    rho = np.array([spearmanr(s8, wl_signal[:, j])[0] for j in range(n_wl)])
    report(
        "WARN" if np.any(rho < S8_MIN_SPEARMAN) else "PASS",
        "lensing signal vs S8",
        f"Spearman across files {_per_bin(rho, ctx.bins_wl, '{:.3f}')}",
    )
    return s8, wl_signal, signal, bg2, noise, poisson


def _corr(x, y):
    """Per file, the correlation over records of every column of x (files, records, a) with every column of y."""
    x = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
    y = (y - y.mean(1, keepdims=True)) / y.std(1, keepdims=True)
    return np.einsum("fna,fnb->fab", x, y) / x.shape[1]


def check_projection(ctx, scans, report):
    from msfm.utils import files, lensing

    nside, layout = np.unique(scans["projection_nside"]), np.unique(scans["projection_layout"])
    report("PASS" if len(nside) == 1 else "FAIL", "projection nside", f"{nside}")
    report("PASS" if len(layout) == 1 else "FAIL", "projection layout", f"{len(layout)} distinct: {layout[:2]}")

    if lensing.m_bias_from_projection(ctx.conf):
        m = scans["projection_m"]
        constant = [i for i, mi in zip(scans["index"], m) if len(np.unique(mi, axis=0)) != ctx.n_perms]
        report(
            "FAIL" if constant else "PASS",
            "m across perms",
            f"{len(constant)} cosmologies where two perms share m" + (f", e.g. {constant[:10]}" * bool(constant)),
        )
        flat = m.reshape(-1, ctx.n_wl)
        n_unique = len(np.unique(flat, axis=0))
        report("PASS" if n_unique == len(flat) else "FAIL", "m across files", f"{n_unique} distinct of {len(flat)}")
        prior = ctx.conf["survey"]["metacal"]["shear_bias"]["multiplicative"]
        for name, value, expected in (("mean", flat.mean(0), prior["mu"]), ("std", flat.std(0), prior["sigma"])):
            report(
                "INFO",
                f"m {name} against the prior",
                f"{_per_bin(value, ctx.bins_wl, '{:+.5f}')} (prior {_per_bin(expected, ctx.bins_wl, '{:+.5f}')})",
            )

    bins = ctx.bins_wl + ctx.bins_gc
    mean_z = scans["projection_mean_z"]
    varies = np.mean([[len(np.unique(mz[:, k])) > 1 for k in range(len(bins))] for mz in mean_z], axis=0)
    fiducial = []
    for sample in ("metacal", "maglim"):
        fiducial += [_mean_z(z, nz) for z, nz in zip(*files.load_redshift_distributions(sample, ctx.conf))]
    report(
        "INFO", "n(z) across perms", f"fraction of cosmologies whose mean z varies: {_per_bin(varies, bins, '{:.2f}')}"
    )
    report(
        "INFO",
        "n(z) against the config",
        f"<mean z> - config mean z: {_per_bin(mean_z.reshape(-1, len(bins)).mean(0) - fiducial, bins, '{:+.4f}')}",
    )


def check_reproduction(ctx, args, report):
    import tensorflow as tf

    paths = sorted(glob.glob(os.path.join(args.reproduce_dir, REPRODUCE_GLOB)))
    if not paths:
        report("FAIL", "reproduce", f"no .tfrecord in {args.reproduce_dir}")
        return
    n_records, differ = 0, []
    for path in paths:
        index = int(re.search(r"_(\d+)\.tfrecord$", path).group(1))
        rerun = [raw.numpy() for raw in tf.data.TFRecordDataset(path)]
        production = tf.data.TFRecordDataset(ctx.tfrecord(args.dir_tfrecords, index))
        differ += [f"index {index} record {r}" for r, (a, b) in enumerate(zip(rerun, production)) if a != b.numpy()]
        n_records += len(rerun)
    report(
        "FAIL" if differ or not n_records else "PASS",
        "reproduce",
        f"{len(paths)} rerun files, {n_records} records, {len(differ)} differ from the production files"
        + (f", e.g. {differ[:5]}" * bool(differ)),
    )


def plot(ctx, dir_out, s8, wl_signal, gc_signal, bg2, noise, poisson):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_col = max(ctx.n_wl, ctx.n_gc)
    fig, axes = plt.subplots(2, n_col, figsize=(3.2 * n_col, 6), squeeze=False)
    for j, label in enumerate(ctx.bins_wl):
        axes[0, j].hist(noise[:, j], bins=50, color="C0")
        axes[0, j].set(title=label, xlabel="lensing noise variance per file")
    for j, label in enumerate(ctx.bins_gc):
        axes[1, j].hist(poisson[:, :, j].ravel(), bins=50, color="C1")
        axes[1, j].axvline(1, color="k", lw=0.8)
        axes[1, j].set(title=label, xlabel="noise variance / Poisson expectation")
    for ax in axes.flat:
        if not ax.has_data():
            ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(dir_out, "noise.png"), dpi=110)

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, n_col, figsize=(3.2 * n_col, 6), squeeze=False)
    for j, label in enumerate(ctx.bins_wl):
        axes[0, j].scatter(s8, wl_signal[:, j], s=2, color="C0")
        axes[0, j].set(title=label, xlabel="S8", ylabel="lensing signal variance")
    for j, label in enumerate(ctx.bins_gc):
        x, y = bg2[:, :, j].ravel(), gc_signal[:, :, j].ravel()
        pick = rng.choice(len(x), size=min(len(x), 20000), replace=False)
        axes[1, j].scatter(x[pick], y[pick], s=1, color="C1")
        axes[1, j].set(title=label, xlabel="bg^2 label", ylabel="lens signal variance")
    for ax in axes.flat:
        if not ax.has_data():
            ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(dir_out, "signal.png"), dpi=110)


def merge(indices, argv):
    args = setup(argv)
    ctx = Context(args)
    report = Report()

    scans, failures = load_scans(ctx, args.dir_out)
    write_scan(os.path.join(args.dir_out, "scan.h5"), scans, failures)
    check_files(ctx, args, scans, failures, report)
    if "projection_m" in scans:
        check_projection(ctx, scans, report)
    if "noise_var" in scans:
        plot(ctx, args.dir_out, *check_tfrecords(ctx, scans, report))
    if args.reproduce_dir:
        check_reproduction(ctx, args, report)

    with open(os.path.join(args.dir_out, "report.txt"), "w") as f:
        f.write(f"{args.config}\n{args.dir_tfrecords}\n\n" + "\n".join(report.lines) + "\n")
    print(f"{TAG} {report.n_fail} FAIL, wrote {args.dir_out}/report.txt", flush=True)
    if report.n_fail:
        sys.exit(1)
