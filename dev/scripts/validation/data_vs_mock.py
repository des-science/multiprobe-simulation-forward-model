"""Data vs mock: msfm's observation path on a survey's catalog maps, next to the forward model.

Five checks, the last two optional, for any survey config. Run it on DES Y3 (v18) too, as the reference for the
numbers:

1. observation: observation.forward_model_observation_map on the cached catalog maps, in the four variants that
   deep_lss.utils.evaluation.evaluate_obs_des evaluates (psi rotation on/off x lens weights on/off) and with its
   arguments. Asserts the data vector shape, finiteness and zero padding, the Cl shape of the tfrecords, and that the
   rotation changes only the lensing channels and the weights only the clustering channels.
2. noise model: at the observed source counts, observation.make_shape_noise_map with in_place (the real galaxies at
   their positions, randomly rotated: the data's own shape noise) and with count (the forward model's draws from the
   bin's ellipticity distribution), through the same observation path. count / in_place of the per-pixel kappa
   variance and of the Cl at high ell does not depend on cosmology and should be close to 1.
3. data vs mock: the observation next to the maps of a grid .tfrecord (signal plus noise, all records and noise
   realizations), per bin: the footprint std and the Cl in three ell bands. If the mock's cosmology differs from the
   data's, only the noise-dominated high-ell band is expected near 1.
4. grid noise (--scan, the scan.h5 of grid_tfrecords.py in this folder): the lensing noise variance of every
   grid record over the in_place noise variance of check 2, per bin. It tests the noise of the source counts the grid
   drew, not of the observed ones.
5. grid coverage (--grid_cls, the merged grid_cls.h5): the percentile of the observed binned Cls among all binned grid
   Cls, per Cl pair and ell bin. A percentile at 0 or 1 means the training set does not reach the observation.

Writes obs.npz (the four observed data vectors and Cls), summary.npz, cls.png and with --grid_cls coverage.png to
--out. CPU, tf_env (make_shape_noise_map uses tensorflow_probability), on a compute node:
    python dev/scripts/validation/data_vs_mock.py --config configs/desy6/v1/default.yaml \
        --tfrecord <grid .tfrecord> --out <dir> [--scan <scan.h5>] [--grid_cls <grid_cls.h5>]
"""

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import h5py  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from msfm.utils import catalog, files, observation  # noqa: E402

N_NOISE = 3
ELL_BANDS = ((30, 200), (200, 1000), (1000, None))
HIGH_ELL = 1000
# percentiles of the observation among the grid Cls that count as outside the grid
COVERAGE_EDGE = 0.001


def auto_index(i, n):
    """Index of the auto-spectrum of channel i in the ordering of power_spectra.get_cls: 11, 12, ..., 1n, 22, ..."""
    return i * n - i * (i - 1) // 2


def observe(conf, wl=None, gc=None, apply_sys=True):
    """The call of deep_lss.utils.evaluation.evaluate_obs_des. The function masks its inputs in place."""
    dv, cls, _ = observation.forward_model_observation_map(
        wl_gamma_map=None if wl is None else wl.copy(),
        gc_count_map=None if gc is None else gc.copy(),
        conf=conf,
        apply_norm=True,
        with_padding=True,
        nest_in=False,
        apply_maglim_sys_map=apply_sys,
    )
    return dv, cls


def read_mock(path, n_noise):
    """kappa (signal plus shape noise), lens counts and Cls of every record and noise realization of a .tfrecord."""
    import tensorflow as tf

    from msfm.utils import tfrecords

    kg, dg, cl = [], [], []
    for raw in tf.data.TFRecordDataset(path):
        inv = tfrecords.parse_inverse_grid(raw, range(n_noise), with_cross=False)
        for i in range(n_noise):
            kg.append(inv[f"kg_{i}"].numpy())
            dg.append(inv[f"dg_{i}"].numpy())
            cl.append(inv[f"cl_{i}"].numpy())
    return np.array(kg), np.array(dg), np.array(cl)


def check_observation(conf, footprint, n_wl, n_cl, mock_cl_shape):
    """The four variants of evaluate_obs_des, with their asserts. Returns them keyed like its labels."""
    wl, source_count = catalog.build_metacal_map_from_cat(conf)
    wl_no_rot = catalog.build_metacal_map_from_cat(conf, apply_shear_rotation=False, debug=False)[0]
    gc = catalog.build_maglim_map_from_cat(conf)

    obs = {}
    for rot, wl_map in (("", wl), ("_no_psi_rot", wl_no_rot)):
        for apply_sys in (True, False):
            obs[("sys" if apply_sys else "no_sys") + rot] = observe(conf, wl_map, gc, apply_sys)

    for key, (dv, cls) in obs.items():
        assert dv.shape == (len(footprint), n_wl + n_cl), f"{key}: data vector {dv.shape}"
        assert np.all(np.isfinite(dv)) and np.all(np.isfinite(cls)), f"{key}: not finite"
        assert not np.any(dv[~footprint]), f"{key}: the padding is not 0"
        assert cls.shape == mock_cl_shape, f"{key}: Cls {cls.shape}, the tfrecord's {mock_cl_shape}"

    def same(a, b):
        return np.allclose(a, b, rtol=1e-5, atol=0)

    for sys_key in ("sys", "no_sys"):
        a, b = obs[sys_key][0], obs[sys_key + "_no_psi_rot"][0]
        assert same(a[:, n_wl:], b[:, n_wl:]), "the psi rotation changes the clustering channels"
        assert not same(a[:, :n_wl], b[:, :n_wl]), "the psi rotation does not change the lensing channels"
    for rot in ("", "_no_psi_rot"):
        a, b = obs["sys" + rot][0], obs["no_sys" + rot][0]
        assert same(a[:, :n_wl], b[:, :n_wl]), "the lens weights change the lensing channels"
        assert not same(a[:, n_wl:], b[:, n_wl:]), "the lens weights do not change the clustering channels"
    print(f"[data_vs_mock] observation OK: 4 variants, {n_wl} + {n_cl} channels, Cls {mock_cl_shape}", flush=True)

    return obs, source_count


def noise_model(conf, source_count, footprint, n_wl):
    """Per-pixel kappa variance and high-ell Cl of the in_place and count shape noise at the observed counts, and the
    auto Cls of each method's first realization for the plot."""
    assert np.array_equal(source_count, np.round(source_count)), "the source counts are not integers"
    counts = source_count.astype(int)

    var, cl_high, cl_first = {}, {}, {}
    for method in ("in_place", "count"):
        var[method], cl_high[method] = [], []
        for i in range(N_NOISE):
            gamma = observation.make_shape_noise_map(counts, conf, method=method, noise_seed=12 + i)
            dv, cls = observe(conf, wl=gamma)
            var[method].append(dv[footprint].var(axis=0))
            cl_high[method].append([cls[HIGH_ELL:, auto_index(j, n_wl)].mean() for j in range(n_wl)])
            if i == 0:
                cl_first[method] = {j: cls[:, auto_index(j, n_wl)] for j in range(n_wl)}
            print(f"[data_vs_mock] noise {method} {i}: var {var[method][-1]}", flush=True)
    return {k: np.array(v) for k, v in var.items()}, {k: np.array(v) for k, v in cl_high.items()}, cl_first


def plot(path, labels, data_cl, mock_cl, noise_cl, n):
    edges = np.unique(np.geomspace(30, data_cl.shape[0], 25).astype(int))
    centres = np.sqrt(edges[:-1] * edges[1:])

    def binned(cl):
        return np.array([cl[lo:hi].mean() for lo, hi in zip(edges[:-1], edges[1:])])

    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 7), squeeze=False)
    for j, (ax, label) in enumerate(zip(axes.flat, labels)):
        k = auto_index(j, n)
        mock = np.array([binned(c[:, k]) for c in mock_cl])
        ax.fill_between(centres, *np.percentile(mock, [16, 84], axis=0), color="C0", alpha=0.3, label="mock 68%")
        ax.plot(centres, binned(data_cl[:, k]), color="k", label="data")
        for method, color in (("in_place", "C1"), ("count", "C2")):
            if j in noise_cl[method]:
                ax.plot(centres, binned(noise_cl[method][j]), color=color, ls="--", label=f"{method} noise")
        ax.set(xscale="log", yscale="log", xlabel=r"$\ell$", title=label)
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def grid_noise(scan_path, var_in_place, labels, n_wl):
    """The lensing noise variance of every grid record over the observation's in_place noise variance, per bin."""
    with h5py.File(scan_path, "r") as f:
        grid = f["noise_var"][:, :, :n_wl].reshape(-1, n_wl)
    ratio = grid[np.all(np.isfinite(grid), axis=1)] / var_in_place
    quantiles = np.percentile(ratio, [2.5, 50, 97.5], axis=0)

    print(f"\n[data_vs_mock] grid noise / in_place noise at the observed counts ({len(ratio)} grid records)")
    print(f"{'bin':>10s} {'median':>8s} {'2.5%':>8s} {'97.5%':>8s}")
    for j in range(n_wl):
        print(f"{labels[j]:>10s} {quantiles[1, j]:8.4f} {quantiles[0, j]:8.4f} {quantiles[2, j]:8.4f}")
    return quantiles


def pair_labels(labels):
    """Cl pair names in the ordering of power_spectra.get_cls."""
    return [f"{a} x {b}" for i, a in enumerate(labels) for b in labels[i:]]


def grid_coverage(conf, grid_cls_path, data_cl, labels):
    """Percentile of the observed binned Cls among all binned grid Cls, per ell bin and Cl pair."""
    from msfm.utils import power_spectra

    data_binned, bin_edges = power_spectra.bin_according_to_config(data_cl[np.newaxis], conf)
    data_binned = data_binned[0]
    below, total = np.zeros(data_binned.shape), 0
    with h5py.File(grid_cls_path, "r") as f:
        binned = f["cls/binned"]
        assert binned.shape[2:] == data_binned.shape, f"grid {binned.shape}, observation {data_binned.shape}"
        # one cosmology at a time, the whole set is several GB
        for k in range(binned.shape[0]):
            row = binned[k]
            below += np.sum(row < data_binned, axis=0)
            total += len(row)
    percentile = below / total

    pairs = pair_labels(labels)
    outside = (percentile < COVERAGE_EDGE) | (percentile > 1 - COVERAGE_EDGE)
    print(
        f"\n[data_vs_mock] observation among {total} grid Cls: percentile per pair over the {len(percentile)} ell bins"
    )
    print(f"{'pair':>22s} {'min':>7s} {'median':>7s} {'max':>7s} {'outside':>8s}")
    for p, pair in enumerate(pairs):
        q = percentile[:, p]
        print(f"{pair:>22s} {q.min():7.4f} {np.median(q):7.4f} {q.max():7.4f} {outside[:, p].sum():8d}")
    print(f"({outside.sum()} of {outside.size} beyond {COVERAGE_EDGE:g} of either end)")
    return percentile, np.squeeze(bin_edges), pairs


def plot_coverage(path, percentile, pairs):
    fig, ax = plt.subplots(figsize=(8, 0.18 * len(pairs) + 1.5))
    image = ax.imshow(percentile.T, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(pairs)), pairs, fontsize=6)
    ax.set_xlabel("ell bin")
    fig.colorbar(image, ax=ax, label="percentile of the observation among the grid Cls")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--tfrecord", required=True, help="a grid .tfrecord of the same config")
    p.add_argument("--out", required=True)
    p.add_argument("--scan", help="scan.h5 of dev/scripts/validation/grid_tfrecords.py, for check 4")
    p.add_argument("--grid_cls", help="the merged grid_cls.h5 of the same config, for check 5")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    conf = files.load_config(args.config)
    survey = conf["survey"]["name"]
    labels = conf["survey"]["metacal"]["z_bins"] + conf["survey"]["maglim"]["z_bins"]
    n_wl, n_cl = len(conf["survey"]["metacal"]["z_bins"]), len(conf["survey"]["maglim"]["z_bins"])
    n = n_wl + n_cl
    dv_masks = files.get_tomo_dv_masks(conf)
    footprint = dv_masks["metacal"][:, 0].astype(bool)
    assert all(np.array_equal(m.astype(bool), np.tile(footprint[:, None], m.shape[1])) for m in dv_masks.values())

    mock_kg, mock_dg, mock_cl = read_mock(args.tfrecord, conf["analysis"]["grid"]["n_noise_per_signal"])
    print(f"[data_vs_mock] {survey}: {len(mock_kg)} mock maps from {args.tfrecord}", flush=True)

    # 1. observation
    obs, source_count = check_observation(conf, footprint, n_wl, n_cl, mock_cl.shape[1:])
    np.savez(os.path.join(args.out, "obs.npz"), **{f"{k}__{q}": v for k, vs in obs.items() for q, v in zip("dc", vs)})
    data_dv, data_cl = obs["sys"]

    # 2. noise model
    var, cl_high, noise_cl = noise_model(conf, source_count, footprint, n_wl)

    # 3. data vs mock
    mock_dv = np.concatenate([mock_kg, mock_dg], axis=-1)
    data_std = data_dv[footprint].std(axis=0)
    mock_std = mock_dv[:, footprint].std(axis=1)
    data_band = np.array([[data_cl[lo:hi, auto_index(j, n)].mean() for lo, hi in ELL_BANDS] for j in range(n)])
    mock_band = np.array(
        [[[c[lo:hi, auto_index(j, n)].mean() for lo, hi in ELL_BANDS] for j in range(n)] for c in mock_cl]
    )

    print(f"\n[data_vs_mock] {survey}, noise model at the observed counts ({N_NOISE} realizations each)")
    print(f"{'bin':>10s} {'var count/in_place':>20s} {'Cl(l>1000) count/in_place':>27s} {'data var/in_place':>18s}")
    var_ratio = var["count"].mean(0) / var["in_place"].mean(0)
    cl_ratio = cl_high["count"].mean(0) / cl_high["in_place"].mean(0)
    data_over_noise = data_dv[footprint, :n_wl].var(axis=0) / var["in_place"].mean(0)
    for j in range(n_wl):
        print(f"{labels[j]:>10s} {var_ratio[j]:20.4f} {cl_ratio[j]:27.4f} {data_over_noise[j]:18.4f}")

    print(f"\n[data_vs_mock] {survey}, data / mock (mock: mean of {len(mock_dv)} maps)")
    bands = "".join(f"{f'Cl {lo}-{hi or data_cl.shape[0]}':>14s}" for lo, hi in ELL_BANDS)
    print(f"{'bin':>10s} {'std':>8s}{bands}")
    std_ratio = data_std / mock_std.mean(0)
    band_ratio = data_band / mock_band.mean(0)
    for j in range(n):
        print(f"{labels[j]:>10s} {std_ratio[j]:8.4f}" + "".join(f"{r:14.4f}" for r in band_ratio[j]))
    print(f"(mock scatter of the std, relative: {np.array2string(mock_std.std(0) / mock_std.mean(0), precision=4)})")

    extra = {}
    # 4. grid noise
    if args.scan:
        extra["grid_noise_over_in_place_q"] = grid_noise(args.scan, var["in_place"].mean(0), labels, n_wl)

    # 5. grid coverage
    if args.grid_cls:
        percentile, bin_edges, pairs = grid_coverage(conf, args.grid_cls, data_cl, labels)
        extra.update(coverage_percentile=percentile, coverage_bin_edges=bin_edges, coverage_pairs=pairs)
        plot_coverage(os.path.join(args.out, "coverage.png"), percentile, pairs)

    np.savez(
        os.path.join(args.out, "summary.npz"),
        **extra,
        labels=labels,
        ell_bands=np.array([(lo, hi or data_cl.shape[0]) for lo, hi in ELL_BANDS]),
        var_count_over_in_place=var_ratio,
        cl_high_count_over_in_place=cl_ratio,
        data_var_over_in_place=data_over_noise,
        std_data_over_mock=std_ratio,
        band_data_over_mock=band_ratio,
        var=np.array([var["in_place"], var["count"]]),
        mock_std=mock_std,
        mock_band=mock_band,
    )
    plot(os.path.join(args.out, "cls.png"), labels, data_cl, mock_cl, noise_cl, n)
    print(f"\n[data_vs_mock] OK, wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
