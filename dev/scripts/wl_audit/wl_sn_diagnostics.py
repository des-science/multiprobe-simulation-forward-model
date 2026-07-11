# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created July 2026
Author: Arne Thomsen (with Claude)

Weak lensing S/N and delta-NLA audit diagnostics for the v16 rot_in_place forward model
(see plans/do-a-conceptual-audit-golden-truffle.md):

 1. noise level: pseudo-Cl of in_place shape-noise kappa_E maps generated exactly like
    postprocess_shape_noise, compared against the analytic expectation from the per-pixel
    catalog variance sum(w^2 |e|^2) / (sum w)^2 (E-modes get half the two-component power).
    Optionally cross-checked against the high-ell plateau of the stored grid Cls.
 2. signal level: full-sky Cl of a fiducial CosmoGrid kappa map vs. the pseudo-Cl of the same
    map after the full kappa -> gamma -> patch -> pseudo-E -> kappa mode-removal chain,
    rescaled by the patch sky fraction. A ratio ~1 at intermediate ell confirms that the
    Kaiser-Squires round trip does not lose signal amplitude.
 3. ds amplitude: full-sky Cls of the delta-NLA ds = (ia - <ia>) * delta_2D template vs. the
    ia map itself, per metacal bin. Cl(ds)/Cl(ia) bounds the relative effect of bta.

Meant to be run on a Clariden CPU allocation via wl_sn_diagnostics.sh. Results (one .npz and
diagnostic .png figures) go to --out_dir on scratch.
"""

import argparse, os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py

from msfm.utils import files, filenames, imports, lensing, logger, maps, postprocessing, scales

hp = imports.import_healpy()
LOGGER = logger.get_logger(__file__)


def setup(args=None):
    parser = argparse.ArgumentParser(description="WL S/N and delta-NLA audit diagnostics")
    parser.add_argument(
        "--config",
        type=str,
        default="/users/athomsen/dlss/repos/multiprobe-simulation-forward-model/configs/v16/rot_in_place.yaml",
    )
    parser.add_argument(
        "--fiducial_perm_dir",
        type=str,
        default="/users/athomsen/scratch/deep_lss/data/projected/fiducial/cosmo_fiducial/perm_0000",
        help="CosmoGrid permutation used for the signal and ds diagnostics",
    )
    parser.add_argument(
        "--grid_cls_file",
        type=str,
        default="/users/athomsen/scratch/deep_lss/data/v16/rot_in_place/cls/grid_cls.h5",
        help="stored grid Cls for the high-ell noise-plateau cross-check (skipped if missing)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/users/athomsen/scratch/deep_lss/runs/wl_audit",
    )
    parser.add_argument("--n_noise", type=int, default=3, help="noise realizations per bin for diagnostic 1")
    return parser.parse_args(args)


def _hp_datapath(conf):
    file_dir = os.path.dirname(os.path.abspath(files.__file__))
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    return os.path.join(repo_dir, conf["files"]["healpy_data"])


def _cross_index(i, j, n):
    """Index of the (i, j <= i swapped) pair in the 11, 12, ..., 1n, 22, ... enumeration of get_cls."""
    i, j = min(i, j), max(i, j)
    return i * n - i * (i - 1) // 2 + (j - i)


def _smooth_dv(dv, i_z, conf, pixel_file, mask):
    """Single-bin version of the lensing smoothing in run_grid_postprocessing._get_lensing_transform."""
    sc = conf["analysis"]["scale_cuts"]["lensing"]
    l_min = sc["l_min"][i_z] if isinstance(sc["l_min"], list) else sc["l_min"]
    l_max = sc["l_max"][i_z] if isinstance(sc["l_max"], list) else sc["l_max"]
    theta_fwhm = sc["theta_fwhm"][i_z] if isinstance(sc["theta_fwhm"], list) else sc["theta_fwhm"]

    return scales.data_vector_to_smoothed_data_vector(
        dv,
        data_vec_pix=pixel_file[0],
        n_side=conf["analysis"]["n_side"],
        l_min=l_min,
        l_max=l_max,
        theta_fwhm=theta_fwhm,
        arcmin=True,
        mask=mask,
        conf=conf,
        hard_cut=conf["analysis"]["scale_cuts"]["hard_cut"],
    )


def _read_full_sky_bin(conf, full_maps_file, in_map_type, z_bin):
    return postprocessing._read_full_sky_bin(conf, full_maps_file, in_map_type, z_bin)


# diagnostic 1: shape-noise level #####################################################################################


def noise_level(conf, pixel_file, noise_file, args, results):
    LOGGER.warning("Diagnostic 1: in_place shape-noise level vs. analytic expectation")

    n_side = conf["analysis"]["n_side"]
    n_pix = hp.nside2npix(n_side)
    n_z = len(conf["survey"]["metacal"]["z_bins"])
    l = np.arange(3 * n_side)

    metacal_mask = files.get_tomo_dv_masks(conf)["metacal"]
    data_vec_pix, patches_pix_dict, corresponding_pix_dict, _ = pixel_file
    data_vec_len = len(data_vec_pix)

    _, gamma2kappa_fac, _ = lensing.get_kaiser_squires_factors(3 * n_side - 1)
    hp_datapath = _hp_datapath(conf)

    # the mode removal multiplies the E alm by gamma2kappa_fac, so the white E level is scaled by its square
    ks_cl_fac = np.ones_like(l, dtype=np.float64)
    ks_cl_fac[2:] = (l[2:] * (l[2:] + 1.0)) / ((l[2:] + 2.0) * (l[2:] - 1.0))
    ks_cl_fac[:2] = 0.0

    emp_cls = np.zeros((n_z, args.n_noise, len(l)))
    ana_cls = np.zeros((n_z, len(l)))
    for i_z in range(n_z):
        gamma_cat = noise_file[i_z]
        gamma_abs = np.abs(gamma_cat[:, 0] + 1j * gamma_cat[:, 1])
        w = gamma_cat[:, 2]
        pix_cat = gamma_cat[:, 3]

        base_patch_pix = patches_pix_dict["metacal"][i_z][0]
        corresponding_pix = corresponding_pix_dict["metacal"][i_z]

        # analytic: pseudo-Cl of a masked map with independent pixels of per-component variance var_ref / 2,
        # <Cl> = (4 pi / n_pix^2) * sum_i sigma_i^2, and the KS factor from the E -> kappa conversion
        var_ref = lensing.shape_noise_variance_map(gamma_abs, w, pix_cat, n_pix)
        ana_cls[i_z] = 4.0 * np.pi / n_pix**2 * np.sum(var_ref[base_patch_pix] / 2.0) * ks_cl_fac

        # empirical: exactly the postprocess_shape_noise in_place path
        gamma1, gamma2 = lensing.noise_gen_in_place(
            gamma_abs.astype(np.float32), w, pix_cat, base_patch_pix, n_pix, args.n_noise
        )
        for i_noise in range(args.n_noise):
            gamma1_patch = np.zeros(n_pix, dtype=np.float32)
            gamma1_patch[base_patch_pix] = gamma1[:, i_noise]
            gamma2_patch = np.zeros(n_pix, dtype=np.float32)
            gamma2_patch[base_patch_pix] = gamma2[:, i_noise]

            kappa_patch = lensing.mode_removal(
                gamma1_patch, gamma2_patch, gamma2kappa_fac, n_side, hp_datapath=hp_datapath
            )
            kappa_dv = maps.map_to_data_vec(
                hp_map=kappa_patch,
                data_vec_len=data_vec_len,
                corresponding_pix=corresponding_pix,
                cutout_pix=base_patch_pix,
                remove_mean=True,
            )
            _, alm = _smooth_dv(kappa_dv, i_z, conf, pixel_file, metacal_mask[:, i_z])
            emp_cls[i_z, i_noise] = hp.alm2cl(alm)

        ratio = np.mean(emp_cls[i_z], axis=0)[300:1500] / ana_cls[i_z][300:1500]
        LOGGER.warning(
            f"metacal{i_z + 1}: empirical / analytic noise Cl (median over ell in [300, 1500]) "
            f"= {np.median(ratio):.4f}"
        )

    results["noise_emp_cls"] = emp_cls
    results["noise_ana_cls"] = ana_cls

    # optional cross-check against the high-ell plateau of the stored (signal + noise) grid Cls
    if os.path.exists(args.grid_cls_file):
        LOGGER.info(f"Cross-checking against the stored grid Cls in {args.grid_cls_file}")
        with h5py.File(args.grid_cls_file, "r") as f:
            n_cross = f["cls/raw"].shape[-1]
            # 8 channels (4 wl + 4 gc) -> 36, 4 channels (wl only) -> 10
            n_channels = int((np.sqrt(8 * n_cross + 1) - 1) / 2)
            # subset of cosmologies, first noise block (i_noise = 0)
            stored = f["cls/raw"][::250, :80]
        wl_auto = np.stack([stored[..., _cross_index(i, i, n_channels)] for i in range(n_z)], axis=-1)
        results["grid_plateau_cls"] = np.mean(wl_auto, axis=(0, 1))
        for i_z in range(n_z):
            plateau = np.median(results["grid_plateau_cls"][1200:1500, i_z])
            ana = np.median(ana_cls[i_z][1200:1500])
            LOGGER.warning(
                f"metacal{i_z + 1}: stored grid (kg + sn) Cl plateau / analytic noise Cl "
                f"(ell in [1200, 1500]) = {plateau / ana:.4f} (> 1 expected: includes signal)"
            )
    else:
        LOGGER.warning(f"{args.grid_cls_file} not found, skipping the stored-Cls cross-check")

    # figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for i_z, ax in enumerate(axes.flat):
        ax.loglog(l[2:], np.mean(emp_cls[i_z], axis=0)[2:], label="empirical in_place noise")
        ax.loglog(l[2:], ana_cls[i_z][2:], "--", label="analytic")
        if "grid_plateau_cls" in results:
            ax.loglog(l[2:], results["grid_plateau_cls"][2:, i_z], ":", label="stored grid kg+sn", alpha=0.7)
        ax.set_title(f"metacal{i_z + 1}")
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$C_\ell$")
        ax.legend()
    fig.suptitle("Diagnostic 1: shape-noise pseudo-$C_\\ell$ (kappa E-mode, masked)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "noise_level.png"), dpi=150)
    plt.close(fig)


# diagnostic 2: signal level through the KS / mode-removal chain ######################################################


def signal_level(conf, pixel_file, args, results):
    LOGGER.warning("Diagnostic 2: signal amplitude through the KS round trip and mode removal")

    n_side = conf["analysis"]["n_side"]
    n_pix = hp.nside2npix(n_side)
    n_z = len(conf["survey"]["metacal"]["z_bins"])
    l = np.arange(3 * n_side)

    metacal_mask = files.get_tomo_dv_masks(conf)["metacal"]
    _, patches_pix_dict, _, _ = pixel_file
    hp_datapath = _hp_datapath(conf)

    full_maps_file = filenames.get_filename_full_maps(
        args.fiducial_perm_dir, with_bary=conf["analysis"]["modelling"]["baryonified"]
    )

    full_cls = np.zeros((n_z, len(l)))
    patch_cls = np.zeros((n_z, len(l)))
    f_patch = np.zeros(n_z)
    for i_z, z_bin in enumerate(conf["survey"]["metacal"]["z_bins"]):
        kappa_full = _read_full_sky_bin(conf, full_maps_file, "kg", z_bin)

        alm_full = hp.map2alm(kappa_full, use_pixel_weights=True, datapath=hp_datapath)
        full_cls[i_z] = hp.alm2cl(alm_full)

        # the exact pipeline: kappa -> gamma (full sky) -> patch cutout -> pseudo-E -> kappa -> dv
        kappa_dvs = postprocessing.postprocess_lensing(kappa_full, conf, pixel_file, i_z)
        _, alm_patch = _smooth_dv(kappa_dvs[0], i_z, conf, pixel_file, metacal_mask[:, i_z])
        patch_cls[i_z] = hp.alm2cl(alm_patch)

        f_patch[i_z] = len(patches_pix_dict["metacal"][i_z][0]) / n_pix
        ratio = patch_cls[i_z][100:1000] / (f_patch[i_z] * full_cls[i_z][100:1000])
        LOGGER.warning(
            f"metacal{i_z + 1}: pseudo-Cl / (f_patch * full-sky Cl) (median over ell in [100, 1000]) "
            f"= {np.median(ratio):.4f} (~1 means no amplitude loss in KS / mode removal)"
        )

    results["signal_full_cls"] = full_cls
    results["signal_patch_cls"] = patch_cls
    results["signal_f_patch"] = f_patch

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for i_z, ax in enumerate(axes.flat):
        ax.semilogx(l[2:], patch_cls[i_z][2:] / (f_patch[i_z] * full_cls[i_z][2:]))
        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.set_ylim(0, 1.5)
        ax.set_title(f"metacal{i_z + 1}")
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"pseudo-$C_\ell$ / ($f_{\rm patch}$ full-sky $C_\ell$)")
    fig.suptitle("Diagnostic 2: KS + mode-removal signal transfer")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "signal_level.png"), dpi=150)
    plt.close(fig)


# diagnostic 3: delta-NLA ds template amplitude #######################################################################


def ds_amplitude(conf, args, results):
    LOGGER.warning("Diagnostic 3: delta-NLA ds template amplitude (full sky)")

    n_side = conf["analysis"]["n_side"]
    n_z = len(conf["survey"]["metacal"]["z_bins"])
    l = np.arange(3 * n_side)
    hp_datapath = _hp_datapath(conf)

    full_maps_file = filenames.get_filename_full_maps(
        args.fiducial_perm_dir, with_bary=conf["analysis"]["modelling"]["baryonified"]
    )

    cl_kg = np.zeros((n_z, len(l)))
    cl_ia = np.zeros((n_z, len(l)))
    cl_ds = np.zeros((n_z, len(l)))
    cl_ia_ds = np.zeros((n_z, len(l)))
    for i_z, z_bin in enumerate(conf["survey"]["metacal"]["z_bins"]):
        kg = _read_full_sky_bin(conf, full_maps_file, "kg", z_bin)
        ia = _read_full_sky_bin(conf, full_maps_file, "ia", z_bin)
        dg = _read_full_sky_bin(conf, full_maps_file, "dg", z_bin)

        # exactly like postprocess_metacal_bin
        ds = (ia - np.mean(ia)) * ((dg - np.mean(dg)) / np.mean(dg))

        alm_kg = hp.map2alm(kg, use_pixel_weights=True, datapath=hp_datapath)
        alm_ia = hp.map2alm(ia, use_pixel_weights=True, datapath=hp_datapath)
        alm_ds = hp.map2alm(ds, use_pixel_weights=True, datapath=hp_datapath)

        cl_kg[i_z] = hp.alm2cl(alm_kg)
        cl_ia[i_z] = hp.alm2cl(alm_ia)
        cl_ds[i_z] = hp.alm2cl(alm_ds)
        cl_ia_ds[i_z] = hp.alm2cl(alms1=alm_ia, alms2=alm_ds)

        sl = slice(100, 1000)
        bta_max = conf["analysis"]["grid"]["priors"]["bta"][1]
        # the IA term is Aia * (ia + bta * ds): relative modification of its power at bta = bta_max
        rel = (2.0 * bta_max * cl_ia_ds[i_z][sl] + bta_max**2 * cl_ds[i_z][sl]) / cl_ia[i_z][sl]
        LOGGER.warning(
            f"metacal{i_z + 1}: Cl(ds)/Cl(ia) (median, ell in [100, 1000]) "
            f"= {np.median(cl_ds[i_z][sl] / cl_ia[i_z][sl]):.3e}; "
            f"relative IA-term power change at bta={bta_max}: {np.median(rel):.3e}"
        )

    results["ds_cl_kg"] = cl_kg
    results["ds_cl_ia"] = cl_ia
    results["ds_cl_ds"] = cl_ds
    results["ds_cl_ia_ds"] = cl_ia_ds

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for i_z, ax in enumerate(axes.flat):
        ax.loglog(l[2:], cl_kg[i_z][2:], label=r"$C_\ell(\kappa_g)$")
        ax.loglog(l[2:], cl_ia[i_z][2:], label=r"$C_\ell({\rm ia})$")
        ax.loglog(l[2:], cl_ds[i_z][2:], label=r"$C_\ell({\rm ds})$")
        ax.set_title(f"metacal{i_z + 1}")
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$C_\ell$")
        ax.legend()
    fig.suptitle("Diagnostic 3: delta-NLA ds template vs. ia and kg (full sky, fiducial)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "ds_amplitude.png"), dpi=150)
    plt.close(fig)


def main(args=None):
    args = setup(args)
    os.makedirs(args.out_dir, exist_ok=True)

    conf = files.load_config(args.config)
    pixel_file = files.load_pixel_file(conf)
    noise_file = files.load_noise_file(conf)

    results = {}
    noise_level(conf, pixel_file, noise_file, args, results)
    signal_level(conf, pixel_file, args, results)
    ds_amplitude(conf, args, results)

    out_file = os.path.join(args.out_dir, "wl_sn_diagnostics.npz")
    np.savez(out_file, **results)
    LOGGER.warning(f"Saved all results to {out_file}")


if __name__ == "__main__":
    main()
