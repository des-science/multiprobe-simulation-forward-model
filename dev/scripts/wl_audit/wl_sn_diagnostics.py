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
 4. sn_ratio: the kg-signal vs. shape-noise (sn) S/N on the same masked-KS pseudo-Cl footing as
    diagnostics 1/2 (the mask/KS transfer cancels in the ratio), reported as the per-mode
    Cl_s/Cl_n(ell), the ell where it crosses 1, and the Gaussian detection S/N per DES-area patch
    as a function of the l_max cut. Anchored to the published DES Y3 cosmic-shear S/N (~27) to check
    the data path is not secretly shape-noise-swamping the signal. This is the core check of whether
    WL constraining power is unknowingly reduced at the data level.
 5. ia_scale: the intrinsic-alignment (NLA) contribution to the observable kg + tomo_Aia * ia,
    relative to kg, at the fiducial Aia and across the Aia prior -- IA/kg amplitude and II/GG power
    fractions per z-bin, to confirm the IA amplitude is of the right physical scale.
 6. component_budget: the full-sky input Cl of each lensing component (kg signal, ia intrinsic
    alignments, sn shape noise) vs. its spectrum after the full masking + KS mode-removal + smoothing
    chain, all on a common full-sky-equivalent footing (masked pseudo-Cls / f_patch) so the residual
    input -> final gap is the KS/mask/smoothing transfer, not the trivial f_sky suppression. Shape noise
    has no full-sky map, so its input is the flat white E-mode level; the only new work is pushing the
    ia template through postprocess_lensing.

Meant to be run on a Clariden CPU allocation via wl_sn_diagnostics.sh. Results (one .npz and
diagnostic .png figures) go to --out_dir on scratch. Diagnostics 4, 5 and 6 only consume the spectra
computed by 1-3 (plus, for 6, one extra ia forward pass), so they add little runtime.
"""

import argparse, os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py

from msfm.utils import files, filenames, imports, lensing, logger, maps, postprocessing, redshift, scales

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
    parser.add_argument(
        "--sn_lmin",
        type=int,
        default=30,
        help="lower ell of the S/N sums (diagnostic 4); matches the hard scale_cuts.lensing.l_min",
    )
    parser.add_argument(
        "--lmax_cuts",
        type=int,
        nargs="+",
        default=[589, 863, 1159, 1382],
        help="per-bin l_max at which to quote the analysis S/N. This is the PHYSICAL analysis cut "
        "(default = the 8 Mpc/h '8wl' cuts), which is deliberately SEPARATE from the smoothing band: "
        "v16/rot_in_place.yaml sets scale_cuts.lensing.l_max=Null (smoothing at 3*n_side-1), so the "
        "S/N-quoting cut is not derivable from the config. Pass explicitly for other cuts, e.g. "
        "[1024, 1024, 1024, 1024] for the lmax_1024 run",
    )
    parser.add_argument(
        "--des_ref_sn",
        type=float,
        default=27.0,
        help="published DES Y3 cosmic-shear detection S/N used only as a physical anchor in the log",
    )
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
    white_level = np.zeros(n_z)  # flat white E-mode level (no KS tilt, no mask), for the component budget
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
        # flat white E-mode level (no KS tilt, no mask transfer); ana_cls applies the KS (E -> kappa) factor
        white_level[i_z] = 4.0 * np.pi / n_pix**2 * np.sum(var_ref[base_patch_pix] / 2.0)
        ana_cls[i_z] = white_level[i_z] * ks_cl_fac

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
    results["noise_white_level"] = white_level

    # optional cross-check against the high-ell plateau of the stored (signal + noise) grid Cls
    if os.path.exists(args.grid_cls_file):
        LOGGER.info(f"Cross-checking against the stored grid Cls in {args.grid_cls_file}")
        with h5py.File(args.grid_cls_file, "r") as f:
            n_cross = f["cls/raw"].shape[-1]
            # 8 channels (4 wl + 4 gc) -> 36, 4 channels (wl only) -> 10
            n_channels = int((np.sqrt(8 * n_cross + 1) - 1) / 2)
            # this cross-check assumes the WL autos are the first n_z channels of cls/raw (WL before
            # GC in the get_cls enumeration); guard against a channel-selection regression
            assert n_channels >= n_z, f"cls/raw has {n_channels} channels < n_z={n_z}"
            # thin to every 250th cosmology, first 80 entries of the second axis
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
            f"= {np.median(ratio):.4f} (this is the COMBINED transfer: mask + KS/mode-removal E/B "
            f"leakage, plus any beam/l_max only if the config sets theta_fwhm/l_max -- both Null in "
            f"v16/rot_in_place. patch_cls goes through _smooth_dv, full_cls is raw, so a value < 1 is "
            f"not by itself KS signal loss; it cancels in the diag-4 S/N, where signal and noise "
            f"share this identical footing)"
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
    cl_kg_ia = np.zeros((n_z, len(l)))  # GI cross term: the dominant IA contribution to the observable
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
        cl_kg_ia[i_z] = hp.alm2cl(alms1=alm_kg, alms2=alm_ia)

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
    results["ds_cl_kg_ia"] = cl_kg_ia

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


# diagnostic 4: kg-signal vs. shape-noise S/N and how it changes with scale ##########################################


def sn_ratio(conf, args, results):
    """Signal-to-noise of the weak-lensing signal (kg) against the shape noise (sn), on the SAME
    masked-KS pseudo-Cl footing produced by diagnostics 1 and 2, and how it changes with the ell cut.

    Both inputs are pseudo-Cls of maps that went through the identical kappa -> gamma -> patch ->
    pseudo-E -> kappa -> dv -> smoothing chain, so the mask/KS transfer that suppresses each of them
    (signal ~0.80, noise ~0.89, section 3.2) cancels in the ratio Cl_s / (Cl_s + Cl_n) up to the small
    E-only-vs-isotropic difference -- the S/N below is therefore what the network actually sees, not an
    idealized full-sky forecast.

    Reported per DES-footprint patch (one patch = one DES-area mock survey; 4 disjoint patches per
    permutation are ~independent, section 3.5): the Gaussian detection significance
        (S/N)^2 = sum_ell (2 ell + 1) f_patch / 2 * [Cl_s / (Cl_s + Cl_n)]^2,
    auto-spectra only (no tomographic cross-bin terms, so this is a conservative lower bound on the
    full tomographic S/N), swept as a function of the upper ell cut.
    """
    LOGGER.warning("Diagnostic 4: kg-signal vs. shape-noise S/N and its scale dependence")

    l = np.arange(3 * conf["analysis"]["n_side"])
    n_z = len(conf["survey"]["metacal"]["z_bins"])
    lmin = args.sn_lmin

    sig = results["signal_patch_cls"]  # pure fiducial kg, pseudo-Cl through the full chain
    noi = np.mean(results["noise_emp_cls"], axis=1)  # in_place shape noise, same chain, averaged
    f_patch = results["signal_f_patch"]

    lmax_cuts = np.atleast_1d(args.lmax_cuts)
    if lmax_cuts.size == 1:
        lmax_cuts = np.full(n_z, lmax_cuts[0])

    with np.errstate(invalid="ignore", divide="ignore"):
        cl_ratio = np.where((sig > 0) & (noi > 0), sig / noi, np.nan)  # per-mode Cl_s / Cl_n
        info_frac = np.nan_to_num(sig / (sig + noi))  # Cl_s / (Cl_s + Cl_n), the S/N weight

    # per-mode density d(S/N)^2 / d ell and its cumulative sum from lmin
    sn2_density = (2.0 * l + 1.0) * f_patch[:, None] / 2.0 * info_frac**2
    sn2_density[:, :lmin] = 0.0
    sn2_cumulative = np.cumsum(sn2_density, axis=1)  # (n_z, n_ell), = (S/N)^2(< ell)

    l_cross = np.full(n_z, -1)
    sn_at_cut = np.zeros(n_z)
    sn_full = np.zeros(n_z)
    for i_z in range(n_z):
        # first ell >= lmin at which the per-mode signal Cl drops below the noise Cl
        below = np.where((l >= lmin) & (cl_ratio[i_z] < 1.0))[0]
        l_cross[i_z] = int(below[0]) if below.size else -1
        sn_at_cut[i_z] = np.sqrt(sn2_cumulative[i_z, min(lmax_cuts[i_z], len(l) - 1)])
        sn_full[i_z] = np.sqrt(sn2_cumulative[i_z, -1])
        LOGGER.warning(
            f"metacal{i_z + 1}: per-mode Cl_s/Cl_n peaks at {np.nanmax(cl_ratio[i_z][lmin:]):.2f} "
            f"(l={lmin + int(np.nanargmax(cl_ratio[i_z][lmin:]))}), crosses 1 at l={l_cross[i_z]}; "
            f"detection S/N per patch = {sn_at_cut[i_z]:.2f} (l<={lmax_cuts[i_z]}) / "
            f"{sn_full[i_z]:.2f} (l<={len(l) - 1})"
        )

    sn_combined_cut = np.sqrt(np.sum(sn_at_cut**2))
    sn_combined_full = np.sqrt(np.sum(sn_full**2))
    LOGGER.warning(
        f"auto-only quadrature over bins, ONE DES-area patch: S/N = {sn_combined_cut:.1f} "
        f"(l<={list(lmax_cuts)}) / {sn_combined_full:.1f} (no l_max) "
        f"-- physical anchor: published DES Y3 cosmic-shear detection S/N ~ {args.des_ref_sn:.0f} "
        f"(all auto+cross, real space). A map S/N of this order (not << it) means the data path is "
        f"NOT secretly shape-noise-swamping the signal."
    )

    results["sn_cl_ratio"] = cl_ratio
    results["sn2_cumulative"] = sn2_cumulative
    results["sn_l_cross"] = l_cross
    results["sn_lmax_cuts"] = lmax_cuts
    results["sn_per_bin_at_cut"] = sn_at_cut
    results["sn_per_bin_full"] = sn_full

    # figure: (left) per-mode Cl_s/Cl_n, (right) cumulative detection S/N(<ell) -> "how S/N changes"
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 5))
    for i_z in range(n_z):
        axl.loglog(l[lmin:], cl_ratio[i_z][lmin:], label=f"metacal{i_z + 1}")
        axr.semilogx(l[lmin:], np.sqrt(sn2_cumulative[i_z][lmin:]), label=f"metacal{i_z + 1}")
        axr.axvline(lmax_cuts[i_z], color=f"C{i_z}", ls=":", lw=0.8)
    axl.axhline(1.0, color="k", ls="--", lw=0.8)
    axl.set_xlabel(r"$\ell$")
    axl.set_ylabel(r"per-mode $C_\ell^{\rm kg} / C_\ell^{\rm sn}$")
    axl.set_title("per-mode signal / shape-noise")
    axl.legend()
    axr.semilogx(
        l[lmin:], np.sqrt(np.sum(sn2_cumulative, axis=0))[lmin:], "k-", lw=2, label="auto-combined"
    )
    axr.set_xlabel(r"$\ell_{\max}$")
    axr.set_ylabel(r"detection S/N ($\ell_{\min} \leq \ell \leq \ell_{\max}$), one patch")
    axr.set_title("cumulative detection S/N vs. scale cut")
    axr.legend()
    fig.suptitle("Diagnostic 4: kg vs. shape-noise S/N (masked-KS pseudo-$C_\\ell$, per DES-area patch)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "sn_ratio.png"), dpi=150)
    plt.close(fig)


# diagnostic 5: intrinsic-alignment amplitude scale ##################################################################


def ia_scale(conf, args, results):
    """Is the intrinsic-alignment amplitude of the right physical scale? Quantifies the IA (NLA)
    contribution to the observable kappa = kg + tomo_Aia * ia relative to the lensing (kg) signal, at
    the fiducial Aia and across the Aia prior, using the full-sky Cls from diagnostic 3.

    tomo_Aia is the per-bin NLA amplitude from redshift.get_tomo_amplitudes (the DeepLSS (1+z)^n_Aia
    evolution) at the config fiducial (Aia, n_Aia); ia is the A_IA = 1 NLA template map, so the II auto
    power at fiducial is tomo_Aia^2 * Cl(ia) and the amplitude fraction is tomo_Aia * sqrt(Cl(ia)/Cl(kg)).
    For DES Y3 NLA (A_IA ~ 0.3-0.7) the IA amplitude fraction is expected at the ~10-20 % level in the
    lowest-z bin and to fall with z (lensing kernel grows, IA template ~flat): the signature to confirm.
    """
    LOGGER.warning("Diagnostic 5: intrinsic-alignment amplitude scale")

    n_z = len(conf["survey"]["metacal"]["z_bins"])
    l = np.arange(3 * conf["analysis"]["n_side"])
    sl = slice(100, 1000)

    tomo_z, tomo_nz = files.load_redshift_distributions("metacal", conf)
    fid = conf["analysis"]["fiducial"]
    nla_conf = conf["analysis"]["modelling"]["lensing"]["nla"]
    tomo_Aia = redshift.get_tomo_amplitudes(
        fid["Aia"],
        fid["n_Aia"],
        tomo_z,
        tomo_nz,
        z0=conf["survey"]["metacal"]["z0"],
        truncate_nz=nla_conf["truncate_nz"],
        z_min_quantile=nla_conf["z_min_quantile"],
        z_max_quantile=nla_conf["z_max_quantile"],
    )
    Aia_prior = conf["analysis"]["grid"]["priors"]["Aia"]

    cl_kg = results["ds_cl_kg"]
    cl_ia = results["ds_cl_ia"]
    cl_kg_ia = results.get("ds_cl_kg_ia")  # GI cross term (may be absent in an old npz)

    ia_over_kg_amp = np.zeros(n_z)  # sqrt(Cl_ia / Cl_kg), per unit A_IA (template only)
    gi_corr = np.full(n_z, np.nan)  # Pearson r(kg, ia) per bin: sign + strength of the GI term
    ia_obs_frac_fid = np.zeros(n_z)  # (Cl_obs - Cl_kg) / Cl_kg at fiducial: full II + GI impact
    for i_z in range(n_z):
        ia_over_kg_amp[i_z] = np.sqrt(np.median(cl_ia[i_z][sl] / cl_kg[i_z][sl]))
        amp_frac_fid = tomo_Aia[i_z] * ia_over_kg_amp[i_z]
        pow_frac_fid = amp_frac_fid**2  # II auto / GG power

        # full observable kappa = kg + tomo_Aia * ia:  Cl_obs = Cl_kg + 2 tomo_Aia Cl_kg,ia + tomo_Aia^2 Cl_ia
        if cl_kg_ia is not None:
            gi_corr[i_z] = np.median(cl_kg_ia[i_z][sl] / np.sqrt(cl_kg[i_z][sl] * cl_ia[i_z][sl]))
            gi_frac = 2.0 * tomo_Aia[i_z] * np.median(cl_kg_ia[i_z][sl] / cl_kg[i_z][sl])  # GI / GG
            ia_obs_frac_fid[i_z] = gi_frac + pow_frac_fid
            gi_msg = (
                f"GI/GG {gi_frac:+.3f} (r={gi_corr[i_z]:+.2f}), "
                f"total (Cl_obs-Cl_kg)/Cl_kg {ia_obs_frac_fid[i_z]:+.3f}"
            )
        else:
            gi_msg = "GI cross term absent (rerun to populate ds_cl_kg_ia)"

        tomo_edge = tomo_Aia[i_z] / fid["Aia"] * max(abs(Aia_prior[0]), abs(Aia_prior[1]))
        LOGGER.warning(
            f"metacal{i_z + 1}: tomo_Aia(fid)={tomo_Aia[i_z]:.3f}, "
            f"template sqrt(Cl_ia/Cl_kg)={ia_over_kg_amp[i_z]:.3f}; "
            f"II/GG power {pow_frac_fid:.3f}, {gi_msg}; "
            f"IA/kg amplitude {tomo_edge * ia_over_kg_amp[i_z]:.2f} "
            f"at |Aia|={max(abs(Aia_prior[0]), abs(Aia_prior[1]))}"
        )

    results["ia_tomo_Aia_fid"] = tomo_Aia
    results["ia_over_kg_amp"] = ia_over_kg_amp
    results["ia_gi_corr"] = gi_corr
    results["ia_obs_power_frac_fid"] = ia_obs_frac_fid

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.arange(1, n_z + 1)
    ax.plot(bins, tomo_Aia * ia_over_kg_amp, "o-", label="II amplitude / kg at fiducial")
    ax.plot(
        bins,
        ia_over_kg_amp,
        "s--",
        label=r"template sqrt($C_\ell^{\rm ia}/C_\ell^{\rm kg}$) per unit $A_{\rm IA}$",
    )
    if cl_kg_ia is not None:
        ax.plot(
            bins,
            ia_obs_frac_fid,
            "^-",
            label=r"total $(C_\ell^{\rm obs}-C_\ell^{\rm kg})/C_\ell^{\rm kg}$ at fid",
        )
        ax.axhline(0.0, color="k", ls=":", lw=0.8)
    ax.set_xticks(bins)
    ax.set_xlabel("metacal z-bin")
    ax.set_ylabel("fraction")
    ax.set_title("Diagnostic 5: intrinsic-alignment amplitude scale")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "ia_scale.png"), dpi=150)
    plt.close(fig)


# diagnostic 6: per-component (kg / ia / sn) input vs. final Cl budget #################################################


def component_budget(conf, pixel_file, args, results):
    """Per-component Cl budget: the full-sky *input* spectrum of each lensing component against its
    spectrum after the full masking + KS mode-removal + smoothing chain that produces the final kappa
    data vector, for the lensing signal (kg), the intrinsic-alignment template (ia), and the shape
    noise (sn).

    Everything is put on a common full-sky-equivalent footing: the masked pseudo-Cls are divided by the
    patch sky fraction f_patch, so the trivial ~f_sky power suppression from cutting to one DES-area
    patch is removed and the residual input -> final gap is the genuine KS / mask / smoothing transfer
    (~0.80 for signal, ~0.89 for noise; sections 3.1-3.2). Reuses the spectra already computed by
    diagnostics 1-3; the only new work is pushing the ia template through postprocess_lensing (the same
    kappa -> gamma -> patch -> pseudo-E -> kappa chain used for kg in diagnostic 2 -- exact because KS is
    linear, so each component transfers independently in the auto-spectra).

     kg: full-sky Cl (diag 2) vs. masked-KS pseudo-Cl / f_patch (diag 2).
     ia: full-sky A_IA = 1 template Cl (diag 3) vs. the ia map through the lensing chain / f_patch (new).
     sn: flat white E-mode level (diag 1) / f_patch vs. empirical masked-KS noise (diag 1) / f_patch.
         There is no full-sky shape-noise map (the catalog only covers the footprint); the flat white
         level is the injected noise power assuming ~uniform depth, and dividing both by f_patch keeps it
         on the same full-sky-equivalent footing as kg and ia.
    """
    LOGGER.warning("Diagnostic 6: per-component (kg / ia / sn) input vs. final Cl budget")

    n_side = conf["analysis"]["n_side"]
    n_z = len(conf["survey"]["metacal"]["z_bins"])
    l = np.arange(3 * n_side)

    metacal_mask = files.get_tomo_dv_masks(conf)["metacal"]
    full_maps_file = filenames.get_filename_full_maps(
        args.fiducial_perm_dir, with_bary=conf["analysis"]["modelling"]["baryonified"]
    )

    f_patch = results["signal_f_patch"]

    # kg: reuse diagnostic 2 (same fiducial map: full-sky input and final masked pseudo-Cl)
    kg_input = results["signal_full_cls"]
    kg_final = results["signal_patch_cls"] / f_patch[:, None]

    # ia: full-sky A_IA = 1 template from diagnostic 3; final = ia map through the exact lensing chain (new)
    ia_input = results["ds_cl_ia"]
    ia_final = np.zeros((n_z, len(l)))
    for i_z, z_bin in enumerate(conf["survey"]["metacal"]["z_bins"]):
        ia_full = _read_full_sky_bin(conf, full_maps_file, "ia", z_bin)
        ia_dvs = postprocessing.postprocess_lensing(ia_full, conf, pixel_file, i_z)
        _, alm_ia_patch = _smooth_dv(ia_dvs[0], i_z, conf, pixel_file, metacal_mask[:, i_z])
        ia_final[i_z] = hp.alm2cl(alm_ia_patch) / f_patch[i_z]

    # sn: flat white E-level (diag 1) vs. empirical masked-KS noise (diag 1), both to full-sky footing
    sn_input = (results["noise_white_level"] / f_patch)[:, None] * np.ones(len(l))
    sn_final = np.mean(results["noise_emp_cls"], axis=1) / f_patch[:, None]

    sig_band = slice(100, 1000)
    noise_band = slice(300, 1500)
    for i_z in range(n_z):
        kg_t = np.median(kg_final[i_z][sig_band] / kg_input[i_z][sig_band])
        ia_t = np.median(ia_final[i_z][sig_band] / ia_input[i_z][sig_band])
        sn_t = np.median(sn_final[i_z][noise_band] / sn_input[i_z][noise_band])
        LOGGER.warning(
            f"metacal{i_z + 1}: final / input transfer (both / f_patch) -- "
            f"kg {kg_t:.3f} (ell in [100,1000]), ia {ia_t:.3f} (ell in [100,1000]), "
            f"sn {sn_t:.3f} (ell in [300,1500])"
        )

    results["budget_f_patch"] = f_patch
    results["budget_kg_input"] = kg_input
    results["budget_kg_final"] = kg_final
    results["budget_ia_input"] = ia_input
    results["budget_ia_final"] = ia_final
    results["budget_sn_input"] = sn_input
    results["budget_sn_final"] = sn_final

    comps = [
        ("kg", r"$\kappa_g$ (signal)", kg_input, kg_final, "C0"),
        ("ia", r"ia ($A_{\rm IA}{=}1$)", ia_input, ia_final, "C1"),
        ("sn", "sn (shape noise)", sn_input, sn_final, "C2"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for i_z, ax in enumerate(axes.flat):
        for _, label, c_in, c_fin, color in comps:
            ax.loglog(l[2:], c_in[i_z][2:], color=color, lw=1.6, label=f"{label} input")
            ax.loglog(l[2:], c_fin[i_z][2:], color=color, ls=":", lw=1.2, label=f"{label} final / $f_{{\\rm patch}}$")
        ax.set_title(f"metacal{i_z + 1}")
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$C_\ell$ (full-sky-equivalent)")
        if i_z == 0:
            ax.legend(fontsize=7, ncol=1)
    fig.suptitle(
        "Diagnostic 6: per-component input vs. final $C_\\ell$ "
        "(solid = full-sky input, dotted = masked-KS final / $f_{\\rm patch}$)"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "component_budget.png"), dpi=150)
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
    sn_ratio(conf, args, results)
    ia_scale(conf, args, results)
    component_budget(conf, pixel_file, args, results)

    out_file = os.path.join(args.out_dir, "wl_sn_diagnostics.npz")
    np.savez(out_file, **results)
    LOGGER.warning(f"Saved all results to {out_file}")


if __name__ == "__main__":
    main()
