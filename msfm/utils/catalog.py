import h5py
import numpy as np
import healpy as hp

from msfm.utils import logger, files

LOGGER = logger.get_logger(__file__)


def _get_rot_x(ang):
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(ang), -np.sin(ang)], [0.0, np.sin(ang), np.cos(ang)]]
    ).T  # Inverse because of healpy


def _get_rot_y(ang):
    return np.array(
        [[np.cos(ang), 0.0, np.sin(ang)], [0.0, 1.0, 0.0], [-np.sin(ang), 0.0, np.cos(ang)]]
    ).T  # Inverse because of healpy


def _get_rot_z(ang):
    return np.array(
        [[np.cos(ang), -np.sin(ang), 0.0], [np.sin(ang), np.cos(ang), 0.0], [0.0, 0.0, 1.0]]
    ).T  # Inverse because of healpy


def survey_angles_to_pix(conf, ra, dec, n_side):
    """Rotate to the position in Fig. 4 of https://arxiv.org/pdf/2511.04681"""

    conf = files.load_config(conf)

    # healpy convention in radian
    theta = -np.deg2rad(dec) + np.pi / 2
    phi = np.deg2rad(ra)
    vec = hp.ang2vec(theta=theta, phi=phi)

    # rotate footprint to allow for cut-outs
    # https://github.com/des-science/multiprobe-simulation-forward-model/blob/main/notebooks/pixel_file_catalog_level.ipynb
    y_rot = _get_rot_y(conf["analysis"]["footprint"]["rotation"]["y_rad"])
    z_rot = _get_rot_z(conf["analysis"]["footprint"]["rotation"]["z_rad"])
    vec = np.dot(np.dot(z_rot, y_rot), vec.T)

    # per-object pixel index
    pix = hp.vec2pix(n_side, vec[0], vec[1], vec[2])

    return pix


def build_metacal_map_from_cat(conf, debug=True):
    conf = files.load_config(conf)

    n_side = conf["analysis"]["n_side"]
    n_pix = conf["analysis"]["n_pix"]
    n_z = len(conf["survey"]["metacal"]["z_bins"])
    cat_dir = conf["dirs"]["catalog"]

    index = h5py.File(f"{cat_dir}/DESY3_indexcat.h5", "r")
    gold = h5py.File(f"{cat_dir}/DESY3_GOLD_2_2.1.h5", "r")
    metacal = h5py.File(f"{cat_dir}/DESY3_metacal_v03-004.h5", "r")
    dnf = h5py.File(f"{cat_dir}/DESY3_GOLD_2_2.1_DNF.h5", "r")

    wl_gamma_map = np.zeros((n_pix, n_z, 2))
    wl_count_map = np.zeros((n_pix, n_z), dtype=np.int32)
    for i in range(n_z):
        metacal_bin = index[f"/index/select_bin{i+1}"][:]

        # positions
        dec = gold["/catalog/gold/dec"][:][metacal_bin]
        ra = gold["/catalog/gold/ra"][:][metacal_bin]

        pix = survey_angles_to_pix(conf, ra, dec, n_side)
        count_map = np.bincount(pix, minlength=n_pix)

        # properties
        e1 = metacal["/catalog/unsheared/e_1"][:][metacal_bin]
        e2 = metacal["/catalog/unsheared/e_2"][:][metacal_bin]
        weight = metacal["/catalog/unsheared/weight"][:][metacal_bin]

        # weighted maps
        w_map = np.bincount(pix, weights=weight, minlength=n_pix)
        e1_map = np.bincount(pix, weights=e1 * weight, minlength=n_pix)
        e2_map = np.bincount(pix, weights=e2 * weight, minlength=n_pix)

        # normalize
        mask = w_map > 0
        e1_map[mask] /= w_map[mask]
        e2_map[mask] /= w_map[mask]

        wl_gamma_map[:, i, 0] = e1_map
        wl_gamma_map[:, i, 1] = e2_map
        wl_count_map[:, i] = count_map

        # compare with Table 1 of https://arxiv.org/pdf/2105.13543
        if debug:
            LOGGER.info(f"Metacalibration bin {i+1}")

            def w_mean(x):
                return np.sum(weight * x) / np.sum(weight)

            # eq. (12) in https://arxiv.org/pdf/2011.03408
            neff_deg = np.sum(weight) ** 2 / np.sum(weight**2) / conf["survey"]["Aeff"]
            neff_arcmin = neff_deg / 60**2
            LOGGER.info(f"N_gal = {len(metacal_bin)}, n_eff = {neff_arcmin:.3f} [arcmin^-2]")

            # eq. (13) in https://arxiv.org/pdf/2011.03408
            sigma_e_H12 = np.sqrt(0.5 * (np.sum((e1 * weight) ** 2) + np.sum((e2 * weight) ** 2)) / np.sum(weight**2))
            # approximately (missing sigma_m) eq. (10) in https://arxiv.org/pdf/2011.03408
            sigma_e_C13 = np.sqrt(0.5 * np.sum(weight**2 * (e1**2 + e2**2)) / np.sum(weight**2))
            LOGGER.info(f"sigma_e (H12) = {sigma_e_H12:.3f}, sigma_e (C13) = {sigma_e_C13:.3f}")

            LOGGER.info(f"mean(e1) = {w_mean(e1):.2e}, mean(e2) = {w_mean(e2):.2e}")

            z_mc = dnf["/catalog/unsheared/zmc_sof"][:][metacal_bin]
            LOGGER.info(f"z_mean = {w_mean(z_mc):.4f}")

    index.close()
    gold.close()
    metacal.close()
    dnf.close()

    return wl_gamma_map, wl_count_map


def build_maglim_map_from_cat(conf, debug=True):
    conf = files.load_config(conf)

    n_side = conf["analysis"]["n_side"]
    n_pix = conf["analysis"]["n_pix"]
    n_z = len(conf["survey"]["maglim"]["z_bins"])
    cat_dir = conf["dirs"]["catalog"]

    index = h5py.File(f"{cat_dir}/DESY3_indexcat.h5", "r")
    dnf = h5py.File(f"{cat_dir}/DESY3_GOLD_2_2.1_DNF.h5", "r")
    maglim = h5py.File(f"{cat_dir}/DESY3_maglim_redmagic_v0.5.1.h5", "r")

    maglim_index = index["index/maglim/select"][:]
    dec = maglim["catalog/maglim/dec"][:][maglim_index]
    ra = maglim["catalog/maglim/ra"][:][maglim_index]
    z = dnf["catalog/unsheared/zmean_sof"][:][maglim_index]

    gc_count_map = np.zeros((n_pix, n_z))
    for i in range(n_z):
        z_mask = (conf["survey"]["maglim"]["z_lims"][i][0] < z) & (z < conf["survey"]["maglim"]["z_lims"][i][1])

        # positions
        dec_bin = dec[z_mask]
        ra_bin = ra[z_mask]
        pix = survey_angles_to_pix(conf, ra_bin, dec_bin, n_side)

        gc_count_map[:, i] = np.bincount(pix, minlength=n_pix)

        # compare with Table 1 of https://arxiv.org/pdf/2105.13546
        if debug:
            LOGGER.info(f"Maglim bin {i+1}")

            N_gal = np.sum(z_mask)
            neff_deg = N_gal / conf["survey"]["Aeff"]
            neff_arcmin = neff_deg / 60**2
            LOGGER.info(f"N_gal = {N_gal}, n_eff = {neff_arcmin:.3f} [arcmin^-2]")

    index.close()
    dnf.close()
    maglim.close()

    return gc_count_map


def get_shapes_from_cat(conf):
    conf = files.load_config(conf)

    cat_dir = conf["dirs"]["catalog"]

    metacal = h5py.File(f"{cat_dir}/DESY3_metacal_v03-004.h5", "r")
    index = h5py.File(f"{cat_dir}/DESY3_indexcat.h5", "r")

    n_z = len(conf["survey"]["metacal"]["z_bins"])
    e_1 = []
    e_2 = []
    w = []
    for i in range(n_z):
        metacal_bin = index[f"/index/select_bin{i+1}"][:]
        LOGGER.info(f"Metacalibration bin {i+1}: N_gal = {len(metacal_bin)}")

        # properties
        e_1.append(metacal["/catalog/unsheared/e_1"][:][metacal_bin])
        e_2.append(metacal["/catalog/unsheared/e_2"][:][metacal_bin])
        w.append(metacal["/catalog/unsheared/weight"][:][metacal_bin])

    metacal.close()
    index.close()

    return e_1, e_2, w
