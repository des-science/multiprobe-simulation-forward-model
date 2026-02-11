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


def _survey_angles_to_pix(ra, dec, n_side):
    """Rotate to the position in Fig. 4 of https://arxiv.org/pdf/2511.04681"""

    # healpy convention in radian
    theta = -np.deg2rad(dec) + np.pi / 2
    phi = np.deg2rad(ra)

    # rotate to new position, hardcoded like in notebook
    # https://github.com/des-science/multiprobe-simulation-forward-model/blob/main/notebooks/pixel_file_catalog_level.ipynb
    y_rot = _get_rot_y(-0.125)
    z_rot = _get_rot_z(-1.22)

    vec = hp.ang2vec(theta=theta, phi=phi)
    vec = np.dot(np.dot(z_rot, y_rot), vec.T)

    # per-object pixel index
    pix = hp.vec2pix(n_side, vec[0], vec[1], vec[2])

    return pix


def build_metacal_map_from_cat(conf):
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
    for i in LOGGER.progressbar(range(n_z)):
        metacal_bin = index[f"/index/select_bin{i+1}"][:]
        LOGGER.info(f"#gals in metacal bin {i+1}: {len(metacal_bin)}")

        # positions
        dec = gold["/catalog/gold/dec"][:][metacal_bin]
        ra = gold["/catalog/gold/ra"][:][metacal_bin]
        pix = _survey_angles_to_pix(ra, dec, n_side)

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

    index.close()
    gold.close()
    metacal.close()
    dnf.close()

    return wl_gamma_map


def build_maglim_map_from_cat(conf):
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
    for i in LOGGER.progressbar(range(n_z)):
        z_mask = (conf["survey"]["maglim"]["z_lims"][i][0] < z) & (z < conf["survey"]["maglim"]["z_lims"][i][1])

        # positions
        dec_bin = dec[z_mask]
        ra_bin = ra[z_mask]
        pix = _survey_angles_to_pix(ra_bin, dec_bin, n_side)

        gc_count_map[:, i] = np.bincount(pix, minlength=n_pix)

    index.close()
    dnf.close()
    maglim.close()

    return gc_count_map
