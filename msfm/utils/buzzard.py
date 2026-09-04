import h5py
import numpy as np
import healpy as hp

from msfm.utils import files, parameters


def get_cosmo(conf=None, params=None):
    """The Buzzard truth cosmology, as a dict over the parameters of conf (or of params).

    NaN marks a parameter the simulation has no single value for, rather than a value of zero:
    the flock is dark-matter-only (no baryonification), and its galaxies are populated with
    ADDGALS rather than a linear bias, so no bg_i exists. A KeyError here means a config asks
    for a parameter this table has not been extended to.
    """
    conf = files.load_config(conf)
    params = parameters.get_parameters(params, conf)

    # Buzzard v2.0, i.e. the Chinchilla N-body cosmology of https://arxiv.org/pdf/1901.02401
    buzzard_cosmo = {
        "Om": 0.286,
        "s8": 0.82,
        "Ob": 0.047,
        "H0": 70.0,
        "ns": 0.96,
        "w0": -1,
        "bary_Mc": np.nan,
        "bary_nu": np.nan,
        "Aia": 0.0,
        "n_Aia": np.nan,
        "bta": 0.0,
        "bg1": np.nan,
        "bg2": np.nan,
        "bg3": np.nan,
        "bg4": np.nan,
    }

    return {param: buzzard_cosmo[param] for param in params}


def get_lensing_map(lensing_file, nest_in=False, plot_diagnostics=False):
    with h5py.File(lensing_file, "r") as f_in:
        gamma1 = []
        gamma2 = []
        for j in range(1, 5):
            gamma1.append(f_in[f"metacal/raw_gamma1_bin{j}"])
            gamma2.append(f_in[f"metacal/raw_gamma2_bin{j}"])
        gamma1 = np.stack(gamma1, axis=-1)
        gamma2 = np.stack(gamma2, axis=-1)

        wl_gamma_map = np.stack([gamma1, gamma2], axis=-1)

    if plot_diagnostics:
        hp.mollview(gamma1[:, 0], nest=nest_in, title="Buzzard gamma1")
        hp.mollview(gamma2[:, 0], nest=nest_in, title="Buzzard gamma2")

    return wl_gamma_map


def get_metacal_counts(lensing_file):
    """Read the metacal (source) galaxy count map from a raw Buzzard lensing file, full sky.

    This is the true N-body source clustering, used as the expected-count map for the "count"
    shape-noise model (see observation.make_shape_noise_map).
    """
    with h5py.File(lensing_file, "r") as f_in:
        counts = [f_in[f"metacal/galaxy_counts_bin{j}"][:] for j in range(1, 5)]

    return np.stack(counts, axis=-1)


def get_clustering_map(clustering_file, nest_in=False, plot_diagnostics=False):
    with h5py.File(clustering_file, "r") as f_in:
        gc_count_map = []
        for i in range(1, 5):
            gc_count_map.append(f_in[f"maglim/galaxy_counts_bin{i}"][:])
        gc_count_map = np.stack(gc_count_map, axis=-1)

    if plot_diagnostics:
        hp.mollview(gc_count_map[:, 0], nest=nest_in, title="Buzzard galaxy counts")

    return gc_count_map
