# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created September 2022
Author: Arne Thomsen

Transform the full sky weak lensing signal and intrinsic alignment maps into multiple survey footprint cut-outs,
both for the fiducial and the grid cosmology

Meant for Euler (CPU nodes, local scratch)
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os, argparse, warnings, h5py, time, logging

from numba import njit
from icecream import ic

from msfm.utils import logger, input_output, shear, cosmogrid, survey
from msfm.utils.filenames import *

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)

# set the environmental variable OMP_NUM_THREADS to the number of logical processors for healpy parallelixation
try:
    n_cpus = len(os.sched_getaffinity(0))
except AttributeError:
    LOGGER.debug(f"os.sched_getaffinity is not available on this system, use os.cpu_count() instead")
    n_cpus = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(n_cpus)

import healpy as hp

hp_LOGGER = logging.getLogger("healpy")
hp_LOGGER.disabled = True
# hp_LOGGER.setLevel(logging.ERROR)
# warnings.filterwarnings("once", module="healpy")
# hp.disable_warnings()


def resources(args):
    return dict(main_memory=1000, main_time=4, main_scratch=0, main_n_cores=8)


def setup(args):
    description = "Make weak lensing and intrinsic alignment datavectors from full sky maps"
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="logging level",
    )
    parser.add_argument(
        "--simset", type=str, default="grid", choices=("grid", "fiducial"), help="set of simulations to use"
    )
    parser.add_argument("--with_bary", action="store_true", help="activate debug mode")
    parser.add_argument(
        "--dir_in",
        type=str,
        default="/global/cfs/cdirs/des/cosmogrid/DESY3/grid",
        help="input root dir of the simulations",
    )
    parser.add_argument(
        "--dir_out",
        type=str,
        default="/pscratch/sd/a/athomsen/DESY3/grid",
        help="output root dir of the simulations",
    )
    parser.add_argument(
        "--repo_dir",
        type=str,
        default="/global/homes/a/athomsen/multiprobe-simulation-forward-model",
        help="root dir of the msfm repo to convert relative paths to absolute ones",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="configuration yaml file",
    )
    parser.add_argument("--debug", action="store_true", help="activate debug mode")
    parser.add_argument(
        "--max_sleep",
        type=int,
        default=120,
        help="set the maximal amount of time to sleep before copying to avoid clashes",
    )
    parser.add_argument("--store_counts", action="store_true", help="whether to store the metacal galaxy count maps")

    args, _ = parser.parse_known_args(args)

    logger.set_all_loggers_level(args.verbosity)

    args.repo_dir = os.path.abspath(args.repo_dir)

    return args


def main(indices, args):
    args = setup(args)

    LOGGER.timer.start("main")
    LOGGER.info(f"Got index set of size {len(indices)}")
    try:
        LOGGER.info(f"Running on {len(os.sched_getaffinity(0))} cores")
    except AttributeError:
        pass

    if args.debug:
        args.max_sleep = 0
        LOGGER.warning("!!! debug mode !!!")

    sleep_sec = np.random.uniform(0, args.max_sleep) if args.max_sleep > 0 else 0
    LOGGER.info(f"Waiting for {sleep_sec:.2f}s to prevent overloading IO")
    time.sleep(sleep_sec)

    conf_file = os.path.join(args.repo_dir, args.config)
    conf = input_output.read_yaml(conf_file)
    LOGGER.info(f"Loaded configuration file")

    # constants
    n_side = conf["analysis"]["n_side"]
    n_pix = conf["analysis"]["n_pix"]
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_cosmo = conf["analysis"][args.simset]["n_perms_per_cosmo"]
    n_noise_per_example = conf["analysis"][args.simset]["n_noise_per_example"]
    LOGGER.info(f"Looping through {n_perms_per_cosmo} permutations per cosmological parameter set")
    LOGGER.info(f"Generating {n_noise_per_example} noise realizations per example")

    lmax = 3 * n_side - 1
    l = hp.Alm.getlm(lmax)[0]
    l[l == 1] = 0
    kappa2gamma_fac, gamma2kappa_fac = shear.get_kaiser_squires_factors(l)
    l_mask_fac = shear.get_l_mask(l)

    # pixel file
    data_vec_pix, patches_pix, gamma2_signs, tomo_patches_pix, tomo_corresponding_pix = survey.load_pixel_file(
        conf, args.repo_dir
    )
    data_vec_len = len(data_vec_pix)

    # noise file
    tomo_gamma_cat, tomo_n_bar = survey.load_noise_file(conf, args.repo_dir)

    # set up the directories
    meta_info_file = os.path.join(args.repo_dir, conf["files"]["meta_info"])
    cosmo_params_info = cosmogrid.get_cosmo_params_info(meta_info_file, args.simset)
    cosmo_dirs = [cosmo_dir.decode("utf-8") for cosmo_dir in cosmo_params_info["path_par"]]

    # remove baryon perturbations for the fiducial set
    if args.simset == "fiducial" and not args.with_bary:
        cosmo_dirs = [cosmo_dir for cosmo_dir in cosmo_dirs if not "bary" in cosmo_dir]

    cosmo_dirs_in = [os.path.join(args.dir_in, cosmo_dir) for cosmo_dir in cosmo_dirs]
    cosmo_dirs_out = [os.path.join(args.dir_out, cosmo_dir) for cosmo_dir in cosmo_dirs]

    n_cosmos = len(cosmo_dirs_in)
    LOGGER.info(f"Got simulation set {args.simset} of size {n_cosmos} with base path {args.dir_in}")

    # other directories
    hp_datapath = os.path.join(args.repo_dir, conf["files"]["healpy_data"])

    # index corresponds to a cosmological parameter (either on the grid or for the fiducial perturbations) ############
    for index in indices:
        LOGGER.timer.start("index")

        cosmo_dir_in = cosmo_dirs_in[index]
        cosmo_dir_out = cosmo_dirs_out[index]
        if not os.path.isdir(cosmo_dir_out):
            input_output.robust_makedirs(cosmo_dir_out)
        LOGGER.info(f"Index {index} takes input from {cosmo_dir_in}")

        # the perturbations of the fiducial cosmology are treated differently
        if args.simset == "grid" or (args.simset == "fiducial" and "cosmo_fiducial" in cosmo_dir_in):
            is_perturbation = False
            LOGGER.info(f"This is not a perturbation, running all map types")
        else:
            is_perturbation = True
            LOGGER.info(f"This is a perturbation, intrinsic alignment and shape noise are skipped")

        for i_perm in LOGGER.progressbar(range(n_perms_per_cosmo), desc="Loop over permutations\n", at_level="info"):
            LOGGER.timer.start("permutation")
            LOGGER.info(f"Starting simulation permutation {i_perm:04d}")

            # TODO copy the file to local scratch first?
            perm_dir_in = os.path.join(cosmo_dir_in, f"perm_{i_perm:04d}")
            full_maps_file = get_filename_full_maps(perm_dir_in, with_bary=args.with_bary)

            # output containers, one for each permutation, in NEST ordering with padding
            data_vectors = {}

            for map_type_in in conf["survey"]["map_types"]["input"]:
                LOGGER.info(f"Starting with input map type {map_type_in}")
                LOGGER.timer.start("map_type")

                # lensing, metacal sample #############################################################################
                z_bins = conf["survey"]["metacal"]["z_bins"]
                n_z_bins = len(z_bins)
                # FIXME correct metacal bias
                tomo_bias = conf["survey"]["metacal"]["bias"]

                # TODO do every patch multiple times like in KiDS1000 with the redshift errors

                # always do the convergence
                if map_type_in == "kg":
                    map_type_out = map_type_in
                    dvs_shape = (n_patches, data_vec_len, n_z_bins)

                # don't do intrinsic alignment for the fiducial perturbations
                elif map_type_in == "ia" and not is_perturbation:
                    map_type_out = map_type_in
                    dvs_shape = (n_patches, data_vec_len, n_z_bins)

                # don't do shape noise for the fiducial perturbations
                elif map_type_in == "dg" and not is_perturbation:
                    map_type_out = "sn"
                    dvs_shape = (n_patches, n_noise_per_example, data_vec_len, n_z_bins)

                    if args.store_counts:
                        data_vectors["ct"] = np.zeros((n_patches, data_vec_len, n_z_bins), dtype=np.float32)

                else:
                    map_type_out = None
                    LOGGER.info(f"This combination of cosmology and input map type is skipped")
                    continue

                if map_type_out is not None:
                    data_vectors[map_type_out] = np.zeros(dvs_shape, dtype=np.float32)

                for i_z, z_bin in enumerate(z_bins):
                    # only consider this tomographic bin
                    patches_pix = tomo_patches_pix[i_z]
                    corresponding_pix = tomo_corresponding_pix[i_z]

                    # base (rotated) footprint
                    base_patch_pix = patches_pix[0]

                    # load the full sky maps for the metacal redshift bins
                    map_dir = f"{map_type_in}/{z_bin}"
                    with h5py.File(full_maps_file, "r") as f:
                        map_full = f[map_dir][:]
                    LOGGER.debug(f"Loaded {map_dir} from {full_maps_file}")

                    # don't save intrinsic alignment maps for the perturbations of the fiducial
                    if (map_type_out in ["kg", "ia"]) and (map_type_out is not None):
                        kappa_full = map_full

                        # kappa -> gamma (full sky)
                        kappa_alm = hp.map2alm(kappa_full, lmax=lmax, use_pixel_weights=True, datapath=hp_datapath)
                        gamma_alm = kappa_alm * kappa2gamma_fac
                        _, gamma1_full, gamma2_full = hp.alm2map(
                            [np.zeros_like(gamma_alm), gamma_alm, np.zeros_like(gamma_alm)], nside=n_side
                        )

                        for i_patch, patch_pix in enumerate(patches_pix):
                            LOGGER.debug(f"Starting with patch index {i_patch}")

                            # The 90° rots do NOT change the shear, however, the mirroring does,
                            # therefore we have to swap sign of gamma2 for the last 2 patches!
                            gamma2_sign = gamma2_signs[i_patch]
                            LOGGER.debug(f"Using gamma2 sign {gamma2_sign}")

                            # TODO each patch is done multiple times

                            # masking TODO use Janis' memory efficient numba function?
                            gamma1_patch = np.zeros(n_pix, dtype=np.float32)
                            gamma1_patch[base_patch_pix] = gamma1_full[patch_pix]

                            gamma2_patch = np.zeros(n_pix, dtype=np.float32)
                            gamma2_patch[base_patch_pix] = gamma2_full[patch_pix]

                            # fix the sign
                            gamma2_patch *= gamma2_sign

                            # kappa patch is a full sky map, but only the patch is occupied
                            kappa_patch = mode_removal(
                                gamma1_patch, gamma2_patch, gamma2kappa_fac, l_mask_fac, n_side, hp_datapath
                            )

                            # remove mean within the patch (not over the full sky)
                            kappa_patch -= np.mean(kappa_patch[base_patch_pix])

                            # cut out padded data vector
                            kappa_dv = get_data_vec(kappa_patch, data_vec_len, corresponding_pix, base_patch_pix)

                            data_vectors[map_type_out][i_patch, :, i_z] = kappa_dv

                    # don't generate noise maps for the perturbations of the fiducial
                    elif (map_type_out == "sn") and (map_type_out is not None):
                        delta_full = map_full

                        # only consider this tomographic bin
                        n_bar = tomo_n_bar[i_z]
                        bias = tomo_bias[i_z]
                        gamma_cat = tomo_gamma_cat[i_z]
                        n_gals_cat = gamma_cat.shape[0]
                        gamma_abs = tf.math.abs(gamma_cat[:, 0] + 1j * gamma_cat[:, 1])

                        # normalize to number density contrast
                        delta_full = (delta_full - np.mean(delta_full)) / np.mean(delta_full)

                        # number of galaxies per pixel
                        counts_full = n_bar * (1 + bias * delta_full)
                        counts_full = np.where(0 < counts_full, counts_full, 0)
                        counts_full = np.random.poisson(counts_full)

                        for i_patch, patch_pix in enumerate(patches_pix):
                            LOGGER.debug(f"Starting with patch index {i_patch}")
                            LOGGER.timer.start("noise_patch")

                            # not a full healpy map, just the patch with no zeros
                            counts_patch = counts_full[patch_pix]
                            n_gals_patch = np.sum(counts_patch)

                            # TODO Could like https://www.tensorflow.org/guide/function#accumulating_values_in_a_loop
                            # and could probably include most of what's underneath in one large tf.function

                            # indices to sum over all of the galaxies in the individual pixels
                            seg_ids = []
                            for id, n_gals in enumerate(counts_patch):
                                seg_ids.extend(n_gals * [id])

                            # inputs to the tf.function have to be tensors
                            seg_ids = tf.constant(seg_ids, dtype=tf.int32)

                            for i_noise in range(n_noise_per_example):
                                LOGGER.debug(f"Starting with noise realization {i_noise}")
                                LOGGER.timer.start("noise_realization")

                                # randomize TODO set random seed on operator level?
                                phase = tf.random.uniform(shape=(n_gals_cat,), minval=0, maxval=2 * np.pi)
                                gamma1 = tf.math.cos(phase) * gamma_abs
                                gamma2 = tf.math.sin(phase) * gamma_abs

                                # TODO sample within the tf.function? Could be a bit faster
                                # joint samples for e1, e2 and w, this is faster than random indexing
                                emp_dist = tfp.distributions.Empirical(
                                    samples=tf.stack([gamma1, gamma2, gamma_cat[:, 2]], axis=1), event_ndims=1
                                )

                                samples = emp_dist.sample(sample_shape=n_gals_patch)

                                gamma1, gamma2 = tf_noise_gen(samples, seg_ids)

                                # The condition means that the final pixel contains zero galaxies. Then, its index is 
                                # not included in the seg_ids (multiplication with zero) and because it's the last, 
                                # tensorflow has no way of knowing that it should still take the segmented_sum over 
                                # this index, which evaluates to zero. The while loop allows more than one of the last
                                # pixels to be zero.
                                n_zero_pix = 0
                                while counts_patch[-(n_zero_pix+1)] == 0:
                                    n_zero_pix += 1

                                if n_zero_pix > 0:
                                    # There is no galaxy in the final pixels, so the shape noise there is equal to zero
                                    gamma1 = np.concatenate((gamma1, np.zeros(n_zero_pix)))
                                    gamma2 = np.concatenate((gamma2, np.zeros(n_zero_pix)))

                                gamma1_patch = np.zeros(n_pix, dtype=np.float32)
                                gamma1_patch[base_patch_pix] = gamma1

                                gamma2_patch = np.zeros(n_pix, dtype=np.float32)
                                gamma2_patch[base_patch_pix] = gamma2

                                kappa_patch = mode_removal(
                                    gamma1_patch, gamma2_patch, gamma2kappa_fac, l_mask_fac, n_side, hp_datapath
                                )

                                # remove mean within the patch (not over the full sky)
                                kappa_patch -= np.mean(kappa_patch[base_patch_pix])

                                # cut out padded data vector
                                kappa_dv = get_data_vec(kappa_patch, data_vec_len, corresponding_pix, base_patch_pix)

                                data_vectors[map_type_out][i_patch, i_noise, :, i_z] = kappa_dv

                                LOGGER.debug(
                                    f"Done with noise realization {i_noise} after {LOGGER.timer.elapsed('noise_realization')}"
                                )

                            if args.store_counts:
                                # correct cut out procedure involves a full sky map
                                counts_patch_map = np.zeros(n_pix, dtype=np.float32)
                                counts_patch_map[base_patch_pix] = counts_patch
                                counts_dv = get_data_vec(
                                    counts_patch_map, data_vec_len, corresponding_pix, base_patch_pix
                                )
                                data_vectors["ct"][i_patch, :, i_z] = counts_dv

                            LOGGER.debug(
                                f"Done with noise patch {i_patch} after {LOGGER.timer.elapsed('noise_patch')}"
                            )

                # TODO
                # clustering, maglim sample ###########################################################################
                # galaxy_sample = maglim
                # z_bins = conf["survey"]["maglim"]["z_bins"]
                # n_z_bins = len(z_bins)

                # if map_type_in in conf["survey"]["map_types"]["lensing"]:
                #     map_type_out = map_type_in
                # elif map_type_in in conf["survey"]["map_types"]["clustering"]:
                #     map_type_out = "sn"
                #     if args.store_counts:
                #         data_vectors["ct"] = np.zeros((n_patches, data_vec_len, n_z_bins), dtype=np.float32)
                #         if args.store_patches:
                #             data_patches["ct"] = np.zeros((n_patches, patches_len, n_z_bins), dtype=np.float32)

                # # TODO do every patch multiple times
                # data_vectors[map_type_out] = np.zeros((n_patches, data_vec_len, n_z_bins), dtype=np.float32)
                # if args.store_patches:
                #     data_patches[map_type_out] = np.zeros((n_patches, patches_len, n_z_bins), dtype=np.float32)

                # for i_z, z_bin in enumerate(z_bins):
                #     pass

                LOGGER.info(f"Done with map type {map_type_out} after {LOGGER.timer.elapsed('map_type')}")

            # save the results
            data_vec_file = get_filename_data_vectors(cosmo_dir_out, args.with_bary)
            save_output_container(
                "datavectors",
                data_vec_file,
                data_vectors,
                i_perm,
                n_perms_per_cosmo,
                n_patches,
                n_noise_per_example,
                data_vec_len,
                n_z_bins,
            )

            LOGGER.info(f"Done with permutation {i_perm:04d} after {LOGGER.timer.elapsed('permutation')}")

        LOGGER.info(f"Done with index {index} after {LOGGER.timer.elapsed('index')}")
        yield index


def mode_removal(gamma1_patch, gamma2_patch, gamma2kappa_fac, l_mask_fac, n_side, hp_datapath=None):
    """Takes in survey patches of gamma maps and puts out survey patches of kappa maps that only contain E-modes

    Args:
        gamma1_patch (np.ndarray): Array of size n_pix, but only the survey patch is populated
        gamma2_patch (np.ndarray): Same
        gamma2kappa_fac (np.ndarray): Kaiser squires conversion factors
        l_mask_fac (np.ndarray): Mask l = 0,1
        n_side (int): Resolution of the map
        hp_datapath (str, optional): Path to a healpy pixel weights file. Defaults to None.

    Returns:
        np.ndarray: Array of size n_pix, but only the survey patch is populated
    """
    _, gamma_alm_E, gamma_alm_B = hp.map2alm(
        [np.zeros_like(gamma1_patch), gamma1_patch, gamma2_patch],
        use_pixel_weights=True,
        datapath=hp_datapath,
    )
    kappa_alm = gamma_alm_E * gamma2kappa_fac
    LOGGER.debug(f"Mode removal successfull")

    kappa_alm *= l_mask_fac

    kappa_patch = hp.alm2map(kappa_alm, nside=n_side)

    return kappa_patch


@njit
def get_data_vec(m, data_vec_len, corresponding_pix, cutout_pix):
    """
    This function makes cutouts from full sky maps to a nice data vector that can be fed into a DeepSphere network

    Args:
        m (ndarray): The map one should make a cutout from
        data_vec_len (int): length of the full data vec (including padding)
        corresponding_pix (ndarray): pixel inside the data vec that should be populated (excludes padding)
        cutout_pix (ndarray): pixel that should be cut out from the map (excludes padding)

    Returns:
        ndarray: the data vec
    """
    data_vec = np.zeros(data_vec_len, dtype=np.float32)
    n_pix = corresponding_pix.shape[0]

    assert corresponding_pix.shape[0] == cutout_pix.shape[0]

    # assign
    for i in range(n_pix):
        data_vec[corresponding_pix[i]] = m[cutout_pix[i]]

    return data_vec


# the input tensors have variable length because of the varying number of galaxies in the count map
@tf.function(
    # jit_compile=True,
    input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.int32)],
)
def tf_noise_gen(samples, seg_ids):
    LOGGER.warning(
        f"Tracing the noise generator tf.function for samples.shape = {samples.shape} and "
        f"seg_ids.shape = {seg_ids.shape}"
    )

    # shape (total_gals, 3)
    e_samples = samples[:, :2]
    w_samples = tf.expand_dims(samples[:, 2], axis=1)

    # apply weights
    samples = tf.concat([e_samples * w_samples, w_samples], axis=1)

    # same shape as base_patch_pix and counts_patch, unless the final pixel of the patch doesn't contain a galaxy.
    # Then, it's one element smaller
    sum_per_pix = tf.math.segment_sum(samples, seg_ids)

    # normalize with weights, set 0/0 equal to 0 instead of nan
    e_per_pix = tf.math.divide_no_nan(sum_per_pix[:, :2], tf.expand_dims(sum_per_pix[:, 2], axis=1))

    return e_per_pix[:, 0], e_per_pix[:, 1]


def save_output_container(
    label, filename, output_container, i_perm, n_perms_per_cosmo, n_patches, n_noise_per_example, output_len, n_z_bins
):
    """Saves an .h5 file collecting all results on the level of the cosmological parameters (so for different
    permutations/runs and patches)

    Args:
        label (str): either "datavectors" or "patches"
        filename (str): path to the output .h5 file
        output_container (dict): Dictionary of arrays of shape (n_patches, output_len, n_z_bins)
        i_perm (int): Index of the permutation
        n_perms_per_cosmo (int):
        n_patches (int):
        output_len (int):
        n_z_bins (int):
    """
    with h5py.File(filename, "a") as f:
        for map_type in output_container.keys():
            if map_type == "sn":
                out_shape = (n_perms_per_cosmo * n_patches, n_noise_per_example, output_len, n_z_bins)
            else:
                out_shape = (n_perms_per_cosmo * n_patches, output_len, n_z_bins)

            try:
                # create dataset for every parameter level directory, collecting the permutation levels
                f.create_dataset(name=map_type, shape=out_shape)
            except ValueError:
                LOGGER.debug(f"dataset {map_type} already exists in {filename}")

            f[map_type][n_patches * i_perm : n_patches * (i_perm + 1)] = output_container[map_type]

    LOGGER.info(f"Stored {label} in {filename}")


# This main only exists for testing purposes when not using esub
if __name__ == "__main__":
    args = [
        "--simset=grid",
        "--dir_in=/Users/arne/data/CosmoGrid_example/DES/grid",
        "--dir_out=/Users/arne/data/CosmoGrid_example/DES/grid/v1",
        "--repo_dir=/Users/arne/git/multiprobe-simulation-forward-model",
        "--config=configs/config.yaml",
        "--max_sleep=0",
        "--debug",
        "--verbosity=debug",
        "--store_counts",
    ]

    indices = [0]
    for _ in main(indices, args):
        pass

    LOGGER.info(f"Done with main after {LOGGER.timer.elapsed('main')}")
