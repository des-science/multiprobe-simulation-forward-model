# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created December 2022
Author: Arne Thomsen

Convert the .h5 files containing the lensing and clustering maps to ordered .tfrecord files suitable for evaluation
of summary statistics.

Adapted from https://github.com/tomaszkacprzak/CosmoPointNet/blob/main/CosmoPointNet/apps/run_build_tfrecords.py 
by Tomasz Kacprzak
and 
https://cosmo-gitlab.phys.ethz.ch/jafluri/arne_handover/-/blob/main/map_projection/tfr_generation/fiducial.py
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/input_pipeline.py
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/data.py
by Janis Fluri
"""

import numpy as np
import tensorflow as tf
import os, argparse, warnings, h5py

from sobol_seq import i4_sobol

from msfm.utils import (
    files,
    lensing,
    logger,
    input_output,
    cosmogrid,
    tfrecords,
    parameters,
    filenames,
    redshift,
    clustering,
    scales,
)

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
LOGGER.info(f"Setting up healpy to run on {n_cpus} CPUs")

import healpy as hp


def resources(args):
    return dict(main_memory=1024, main_time=4, main_scratch=0, main_n_cores=8)


def setup(args):
    description = "Make .tfrecord files from the survey footprints collected in the .h5 files."
    parser = argparse.ArgumentParser(description=description, add_help=True)

    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="logging level",
    )
    parser.add_argument("--with_bary", action="store_true", help="include baryons")
    parser.add_argument("--n_files", type=int, default=100, help="number of .tfrecord files to produce")
    parser.add_argument(
        "--dir_in",
        type=str,
        default=b"/global/cfs/cdirs/des/cosmogrid/DESY3/grid",
        help="input root dir of the simulations",
    )
    parser.add_argument(
        "--dir_out",
        type=str,
        default="/pscratch/sd/a/athomsen/DESY3/grid",
        help="output root dir of the .tfrecords",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="configuration yaml file",
    )
    parser.add_argument(
        "--make_grf", action="store_true", help="Whether to degrade the maps to Gaussian random fields"
    )
    parser.add_argument(
        "--file_suffix",
        type=str,
        default="",
        help="Optional suffix to be appended to the end of the filename, for example to distinguish different runs",
    )
    parser.add_argument("--include_maglim_systematics", action="store_true", help="Whether to apply the")
    parser.add_argument("--debug", action="store_true", help="activate debug mode")

    args, _ = parser.parse_known_args(args)

    if not os.path.isdir(args.dir_out):
        input_output.robust_makedirs(args.dir_out)

    args.config = os.path.abspath(args.config)

    logger.set_all_loggers_level(args.verbosity)

    if args.include_maglim_systematics:
        LOGGER.debug(f"Including the Maglim systematics maps")

    if args.make_grf:
        LOGGER.warning(f"Degrading the maps to Gaussian Random Fields")

    return args


def main(indices, args):
    args = setup(args)

    LOGGER.timer.start("main")
    LOGGER.info(f"Got index set of size {len(indices)}")
    try:
        LOGGER.info(f"Running on {len(os.sched_getaffinity(0))} cores")
    except AttributeError:
        pass

    conf = files.load_config(args.config)

    # setup up directories
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    meta_info_file = os.path.join(repo_dir, conf["files"]["meta_info"])
    cosmo_params_info = cosmogrid.get_cosmo_params_info(meta_info_file, "grid")
    LOGGER.info(f"Loaded meta information")

    # constants
    n_side = conf["analysis"]["n_side"]
    sobol_priors = parameters.get_prior_intervals(
        conf["analysis"]["params"]["sobol"] + conf["analysis"]["params"]["ia"] + conf["analysis"]["params"]["bg"]
    )
    z0 = conf["analysis"]["systematics"]["z0"]

    # CosmoGrid
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_cosmo = conf["analysis"]["grid"]["n_perms_per_cosmo"]
    n_noise_per_example = conf["analysis"]["grid"]["n_noise_per_example"]
    n_examples_per_cosmo = n_patches * n_perms_per_cosmo * n_noise_per_example

    data_vec_pix, _, _, _ = files.load_pixel_file()

    # lensing (intrinsic alignment)
    tomo_z_metacal, tomo_nz_metacal = files.load_redshift_distributions("metacal", conf)
    m_bias_dist = lensing.get_m_bias_distribution(conf)

    def lensing_transform(kg, ia, Aia, n_Aia):
        # intrinsic alignment
        tomo_Aia = redshift.get_tomo_amplitudes(Aia, n_Aia, tomo_z_metacal, tomo_nz_metacal, z0)
        LOGGER.debug(f"Per z bin Aia = {tomo_Aia}")

        kg += tomo_Aia * ia

        # fixing this in the .tfrecords simplifies reproducibility
        m_bias = m_bias_dist.sample()
        kg *= 1.0 + m_bias

        return kg

    # clustering (linear galaxy bias)
    tomo_z_maglim, tomo_nz_maglim = files.load_redshift_distributions("maglim", conf)
    tomo_n_gal_maglim = np.array(conf["survey"]["maglim"]["n_gal"]) * hp.nside2pixarea(n_side, degrees=True)

    def clustering_smoothing(dg):
        # Gaussian Random Field
        if args.make_grf:
            smoothing_func = scales.data_vector_to_grf_data_vector
        # standard smoothing with a Gaussian kernel
        else:
            smoothing_func = scales.data_vector_to_smoothed_data_vector

        dg = smoothing_func(
            dg,
            l_min=conf["analysis"]["scale_cuts"]["clustering"]["l_min"],
            l_max=conf["analysis"]["scale_cuts"]["clustering"]["l_max"],
            n_side=conf["analysis"]["n_side"],
            data_vec_pix=data_vec_pix,
        )

        return dg

    def clustering_transform(dg, bg, n_bg):
        # linear galaxy biasing
        tomo_bg = redshift.get_tomo_amplitudes(bg, n_bg, tomo_z_maglim, tomo_nz_maglim, z0)
        LOGGER.debug(f"Per z bin bg = {tomo_bg}")

        galaxy_counts = clustering.galaxy_density_to_count(
            dg,
            tomo_n_gal_maglim,
            tomo_bg,
            conf=conf,
            include_systematics=conf["analysis"]["systematics"]["maglim_survey_systematics_map"],
        )

        # draw and smooth noise
        poisson_noises = clustering.galaxy_count_to_noise(galaxy_counts, n_noise_per_example)

        smooth_poisson_noises = []
        for poisson_noise in poisson_noises:
            smooth_poisson_noises.append(clustering_smoothing(poisson_noise))

        smooth_poisson_noises = np.stack(smooth_poisson_noises, axis=0)

        # noiseless
        galaxy_counts = clustering_smoothing(galaxy_counts)

        # shape (n_pix, n_z_maglim) and (n_noise_per_example, n_pix, n_z_maglim)
        return galaxy_counts, smooth_poisson_noises

    # set up the paths
    cosmo_dirs = [cosmo_dir.decode("utf-8") for cosmo_dir in cosmo_params_info["path_par"]]

    cosmo_dirs_in = [os.path.join(args.dir_in, cosmo_dir) for cosmo_dir in cosmo_dirs]
    n_cosmos = len(cosmo_dirs_in)
    LOGGER.info(f"Got simulation set grid of size {n_cosmos} with base path {args.dir_in}")

    # configure file structure
    if n_cosmos % args.n_files == 0:
        n_cosmos_per_file = n_cosmos // args.n_files
        LOGGER.info(f"The number of files implies {n_cosmos_per_file} cosmological parameters per file")
    else:
        raise ValueError(
            f"The total number of cosmologies {n_cosmos} has to be divisible by the number of files {args.n_files}"
        )
    n_examples_per_file = n_examples_per_cosmo * n_cosmos_per_file

    LOGGER.info(
        f"In total, there are n_examples_per_cosmo * n_cosmos_per_file = {n_examples_per_cosmo} * {n_cosmos_per_file}"
        f" = {n_examples_per_file} examples per file"
    )

    # index corresponds to a .tfrecord file ###########################################################################
    for index in indices:
        LOGGER.timer.start("index")

        tfr_file = filenames.get_filename_tfrecords(
            args.dir_out, tag=conf["survey"]["name"] + args.file_suffix, index=index, simset="grid"
        )
        LOGGER.info(f"Index {index} is writing to {tfr_file}")

        # index for the cosmological parameters
        js = index * n_cosmos_per_file
        je = (index + 1) * n_cosmos_per_file
        LOGGER.info(f"And includes {cosmo_dirs[js : je]}")

        n_done = 0
        with tf.io.TFRecordWriter(tfr_file) as file_writer:
            # loop over the cosmological parameters
            for i_cosmo, cosmo_dir_in in LOGGER.progressbar(
                zip(range(js, je), cosmo_dirs_in[js:je]),
                at_level="info",
                desc="Looping through cosmological parameters",
                total=je - js,
            ):
                LOGGER.debug(f"Taking inputs from {cosmo_dir_in}")

                if args.debug and n_done > 5:
                    LOGGER.warning("Debug mode, aborting after 5 subindices")
                    break

                # select the relevant cosmological parameters
                cosmo = [
                    cosmo_params_info[cosmo_param][i_cosmo] for cosmo_param in conf["analysis"]["params"]["cosmo"]
                ]
                cosmo = np.array(cosmo, dtype=np.float32)

                i_sobol = cosmo_params_info["sobol_index"][i_cosmo]

                # extend the Sobol sequence by astrophysical parameters
                sobol_point, _ = i4_sobol(sobol_priors.shape[0], i_sobol)
                sobol_params = sobol_point * np.squeeze(np.diff(sobol_priors)) + sobol_priors[:, 0]
                sobol_params = sobol_params.astype(np.float32)

                # add these to the label, the parameters are ordered as in sobol_priors
                Aia = sobol_params[6]
                n_Aia = sobol_params[7]
                bg = sobol_params[8]
                n_bg = sobol_params[9]
                cosmo = np.concatenate((cosmo, np.array([Aia, n_Aia, bg, n_bg])))

                # redshift evolution, only calculate the integrals once here
                current_lensing_transform = lambda kg, ia: lensing_transform(kg, ia, Aia, n_Aia)
                current_clustering_transform = lambda dg: clustering_transform(dg, bg, n_bg)

                # verify that the Sobol sequences are identical (the parameters are ordered differently)
                assert np.allclose(sobol_params[0], cosmo[0], rtol=1e-3, atol=1e-4)  # Om
                assert np.allclose(sobol_params[1], cosmo[1], rtol=1e-3, atol=1e-4)  # s8
                assert np.allclose(sobol_params[2], cosmo[3], rtol=1e-3, atol=1e-4)  # Ob
                assert np.allclose(sobol_params[3], cosmo[2], rtol=1e-3, atol=1e-4)  # H0
                assert np.allclose(sobol_params[4], cosmo[4], rtol=1e-3, atol=1e-4)  # ns
                assert np.allclose(sobol_params[5], cosmo[5], rtol=1e-3, atol=1e-4)  # w0

                # load the .h5 files
                file_cosmo = filenames.get_filename_data_vectors(cosmo_dir_in, with_bary=args.with_bary)

                kg_examples, ia_examples, sn_examples, dg_examples = load_data_vecs(file_cosmo)

                # loop over the n_examples_per_cosmo
                for i_example, (kg, ia, sn_realz, dg) in LOGGER.progressbar(
                    enumerate(zip(kg_examples, ia_examples, sn_examples, dg_examples)),
                    at_level="debug",
                    desc="Looping through the examples of one cosmology",
                    total=n_examples_per_cosmo,
                ):
                    kg = current_lensing_transform(kg, ia)
                    dg, pn_realz = current_clustering_transform(dg)

                    serialized = tfrecords.parse_forward_grid(
                        kg, sn_realz, dg, pn_realz, cosmo, i_sobol, i_example
                    ).SerializeToString()

                    # check correctness
                    inv_data_vectors = tfrecords.parse_inverse_grid(serialized, n_noise_per_example)

                    for i_noise in range(n_noise_per_example):
                        assert np.allclose(inv_data_vectors[f"kg_{i_noise}"], kg + sn_realz[i_noise])
                        assert np.allclose(inv_data_vectors[f"dg_{i_noise}"], dg + pn_realz[i_noise])
                    assert np.allclose(inv_data_vectors["cosmo"], cosmo)
                    assert np.allclose(inv_data_vectors["i_sobol"], i_sobol)
                    assert np.allclose(inv_data_vectors["i_example"], i_example)

                    LOGGER.debug("decoded successfully")

                    file_writer.write(serialized)

                n_done += 1

        LOGGER.info(f"Done with index {index} after {LOGGER.timer.elapsed('index')}")
        yield index


def load_data_vecs(
    filename,
):
    with h5py.File(filename, "r") as f:
        # shape (n_examples_per_cosmo, n_pix, n_z_bins) before the indexing
        kg = f["kg"][:]
        ia = f["ia"][:]
        sn_realz = f["sn"][:]
        dg = f["dg"][:]

    LOGGER.debug(f"Successfully loaded the data vectors")
    return kg, ia, sn_realz, dg
