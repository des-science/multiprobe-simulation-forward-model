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
import healpy as hp
import tensorflow as tf
import os, argparse, warnings, h5py

from sobol_seq import i4_sobol

from msfm.utils import logger, input_output, cosmogrid, tfrecords, analysis, parameters, shear, filenames, redshift

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def resources(args):
    return dict(main_memory=8192, main_time=4, main_scratch=0, main_n_cores=1)


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
    parser.add_argument("--np_seed", type=int, default=7, help="random seed to shuffle the patches")
    parser.add_argument("--debug", action="store_true", help="activate debug mode")

    args, _ = parser.parse_known_args(args)

    if not os.path.isdir(args.dir_out):
        input_output.robust_makedirs(args.dir_out)

    args.config = os.path.abspath(args.config)

    logger.set_all_loggers_level(args.verbosity)

    return args


def main(indices, args):
    args = setup(args)

    LOGGER.timer.start("main")
    LOGGER.info(f"Got index set of size {len(indices)}")
    try:
        LOGGER.info(f"Running on {len(os.sched_getaffinity(0))} cores")
    except AttributeError:
        pass

    conf = analysis.load_config(args.config)

    # setup up directories
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    meta_info_file = os.path.join(repo_dir, conf["files"]["meta_info"])
    cosmo_params_info = cosmogrid.get_cosmo_params_info(meta_info_file, "grid")
    LOGGER.info(f"Loaded meta information")

    # constants
    n_side = conf["analysis"]["n_side"]
    tomo_n_gal_maglim = np.array(conf["survey"]["maglim"]["n_gal"]) * hp.nside2pixarea(n_side, degrees=True)
    sobol_priors = parameters.get_prior_intervals(conf["analysis"]["grid"]["params_sobol"])

    # CosmoGrid
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_cosmo = conf["analysis"]["grid"]["n_perms_per_cosmo"]
    n_noise_per_example = conf["analysis"]["grid"]["n_noise_per_example"]
    n_examples_per_cosmo = n_patches * n_perms_per_cosmo * n_noise_per_example

    # shear bias distribution, NOTE fixing this in the .tfrecords simplifies reproducibility
    m_bias_dist = shear.get_m_bias_distribution(conf)

    # redshift evolution
    z0 = conf["analysis"]["z0"]
    tomo_z_metacal, tomo_nz_metacal = analysis.load_redshift_distributions("metacal", conf)
    tomo_z_maglim, tomo_nz_maglim = analysis.load_redshift_distributions("maglim", conf)

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
            args.dir_out, tag=conf["survey"]["name"], index=index, simset="grid"
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

                # select the relevant cosmological parameters, the first six are part of the CosmoGrid
                cosmo = [cosmo_params_info[cosmo_param][i_cosmo] for cosmo_param in conf["analysis"]["params"][:6]]
                cosmo = np.array(cosmo, dtype=np.float32)

                i_sobol = cosmo_params_info["sobol_index"][i_cosmo]

                # extend the Sobol sequence by astrophysical parameters
                sobol_point, _ = i4_sobol(sobol_priors.shape[0], i_sobol)
                sobol_params = sobol_point * np.squeeze(np.diff(sobol_priors)) + sobol_priors[:, 0]
                sobol_params = sobol_params.astype(np.float32)

                # add these to the label
                Aia = sobol_params[6]
                bg = sobol_params[7]
                n_Aia = sobol_params[8]
                n_bg = sobol_params[9]
                cosmo = np.concatenate((cosmo, np.array([Aia, bg, n_Aia, n_bg])))

                # redshift evolution
                tomo_Aia = redshift.get_tomo_amplitudes(Aia, n_Aia, tomo_z_metacal, tomo_nz_metacal, z0)
                tomo_bg = redshift.get_tomo_amplitudes(bg, n_bg, tomo_z_maglim, tomo_nz_maglim, z0)
                LOGGER.debug(f"Aia = {tomo_Aia}")
                LOGGER.debug(f"bg = {tomo_bg}")


                # verify that the Sobol sequences are identical (the parameters are ordered differently)
                assert np.allclose(sobol_params[0], cosmo[0], rtol=1e-3, atol=1e-5)  # Om
                assert np.allclose(sobol_params[1], cosmo[1], rtol=1e-3, atol=1e-5)  # s8
                assert np.allclose(sobol_params[2], cosmo[3], rtol=1e-3, atol=1e-5)  # Ob
                assert np.allclose(sobol_params[3], cosmo[2], rtol=1e-3, atol=1e-5)  # H0
                assert np.allclose(sobol_params[4], cosmo[4], rtol=1e-3, atol=1e-5)  # ns
                assert np.allclose(sobol_params[5], cosmo[5], rtol=1e-3, atol=1e-5)  # w0

                # load the .h5 files
                file_cosmo = filenames.get_filename_data_vectors(cosmo_dir_in, with_bary=args.with_bary)
                kg_examples, ia_examples, sn_examples, dg_examples = load_data_vecs(file_cosmo)

                # loop over the n_examples_per_cosmo
                for kg, ia, sn_realz, dg in LOGGER.progressbar(
                    zip(kg_examples, ia_examples, sn_examples, dg_examples),
                    at_level="debug",
                    desc="Looping through the examples of one cosmology",
                    total=n_examples_per_cosmo,
                ):
                    # intrinsic alignment (on kappa level), broadcast (data_vec_len, n_z_metacal) and (n_z_metacal,)
                    kg += tomo_Aia * ia

                    # multiplicative shear bias, broadcast (data_vec_len, n_z_metacal) and (n_z_metacal,)
                    m_bias = m_bias_dist.sample()
                    kg *= 1.0 + m_bias

                    # apply the galaxy bias and Poisson noise, broadcast the tomo bin axis
                    # broadcast (data_vec_len, n_z_maglim) and (n_z_maglim,)
                    dg = tomo_n_gal_maglim * (1 + tomo_bg * dg)
                    dg = np.where(0 < dg, dg, 0)
                    dg = np.random.poisson(dg)

                    serialized = tfrecords.parse_forward_grid(kg, sn_realz, dg, cosmo, i_sobol).SerializeToString()

                    # check correctness
                    i_noise = 0
                    inv_kg, inv_sn, inv_dg, inv_cosmo, inv_index = tfrecords.parse_inverse_grid(serialized, i_noise)

                    assert np.allclose(inv_kg, kg)
                    assert np.allclose(inv_sn, sn_realz[i_noise])
                    assert np.allclose(inv_dg, dg)
                    assert np.allclose(inv_cosmo, cosmo)
                    assert np.allclose(inv_index[0], i_sobol)
                    assert np.allclose(inv_index[1], i_noise)

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
