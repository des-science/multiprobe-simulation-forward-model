# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created December 2022
Author: Arne Thomsen

Evaluate the power spectra from the .tfrecords files produced by the forward model pipelines.
"""

import numpy as np
import os, argparse, warnings, h5py, glob

from msfm import fiducial_pipeline, grid_pipeline
from msfm.utils import power_spectra, files, logger, input_output, imports

hp = imports.import_healpy()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


def get_tasks(args):
    """Returns a list of task indices to be executed by the workers. Each index corresponds to a single .tfrecord file"""
    args = setup(args)

    # associate each file with a task
    tfrecords = sorted(glob.glob(args.tfr_pattern))
    indices = list(range(len(tfrecords)))

    # only read three files in debug mode
    if args.debug:
        indices = indices[:3]

    LOGGER.warning(f"Found {len(tfrecords)} files, running on the first {len(indices)} indices")

    return indices


def resources(args):
    args = setup(args)

    # 1h should actually be enough. CPU utilization is not great, but the memory usage is necessary
    if args.simset == "fiducial":
        resource_dict = dict(main_memory=1024, main_time=4, main_scratch=0, main_n_cores=4)
    elif args.simset == "grid":
        resource_dict = dict(main_memory=2048, main_time=4, main_scratch=0, main_n_cores=4)

    return resource_dict


def setup(args):
    description = "evaluate the power spectra from the input pipelines"
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
        "--tfr_pattern",
        type=str,
        required=True,
        help="input root dir of the .tfrecords to construct the dataset",
    )
    parser.add_argument(
        "--simset", type=str, default="grid", choices=("grid", "fiducial"), help="set of simulations to use"
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
    parser.add_argument("--debug", action="store_true", help="activate debug mode, then only run on 5 indices")

    args, _ = parser.parse_known_args(args)

    # print arguments
    logger.set_all_loggers_level(args.verbosity)
    for key, value in vars(args).items():
        LOGGER.info(f"{key} = {value}")

    if not os.path.isdir(args.dir_out):
        input_output.robust_makedirs(args.dir_out)

    assert args.simset in args.tfr_pattern

    return args


def main(indices, args):
    args = setup(args)
    tfrecords = sorted(glob.glob(args.tfr_pattern))

    conf = files.load_config(args.config)
    data_vec_pix, _, _, _ = files.load_pixel_file(conf)

    # setup up directories
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    hp_datapath = os.path.join(repo_dir, conf["files"]["healpy_data"])

    # constants
    n_side = conf["analysis"]["n_side"]
    n_pix = hp.nside2npix(n_side)

    # power spectra
    n_bins = conf["analysis"]["power_spectra"]["n_bins"]
    l_mins, l_maxs, _ = _get_l_ranges(conf)

    LOGGER.info(f"l_mins = {l_mins}")
    LOGGER.info(f"l_maxs = {l_maxs}")

    # CosmoGrid
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_cosmo = conf["analysis"][args.simset]["n_perms_per_cosmo"]
    n_noise_per_example = conf["analysis"][args.simset]["n_noise_per_example"]
    n_examples_per_cosmo = n_patches * n_perms_per_cosmo * n_noise_per_example

    def maps_to_cls(data_vector):
        # swap the tomographic and pixel axes
        maps = np.zeros((len(l_mins), n_pix))
        maps[:, data_vec_pix] = data_vector.T

        # get the alm coefficients
        alms = power_spectra.get_alms(maps, nest=True, datapath=hp_datapath)

        # get the power spectra
        cls = power_spectra.get_cls(alms, l_mins, l_maxs, n_bins, with_cross=True)

        return cls

    # index corresponds to a .tfrecord file ###########################################################################
    for index in indices:
        LOGGER.timer.start("index")

        tfrecord = tfrecords[index]
        LOGGER.info(f"Index {index} is reading from {tfrecord}")

        if args.simset == "grid":
            pipe = grid_pipeline.GridPipeline(conf, with_lensing=True, with_clustering=True, apply_norm=False)
            dset = pipe.get_dset(
                tfr_pattern=tfrecord, local_batch_size="cosmo", n_noise=n_noise_per_example, n_readers=1, n_prefetch=0
            )
            dset = dset.as_numpy_iterator()

            # one cosmology each
            for data_vectors, cosmos, (i_sobols, i_examples, i_noises) in dset:
                assert n_examples_per_cosmo == data_vectors.shape[0] == cosmos.shape[0] == i_sobols.shape[0]
                assert np.all(i_sobols == i_sobols[0]), f"All i_sobols should be the same, but are {i_sobols}"
                assert np.all(cosmos == cosmos[0]), f"All cosmological parameters should be the same, but are {cosmos}"
                cosmo = cosmos[0]
                i_sobol = i_sobols[0]
                LOGGER.info(f"Processing the cosmology with i_sobol = {i_sobol}")

                # loop over the batch dimension
                cls = []
                for data_vector in LOGGER.progressbar(
                    data_vectors, total=n_examples_per_cosmo, desc="Loop over examples", at_level="info"
                ):
                    cls.append(maps_to_cls(data_vector))

                cls = np.stack(cls, axis=0)

                # save one .h5 file per cosmology, TODO could also save to local scratch instead
                with h5py.File(os.path.join(args.dir_out, f"grid_cls_{i_sobol:06}.h5"), "w") as f:
                    f.create_dataset(name="cls", data=cls)
                    f.create_dataset(name="cosmo", data=cosmo)
                    f.create_dataset(name="i_sobol", data=i_sobol)
                    f.create_dataset(name="i_example", data=i_examples)
                    f.create_dataset(name="i_noise", data=i_noises)

        elif args.simset == "fiducial":
            pipe = fiducial_pipeline.FiducialPipeline(
                conf,
                params=[],
                with_lensing=True,
                with_clustering=True,
                apply_norm=False,
                apply_m_bias=True,
                shape_noise_scale=1.0,
                poisson_noise_scale=1.0,
            )
            dset = pipe.get_dset(
                tfr_pattern=tfrecord,
                local_batch_size=1,
                n_noise=n_noise_per_example,
                n_readers=1,
                n_prefetch=0,
                is_eval=True,
            )
            dset = dset.as_numpy_iterator()

            cls = []
            i_examples = []
            i_noises = []
            # loop over individual examples
            for data_vector, (i_example, i_noise) in LOGGER.progressbar(
                dset, desc="Loop over examples", at_level="info"
            ):
                # get rid of the batch dimension
                i_examples.append(i_example[0])
                i_noises.append(i_noise[0])
                data_vector = np.squeeze(data_vector)

                cls.append(maps_to_cls(data_vector))

            cls = np.stack(cls, axis=0)
            i_examples = np.stack(i_examples, axis=0)
            i_noises = np.stack(i_noises, axis=0)

            # save one .h5 file per input .tfrecord
            with h5py.File(os.path.join(args.dir_out, f"fiducial_cls_{index:06}.h5"), "w") as f:
                f.create_dataset(name="cls", data=cls)
                f.create_dataset(name="i_example", data=i_examples)
                f.create_dataset(name="i_noise", data=i_noises)

        else:
            raise ValueError(f"Unknown simset {args.simset}")

        LOGGER.info(f"Done with index {index} after {LOGGER.timer.elapsed('index')}")
        yield index


def merge(indices, args):
    args = setup(args)
    h5_pattern = os.path.join(args.dir_out, f"{args.simset}_cls_??????.h5")
    h5_files = sorted(glob.glob(h5_pattern))
    n_files = len(h5_files)

    # load the bin configuration
    conf = files.load_config(args.config)
    _, _, cl_bins = _get_l_ranges(conf)

    # determine the per cosmology shapes
    with h5py.File(h5_files[0], "r") as f:
        cls_shape = f["cls"].shape
        i_example_shape = f["i_example"].shape
        i_noise_shape = f["i_noise"].shape

        if args.simset == "grid":
            cosmo_shape = f["cosmo"].shape
            i_sobol_shape = f["i_sobol"].shape

    # open the combined file
    with h5py.File(os.path.join(args.dir_out, f"{args.simset}_cls.h5"), "w") as f_combined:
        f_combined.create_dataset(name="cl_bins", data=cl_bins)

        if args.simset == "grid":
            # define the combined output shapes
            f_combined.create_dataset(name="cls", shape=(n_files,) + cls_shape)
            f_combined.create_dataset(name="i_example", shape=(n_files,) + i_example_shape)
            f_combined.create_dataset(name="i_noise", shape=(n_files,) + i_noise_shape)
            f_combined.create_dataset(name="cosmo", shape=(n_files,) + cosmo_shape)
            f_combined.create_dataset(name="i_sobol", shape=(n_files,) + i_sobol_shape)

            # loop over the per cosmology .h5 files
            for i, h5_file in enumerate(h5_files):
                with h5py.File(h5_file, "r") as f:
                    cls = f["cls"][:]
                    cosmo = f["cosmo"][:]
                    i_sobol = f["i_sobol"][()]
                    i_example = f["i_example"][()]
                    i_noise = f["i_noise"][()]
                os.remove(h5_file)

                # write to the combined .h5 file
                f_combined["cls"][i] = cls
                f_combined["i_example"][i] = i_example
                f_combined["i_noise"][i] = i_noise
                f_combined["i_sobol"][i] = i_sobol
                f_combined["cosmo"][i] = cosmo

        elif args.simset == "fiducial":
            # define the combined output shapes
            f_combined.create_dataset(name="cls", shape=(n_files * cls_shape[0],) + cls_shape[1:])
            f_combined.create_dataset(name="i_example", shape=(n_files * i_example_shape[0],) + i_example_shape[1:])
            f_combined.create_dataset(name="i_noise", shape=(n_files * i_noise_shape[0],) + i_noise_shape[1:])

            # loop over the per .tfrecord file .h5 files
            for i, h5_file in enumerate(h5_files):
                with h5py.File(h5_file, "r") as f:
                    cls = f["cls"][:]
                    i_example = f["i_example"][()]
                    i_noise = f["i_noise"][()]
                os.remove(h5_file)

                # write to the combined .h5 file
                f_combined["cls"][i * cls_shape[0] : (i + 1) * cls_shape[0]] = cls
                f_combined["i_example"][i * i_example_shape[0] : (i + 1) * i_example_shape[0]] = i_example
                f_combined["i_noise"][i * i_noise_shape[0] : (i + 1) * i_noise_shape[0]] = i_noise

        else:
            raise ValueError(f"Unknown simset {args.simset}")

    LOGGER.info(f"Merged the per cosmology files into one and deleted them")

def _get_l_ranges(conf):
    """Helper function to get the l ranges for the power spectra from the configuration file to both main and merge"""

    lensing_l_mins = conf["analysis"]["scale_cuts"]["lensing"]["l_min"]
    lensing_l_maxs = conf["analysis"]["scale_cuts"]["lensing"]["l_max"]

    clustering_l_mins = conf["analysis"]["scale_cuts"]["clustering"]["l_min"]
    clustering_l_maxs = conf["analysis"]["scale_cuts"]["clustering"]["l_max"]

    l_mins = np.asarray(lensing_l_mins + clustering_l_mins, dtype=int)
    l_maxs = np.asarray(lensing_l_maxs + clustering_l_maxs, dtype=int)
    assert len(l_mins) == len(l_maxs)

    # include smaller scales in the cls because of the Gaussian smoothing. TODO justify the factor of 1.5 numerically
    n_side = conf["analysis"]["n_side"]
    l_maxs = np.clip(l_maxs * 1.5, 0, 3 * n_side - 1).astype(int)

    n_bins = conf["analysis"]["power_spectra"]["n_bins"]
    cl_bins = power_spectra.get_cl_bins(l_mins, l_maxs, n_bins)

    return l_mins, l_maxs, cl_bins
