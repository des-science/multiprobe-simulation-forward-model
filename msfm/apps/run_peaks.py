# Copyright (C) 2023 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created December 2023
Author: Arne Thomsen

Evaluate the peaks from the .tfrecords files produced by the forward model pipelines. Based off
https://github.com/des-science/y3-combined-peaks/blob/main/combined_peaks/weak_lensing/peaks/MPI_WL_peaks_fiducial.py
by Virginia Ajani and run_power_spectra.py from this repo.

Before running this, create the binning scheme with dev/scripts/peaks/make_peaks_binning.py.

For the grid, the examples of a cosmology are distributed over a process pool with one worker per available core.
"""

import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import os, sys, types, contextlib, argparse, warnings, h5py, glob, time

from msfm import fiducial_pipeline, grid_pipeline
from msfm.utils import files, logger, input_output, imports, maps, peak_statistics, cosmogrid

hp = imports.import_healpy(parallel=False)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logger.get_logger(__file__)


@contextlib.contextmanager
def _hide_main_module():
    """The spawn and forkserver start methods of multiprocessing re-import the __main__ module in every worker. When
    run with esub, that is esub.submit, which has no __main__ guard and would re-execute the job in every worker. An
    empty placeholder hides it while the workers are started."""
    main_module = sys.modules["__main__"]
    sys.modules["__main__"] = types.ModuleType("__main__")
    try:
        yield
    finally:
        sys.modules["__main__"] = main_module


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

    # memory: in MB per core

    if args.simset == "fiducial":
        resource_dict = dict(main_memory=512, main_time=4, main_scratch=0, main_n_cores=8)
    elif args.simset == "grid":
        # v18: 400 examples (all noise realizations) per .tfrecord and 5 smoothing scales. On a busy Euler node, ~16 min
        # to import and load the .tfrecord, then ~10s per example with 8 workers, so ~80 min in total. The main process
        # holds ~5-13 GB, every worker ~1.3 GB, 17-17.6 GB MaxRSS was measured in total
        resource_dict = dict(main_memory=3072, main_time=3, main_scratch=0, main_n_cores=8)

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
    parser.add_argument(
        "--max_sleep",
        type=int,
        default=60,
        help="set the maximal amount of time to sleep before copying to avoid clashes",
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

    if args.debug:
        args.max_sleep = 0
        LOGGER.warning("!!! debug mode !!!")

    sleep_sec = np.random.uniform(0, args.max_sleep) if args.max_sleep > 0 else 0
    LOGGER.info(f"Waiting for {sleep_sec:.2f}s to prevent overloading IO")
    time.sleep(sleep_sec)

    conf = files.load_config(args.config)

    # setup up directories
    file_dir = os.path.dirname(__file__)
    repo_dir = os.path.abspath(os.path.join(file_dir, "../.."))
    binning_file = os.path.join(repo_dir, conf["files"]["peak_binning"])

    # i_sobol of every .tfrecord file, assuming a single cosmology per file like in run_grid_postprocessing.py
    if args.simset == "grid":
        meta_info_file = os.path.join(repo_dir, conf["files"]["meta_info"])
        i_sobols_of_files = cosmogrid.get_cosmo_params_info(meta_info_file, "grid")["sobol_index"]
        assert len(i_sobols_of_files) == len(tfrecords), "Expected a single cosmology per .tfrecord file"

    # constants
    n_side = conf["analysis"]["n_side"]
    n_pix = hp.nside2npix(n_side)
    n_z_bins = 0
    if conf.get("analysis", {}).get("modelling", {}).get("lensing", {}).get("store", True):
        n_z_bins += len(conf["survey"]["metacal"]["z_bins"])
    if conf.get("analysis", {}).get("modelling", {}).get("clustering", {}).get("store", True):
        n_z_bins += len(conf["survey"]["maglim"]["z_bins"])

    # peaks
    n_bins = conf["analysis"]["peak_statistics"]["n_bins"]
    theta_fwhm = conf["analysis"]["peak_statistics"]["theta_fwhm"]
    # white_noise_sigma = conf["analysis"]["peak_statistics"]["white_noise_sigma"]
    white_noise_sigma = None
    bins_edges, bins_centers, bins_fwhms = peak_statistics.get_peaks_bins(
        binning_file, n_z_bins=n_z_bins, with_cross=True
    )

    # CosmoGrid
    n_patches = conf["analysis"]["n_patches"]
    n_perms_per_cosmo = conf["analysis"][args.simset]["n_perms_per_cosmo"]
    # all noise realizations stored in the .tfrecords
    n_noise_per_example = conf["analysis"][args.simset]["n_noise_per_signal"]
    n_examples_per_cosmo = n_patches * n_perms_per_cosmo * n_noise_per_example

    def data_vector_to_peaks(data_vector, patch_pix):
        full_sky = np.full((n_pix, data_vector.shape[-1]), hp.UNSEEN)
        full_sky[patch_pix] = data_vector
        full_sky = maps.tomographic_reorder(full_sky, n2r=True)

        peaks = peak_statistics.get_peaks(
            full_sky,
            n_side=n_side,
            n_bins=n_bins,
            theta_fwhm=theta_fwhm,
            white_noise_sigma = white_noise_sigma,
            with_cross=True,
            bins_centers=bins_centers,
            bins_edges=bins_edges,
            bins_fwhms=bins_fwhms,
        )

        return peaks


    # index corresponds to a .tfrecord file ###########################################################################
    for index in indices:
        LOGGER.timer.start("index")

        tfrecord = tfrecords[index]
        LOGGER.info(f"Index {index} is reading from {tfrecord}")

        if args.simset == "grid":
            out_file = os.path.join(args.dir_out, f"grid_peaks_{i_sobols_of_files[index]:06}.h5")
            if os.path.isfile(out_file):
                LOGGER.warning(f"Skipping index {index}, since {out_file} already exists")
                yield index
                continue

            pipe = grid_pipeline.GridPipeline(
                conf, with_lensing=True, with_clustering=True, with_padding=False, apply_norm=False
            )
            dset = pipe.get_dset(
                tfr_pattern=tfrecord,
                local_batch_size="cosmo",
                noise_indices=n_noise_per_example,
                n_readers=1,
                n_prefetch=0,
            )

            # The workers are forked from a forkserver, not from the main process, since that runs TensorFlow. The
            # server preloads the peak statistics, which don't import TensorFlow. Every worker is single threaded.
            # Unlike multiprocessing.Pool, the executor raises when a worker dies (e.g. OOM) instead of hanging
            n_workers = len(os.sched_getaffinity(0))
            peaks_kwargs = dict(
                n_side=n_side,
                n_bins=n_bins,
                theta_fwhm=theta_fwhm,
                white_noise_sigma=white_noise_sigma,
                with_cross=True,
                bins_centers=bins_centers,
                bins_edges=bins_edges,
                bins_fwhms=bins_fwhms,
            )
            omp_num_threads = os.environ.get("OMP_NUM_THREADS")
            os.environ["OMP_NUM_THREADS"] = "1"
            mp_context = mp.get_context("forkserver")
            mp_context.set_forkserver_preload(["msfm.utils.peak_statistics"])
            pool = ProcessPoolExecutor(
                n_workers,
                mp_context=mp_context,
                initializer=peak_statistics.init_patch_worker,
                initargs=(pipe.patch_pix, n_pix, peaks_kwargs),
            )
            LOGGER.info(f"Set up a pool of {n_workers} workers")

            # one cosmology each. The Cls are None (return_cls=False), which as_numpy_iterator can't handle
            # the workers are started lazily within pool.map
            with _hide_main_module(), pool:
                for data_vectors, _, cosmos, (i_sobols, i_examples, i_noises) in dset:
                    data_vectors, cosmos = data_vectors.numpy(), cosmos.numpy()
                    i_sobols, i_examples, i_noises = i_sobols.numpy(), i_examples.numpy(), i_noises.numpy()
                    assert n_examples_per_cosmo == data_vectors.shape[0] == cosmos.shape[0] == i_sobols.shape[0]
                    assert np.all(i_sobols == i_sobols[0]), f"All i_sobols should be the same, but are {i_sobols}"
                    i_sobol = i_sobols[0]
                    assert i_sobol == i_sobols_of_files[index], f"Unexpected i_sobol {i_sobol} in {tfrecord}"

                    # shape (n_examples_per_cosmo, n_scales, n_bins, n_z_cross), map keeps the order of the examples
                    peaks = list(
                        LOGGER.progressbar(
                            pool.map(peak_statistics.get_peaks_from_patch, data_vectors),
                            total=n_examples_per_cosmo,
                            desc="Loop over examples",
                            at_level="info",
                        )
                    )
                    peaks = np.stack(peaks, axis=0)

                    # save one .h5 file per cosmology, written to a temporary file first such that an incomplete
                    # file is never mistaken for a finished one
                    with h5py.File(out_file + ".tmp", "w") as f:
                        f.create_dataset(name="peaks", data=peaks)
                        f.create_dataset(name="cosmo", data=cosmos)
                        f.create_dataset(name="i_sobol", data=i_sobol)
                        f.create_dataset(name="i_example", data=i_examples)
                        f.create_dataset(name="i_noise", data=i_noises)
                    os.replace(out_file + ".tmp", out_file)

            # the forkserver and workers are started lazily, so OMP_NUM_THREADS = 1 is kept until the pool is shut down
            if omp_num_threads is None:
                del os.environ["OMP_NUM_THREADS"]
            else:
                os.environ["OMP_NUM_THREADS"] = omp_num_threads

        elif args.simset == "fiducial":
            pipe = fiducial_pipeline.FiducialPipeline(
                conf,
                params=[],
                with_lensing=True,
                with_clustering=True,
                with_padding=False,
                apply_norm=False,
                apply_m_bias=True,
                shape_noise_scale=1.0,
                poisson_noise_scale=1.0,
            )
            dset = pipe.get_dset(
                tfr_pattern=tfrecord,
                local_batch_size=1,
                noise_indices=n_noise_per_example,
                n_readers=1,
                n_prefetch=0,
                is_eval=True,
            )
            dset = dset.as_numpy_iterator()

            peaks = []
            i_examples = []
            i_noises = []
            # loop over individual examples
            for data_vector, (i_example, i_noise) in LOGGER.progressbar(
                dset, total=n_examples_per_cosmo // len(tfrecords), desc="Loop over examples", at_level="info"
            ):
                # get rid of the batch dimension
                i_examples.append(i_example[0])
                i_noises.append(i_noise[0])
                data_vector = np.squeeze(data_vector)

                peaks.append(data_vector_to_peaks(data_vector, pipe.patch_pix))

            # shape (n_examples, n_scales, n_bins, n_z_cross)
            peaks = np.stack(peaks, axis=0)
            i_examples = np.stack(i_examples, axis=0)
            i_noises = np.stack(i_noises, axis=0)

            # save one .h5 file per input .tfrecord
            with h5py.File(os.path.join(args.dir_out, f"fiducial_peaks_{index:06}.h5"), "w") as f:
                f.create_dataset(name="peaks", data=peaks)
                f.create_dataset(name="i_example", data=i_examples)
                f.create_dataset(name="i_noise", data=i_noises)

        else:
            raise ValueError(f"Unknown simset {args.simset}")

        LOGGER.info(f"Done with index {index} after {LOGGER.timer.elapsed('index')}")
        yield index


def merge(indices, args):
    args = setup(args)
    LOGGER.info(f"Beginning with merge for {args.simset}")

    h5_pattern = os.path.join(args.dir_out, f"{args.simset}_peaks_??????.h5")
    h5_files = sorted(glob.glob(h5_pattern))
    n_files = len(h5_files)
    LOGGER.info(f"Found {n_files} files to merge")

    # the per cosmology files are deleted after merging, so only merge when all of them are present. Otherwise, the
    # finished ones would be lost and a rerun of main would have to recompute everything
    if args.simset == "grid":
        conf = files.load_config(args.config)
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        meta_info_file = os.path.join(repo_dir, conf["files"]["meta_info"])
        i_sobols_of_files = cosmogrid.get_cosmo_params_info(meta_info_file, "grid")["sobol_index"]
        expected = [os.path.join(args.dir_out, f"grid_peaks_{i_sobols_of_files[index]:06}.h5") for index in indices]
        missing = [index for index, h5_file in zip(indices, expected) if not os.path.isfile(h5_file)]
        if len(missing) > 0:
            raise FileNotFoundError(
                f"{len(missing)} of {len(indices)} per cosmology files are missing (indices {missing[:20]}...), "
                f"rerun main for them first. Nothing was merged or deleted"
            )

    # determine the per cosmology shapes
    with h5py.File(h5_files[0], "r") as f:
        peaks_shape = f["peaks"].shape
        i_example_shape = f["i_example"].shape
        i_noise_shape = f["i_noise"].shape

        if args.simset == "grid":
            cosmo_shape = f["cosmo"].shape
            i_sobol_shape = f["i_sobol"].shape

    # open the combined file
    with h5py.File(os.path.join(args.dir_out, f"{args.simset}_peaks.h5"), "w") as f_combined:
        LOGGER.info(f"Created the combined {args.simset} .h5 file")

        if args.simset == "grid":
            # define the combined output shapes
            # the peaks are integer counts, which float32 represents exactly at half the size of float64
            f_combined.create_dataset(name="peaks", shape=(n_files,) + peaks_shape, dtype=np.float32)
            f_combined.create_dataset(name="i_example", shape=(n_files,) + i_example_shape, dtype=np.int64)
            f_combined.create_dataset(name="i_noise", shape=(n_files,) + i_noise_shape, dtype=np.int64)
            f_combined.create_dataset(name="cosmo", shape=(n_files,) + cosmo_shape, dtype=np.float32)
            f_combined.create_dataset(name="i_sobol", shape=(n_files,) + i_sobol_shape, dtype=np.int64)

            # loop over the per cosmology .h5 files
            for i, h5_file in LOGGER.progressbar(enumerate(h5_files), desc="loop over files", at_level="info"):
                with h5py.File(h5_file, "r") as f:
                    peaks = f["peaks"][:]
                    cosmo = f["cosmo"][:]
                    i_sobol = f["i_sobol"][()]
                    i_example = f["i_example"][()]
                    i_noise = f["i_noise"][()]

                # write to the combined .h5 file
                f_combined["peaks"][i] = peaks
                f_combined["i_example"][i] = i_example
                f_combined["i_noise"][i] = i_noise
                f_combined["i_sobol"][i] = i_sobol
                f_combined["cosmo"][i] = cosmo
            

        elif args.simset == "fiducial":
            # define the combined output shapes
            f_combined.create_dataset(name="peaks", shape=(n_files * peaks_shape[0],) + peaks_shape[1:], dtype=np.float32)
            f_combined.create_dataset(
                name="i_example", shape=(n_files * i_example_shape[0],) + i_example_shape[1:], dtype=np.int64
            )
            f_combined.create_dataset(
                name="i_noise", shape=(n_files * i_noise_shape[0],) + i_noise_shape[1:], dtype=np.int64
            )

            # loop over the per .tfrecord file .h5 files
            for i, h5_file in LOGGER.progressbar(enumerate(h5_files), desc="loop over files", at_level="info"):
                with h5py.File(h5_file, "r") as f:
                    peaks = f["peaks"][:]
                    i_example = f["i_example"][()]
                    i_noise = f["i_noise"][()]

                # write to the combined .h5 file
                f_combined["peaks"][i * peaks_shape[0] : (i + 1) * peaks_shape[0]] = peaks
                f_combined["i_example"][i * i_example_shape[0] : (i + 1) * i_example_shape[0]] = i_example
                f_combined["i_noise"][i * i_noise_shape[0] : (i + 1) * i_noise_shape[0]] = i_noise

        else:
            raise ValueError(f"Unknown simset {args.simset}")

    LOGGER.info(f"Merged the per cosmology files into one and deleted them")

    # remove files after merged file has been written
    for i, h5_file in LOGGER.progressbar(enumerate(h5_files), desc="loop over files", at_level="info"):
        os.remove(h5_file)
    LOGGER.info(f"Removed temporary files")
