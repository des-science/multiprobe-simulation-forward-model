import numpy as np
import tensorflow as tf
import os

# import h5py
import healpy as hp

# import os, argparse, warnings, time, yaml, h5py, pickle, healpy

# from scipy.stats import qmc
# from sobol_seq import i4_sobol

# from msfm.utils import (
#     logger,
#     imports,
#     filenames,
#     input_output,
#     files,
#     lensing,
#     clustering,
#     cosmogrid,
#     postprocessing,
#     tfrecords,
#     power_spectra,
#     scales,
#     redshift,
#     parameters,
#     configuration,
# )

# hp = imports.import_healpy()


def setup(args):
    print(f"Running on {len(os.sched_getaffinity(0))} cores (setup)")

    return args


def main(indices, args):
    for index in indices:
        print(f"Running on {len(os.sched_getaffinity(0))} cores (main)")
        print(os.environ["SLURM_CPUS_PER_TASK"])

        yield index
