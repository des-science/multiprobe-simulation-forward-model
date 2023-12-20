"""
Created September 2023
Author: Arne Thomsen

Handle some peculiarities around importing healpy.
"""

import os, logging
from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


def import_healpy(parallel=False):
    """set the environmental variable OMP_NUM_THREADS to the number of logical processors for healpy parallelization.

    Args:
        parallel (bool, optional): Whether to set the OMP_NUM_THREADS environmental variable or not. This should only
        be done in scripts where healpy is used for computationally heavy operations like spherical harmonics
        decompositions. Defaults to False.

    Returns:
        namespace: The healpy library.
    """

    if parallel:
        try:
            n_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            LOGGER.debug(f"os.sched_getaffinity is not available on this system, use os.cpu_count() instead")
            n_cpus = os.cpu_count()

        os.environ["OMP_NUM_THREADS"] = str(n_cpus)
        LOGGER.info(f"Setting up healpy to run on {n_cpus} CPUs")

    import healpy as hp

    # disable healpy logging since it tends to be very verbose
    hp_LOGGER = logging.getLogger("healpy")
    hp_LOGGER.disabled = True

    return hp
