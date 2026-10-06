"""
Create the peaks binning scheme (files.peak_binning in the config) from a single forward-modeled grid example. This is
the procedure of dev/notebooks/peaks/binning.ipynb, but reading from a grid .tfrecord, since from v18 on there's no
fiducial set. Pick a .tfrecord whose cosmology is close to the fiducial.

Example:
python dev/scripts/peaks/make_peaks_binning.py \
    --config configs/v18/default.yaml \
    --tfrecord /cluster/work/refregier/jbucko/des_ng_ml/v18/tfrecords/grid/DESy3_grid_dmb_2281.tfrecord \
    --binning_file data/peaks/binning/v18_default.h5
"""

import argparse, os
import numpy as np

from msfm import grid_pipeline
from msfm.utils import files, logger, imports, maps, peak_statistics

hp = imports.import_healpy(parallel=False)
LOGGER = logger.get_logger(__file__)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True, help="configuration yaml file")
    parser.add_argument("--tfrecord", type=str, required=True, help="single grid .tfrecord file")
    parser.add_argument("--binning_file", type=str, required=True, help="output .h5 file, must not exist yet")
    parser.add_argument("--signal_index", type=int, default=0, help="example (patch x permutation) within the file")
    parser.add_argument("--noise_index", type=int, default=0, help="noise realization of that example")
    args = parser.parse_args()

    if os.path.exists(args.binning_file):
        raise FileExistsError(f"{args.binning_file} already exists, delete it first to recreate the binning")

    conf = files.load_config(args.config)
    peak_conf = conf["analysis"]["peak_statistics"]

    pipe = grid_pipeline.GridPipeline(
        conf, with_lensing=True, with_clustering=True, with_padding=False, apply_norm=False
    )
    dset = pipe.get_dset(
        tfr_pattern=args.tfrecord,
        local_batch_size=1,
        signal_indices=[args.signal_index],
        noise_indices=[args.noise_index],
        n_readers=1,
        n_prefetch=0,
    )

    # the Cls are None (return_cls=False), which as_numpy_iterator can't handle
    data_vector, _, cosmo, indices = next(iter(dset))
    data_vector, cosmo = data_vector.numpy(), cosmo.numpy()
    i_sobol, i_signal, i_noise = (index.numpy() for index in indices)
    LOGGER.info(f"i_sobol = {i_sobol[0]}, i_signal = {i_signal[0]}, i_noise = {i_noise[0]}")
    LOGGER.info(f"cosmo = {np.array2string(cosmo[0], precision=4)}")

    # same as in msfm/apps/run_peaks.py
    data_vector = np.squeeze(data_vector, axis=0)
    full_sky = np.full((conf["analysis"]["n_pix"], data_vector.shape[-1]), hp.UNSEEN)
    full_sky[pipe.patch_pix] = data_vector
    full_sky = maps.tomographic_reorder(full_sky, n2r=True)

    # with binning_file, the binning scheme is derived from these peaks and stored
    peaks = peak_statistics.get_peaks(
        full_sky,
        n_side=conf["analysis"]["n_side"],
        n_bins=peak_conf["n_bins"],
        theta_fwhm=peak_conf["theta_fwhm"],
        with_cross=True,
        binning_file=args.binning_file,
    )

    with open(args.binning_file + ".info", "w") as f:
        f.write(
            f"config = {os.path.abspath(args.config)}\ntfrecord = {args.tfrecord}\n"
            f"i_sobol = {i_sobol[0]}\ni_signal = {i_signal[0]}\ni_noise = {i_noise[0]}\ncosmo = {cosmo[0].tolist()}\n"
        )

    LOGGER.info(f"Stored the binning scheme in {args.binning_file}, peaks of this example have shape {peaks.shape}")


if __name__ == "__main__":
    main()
