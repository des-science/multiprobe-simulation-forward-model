import os, h5py, warnings

from msfm.utils import logging, input_output

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("once", category=UserWarning)
LOGGER = logging.get_logger(__file__)


def load_pixel_file(conf, repo_dir):
    """Loads the .h5 file that contains the pixel indices associated with the survey like the different patches. That
    file is generated in notebooks/survey_file_gen/pixel_file.ipynb

    Args:
        conf (yaml): yaml dictionary setting the configuration parameters
        repo_dir (str): absolute path to the repository

    Returns:
        data_vec_pix: data vector pixels including padding in NEST ordering (non-tomographic)
        non_tomo_patches_pix: non padded patches in RING ordering (non-tomographic)
        gamma2_signs: signs for gamma2 that come from mirroring the survey patch
        tomo_patches_pix: tomographic patch indices in RING ordering to cut out from the full sky maps
        tomo_corresponding_pix: needed to convert the pixels in RING ordering to NEST
    """
    pixel_file = os.path.join(repo_dir, conf["files"]["pixels"])

    with h5py.File(pixel_file, "r") as f:
        # pixel indices of padded data vector
        data_vec_pix = f["metacal/map_cut_outs/data_vec_ids"][:]

        # pixel indices of the non padded patches (non tomographic)
        patches_pix = f["metacal/masks/RING/non_tomo"][:]

        # to correct the shear for patch cut outs that have been mirrored
        gamma2_signs = f["metacal/map_cut_outs/patches/gamma_2_sign"][:]

        tomo_patches_pix = []
        tomo_corresponding_pix = []
        for z_bin in conf["survey"]["metacal"]["z_bins"]:
            # shape (4, pix_in_bin)
            patches_pix = f[f"metacal/map_cut_outs/patches/RING/{z_bin}"][:]
            # shape (pix_in_bin,)
            corresponding_pix = f[f"metacal/map_cut_outs/RING_ids_to_data_vec/{z_bin}"][:]

            tomo_patches_pix.append(patches_pix)
            tomo_corresponding_pix.append(corresponding_pix)
    LOGGER.info(f"Loaded pixel file")

    return data_vec_pix, patches_pix, gamma2_signs, tomo_patches_pix, tomo_corresponding_pix
