import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import tensorflow as tf
import tensorflow_probability as tfp
import os, h5py

from tqdm import tqdm
from numba import njit

from msfm.utils.io import read_yaml
from msfm.utils.maps import make_normallized_maps

# global constants
conf_dir = "../../configs/config.yaml"
conf = read_yaml(conf_dir)
n_side = conf["analysis"]["n_side"]
n_pix = conf["analysis"]["n_pix"]
tomo = 1


def load_metacal_catalog(metacal_dir):
    # ellipticities for a fixed tomographic bin
    e1 = h5py.File(os.path.join(metacal_dir, f"cal_e1_tomo={tomo}.h5"))["cal_e1"][:]
    e2 = h5py.File(os.path.join(metacal_dir, f"cal_e2_tomo={tomo}.h5"))["cal_e2"][:]
    w = h5py.File(os.path.join(metacal_dir, f"weight_tomo={tomo}.h5"))["weight"][:]

    # J2000 angles in degrees
    alpha_metacal = h5py.File(os.path.join(metacal_dir, f"ALPHAWIN_J2000_tomo={tomo}.h5"))["ALPHAWIN_J2000"][:]
    delta_metacal = h5py.File(os.path.join(metacal_dir, f"DELTAWIN_J2000_tomo={tomo}.h5"))["DELTAWIN_J2000"][:]
    # angles like in healpy in radian
    theta = -np.deg2rad(delta_metacal) + np.pi / 2
    phi = np.deg2rad(alpha_metacal)

    assert e1.shape == e2.shape == w.shape == alpha_metacal.shape == delta_metacal.shape

    # galaxy level: derived pixel ids
    metacal_ids = hp.ang2pix(nside=n_side, theta=theta, phi=phi)

    # shape (num_galaxies,)
    return metacal_ids, e1, e2, w


def make_map(footprint_ids, values):
    hp_map = np.zeros(n_pix)
    hp_map[footprint_ids] = values

    return hp_map


def make_noise_rotation(metacal_ids, e1, e2, w, N, out_dir):
    """
    Rotate galaxies in place
    """
    n1_maps = []
    n2_maps = []
    for i in range(N):
        print(f"\nRandom map #{i}")

        phase = np.random.uniform(0.0, 2 * np.pi, size=e1.shape)
        n1 = e1 * np.cos(phase)
        n2 = e2 * np.sin(phase)

        n1_map, n2_map, _, _, _ = make_normallized_maps(metacal_ids, n1, n2, w, n_pix)

        n1_maps.append(n1_map)
        n2_maps.append(n2_map)

    n1_maps = np.array(n1_maps)
    n2_maps = np.array(n2_maps)

    np.savez(os.path.join(out_dir, "rotate_in_place"), g1=n1_maps, g2=n2_maps)

    return n1_maps, n1_maps

@njit(parallel=True)
def make_noise_catalog_sampling(footprint_ids, n_gals_per_id, e1, e2, w, N, out_dir):
    """
    Sample the ellipticities per pixel from the catalog. This is way too slow to be used.
    """
    n1_maps = np.zeros((N, n_pix))
    n2_maps = np.zeros((N, n_pix))
    for i in range(N):
        print(f"\nRandom map #{i}")

        j = 0
        # loop over all of the pixels
        for footprint_id, n_gal in zip(footprint_ids, n_gals_per_id):
            rand_inds = np.random.choice(np.arange(len(e1)), n_gal, replace=True)
            rand_inds = np.sort(rand_inds)

            n1_maps[i, footprint_id] = np.sum(e1[rand_inds]*w[rand_inds])/np.sum(w[rand_inds])
            n2_maps[i, footprint_id] = np.sum(e2[rand_inds]*w[rand_inds])/np.sum(w[rand_inds])

            j += 1

    np.savez(os.path.join(out_dir, "sample_catalog"), g1=n1_maps, g2=n2_maps)

    return n1_maps, n2_maps

def make_noise_empirical_sampling(footprint_ids, n_gals_per_id, e1, e2, w, N, out_dir):
    """
    Sample the ellipticities from the histogram of the catalog. Then, the pixels are populated accordingly.
    """
    total_gals = np.sum(n_gals_per_id)

    emp_dist = tfp.distributions.Empirical(samples=np.stack([e1, e2, w], axis=1), event_ndims=1)

    n1_maps = np.zeros((N, n_pix))
    n2_maps = np.zeros((N, n_pix))
    for i in range(N):
        print(f"\nRandom map #{i}")

        # shape (total_gals, 3) for e1, e2 and w
        samples = emp_dist.sample(sample_shape=total_gals)
        e_samples = samples[:,:2]
        w_samples = samples[:,2]

        # apply weights
        e_samples *= tf.expand_dims(w_samples, axis=1)

        seg_ids = []
        for id, n_gals in enumerate(n_gals_per_id):
            seg_ids.extend(n_gals*[id])

        e_per_pix = tf.math.segment_sum(e_samples, seg_ids)
        w_per_pix = tf.math.segment_sum(w_samples, seg_ids)

        # normalize with weights
        e_per_pix /= tf.expand_dims(w_per_pix, axis=1)

        n1_maps[i, footprint_ids] = e_per_pix[:,0].numpy()
        n2_maps[i, footprint_ids] = e_per_pix[:,1].numpy()
    
    np.savez(os.path.join(out_dir, "sample_empirical_dist"), g1=n1_maps, g2=n2_maps)






if __name__ == "__main__":
    # number of noise realizations
    N = 10

    # set paths
    metacal_dir = "/Users/arne/data/DESY3/DES_Y3KP_NGSF/"
    out_dir = "data/"

    metacal_ids, e1, e2, w = load_metacal_catalog(metacal_dir)
    
    # pixel level: derived pixel ids and number of galaxies per pixel
    footprint_ids, n_gals_per_id = np.unique(metacal_ids, return_counts=True)

    # rotate in place
    # print("rotating galaxies in place")
    # n1_maps, n2_maps = make_noise_rotation(metacal_ids, e1, e2, w, N, out_dir)

    # sample from catalog
    # n1_maps, n2_maps = make_noise_catalog_sampling(footprint_ids, n_gals, e1, e2, w, N)

    # sample from empirical distribution
    print("empirical sampling")
    make_noise_empirical_sampling(footprint_ids, n_gals_per_id, e1, e2, w, N, out_dir)
