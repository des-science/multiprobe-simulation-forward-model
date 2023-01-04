import os, sys

def get_filename_data_vectors(out_dir, with_bary=False):
    if with_bary:
        file_name = "data_vectors_baryonified.h5"
    else:
        file_name = "data_vectors_nobaryons.h5"

    return os.path.join(out_dir, file_name)

def get_filename_data_patches(out_dir, with_bary=False):
    if with_bary:
        file_name = "data_patches_baryonified.h5"
    else:
        file_name = "data_patches_nobaryons.h5"

    return os.path.join(out_dir, file_name)

def get_filename_full_maps(grid_dir, with_bary=False):
    if with_bary:
        file_name = "projected_probes_maps_baryonified512.h5"
    else:
        file_name = "projected_probes_maps_nobaryons512.h5"

    return os.path.join(grid_dir, file_name)

def get_filename_tfrecords(out_dir, index, tag, simset):
    return os.path.join(out_dir, f'{tag}_{simset}_patches_{index:03d}.tfrecord')
