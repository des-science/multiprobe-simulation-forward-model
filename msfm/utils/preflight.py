# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created November 2026
Author: Arne Thomsen

The inputs of a grid run, checked on the login node before it is submitted, so a missing file fails at submission
and not in the queue. The paths come from the helpers the apps use (msfm.utils.filenames, the metainfo through
msfm.utils.cosmogrid). The mocks have their own check, msfm.utils.mocks check.

The submission scripts call the CLI with a file of indices, one per line:
    python -m msfm.utils.preflight grid <config> --dir_in <projection> --indices <file>
    python -m msfm.utils.preflight validate <config> --dir_in <projection> --indices <file> --tfr_dir <dir>
"""

import argparse
import os
import sys

from msfm.utils import cosmogrid, filenames, files, logger

LOGGER = logger.get_logger(__file__)

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# a task that dies mid-write leaves a short stub; a real .tfrecord is GBs
MIN_TFRECORD_BYTES = 2**20


def missing_config_files(conf):
    """The files: entries of a config that name a repository path which does not exist. files: also holds values
    that are no paths, which are skipped.

    Returns:
        list: the problems found, empty if none
    """
    return [
        f"files.{key} = {rel} does not exist"
        for key, rel in conf["files"].items()
        if isinstance(rel, str) and "/" in rel and not os.path.exists(os.path.join(REPO_DIR, rel))
    ]


def grid_inputs(conf, dir_in, indices):
    """The projected full-sky files run_grid_postprocessing.py reads for the indices: for each index, the index-th
    grid cosmology of the metainfo, one file per permutation."""
    path_par = cosmogrid.get_cosmo_params_info(os.path.join(REPO_DIR, conf["files"]["meta_info"]), "grid")["path_par"]
    with_bary = conf["analysis"]["modelling"]["baryonified"]
    return [
        filenames.get_filename_full_maps(
            os.path.join(dir_in, "grid", path_par[index].decode("utf-8"), f"perm_{i_perm:04d}"), with_bary=with_bary
        )
        for index in indices
        for i_perm in range(conf["analysis"]["grid"]["n_perms_per_cosmo"])
    ]


def grid_tfrecord(conf, tfr_dir, index):
    """The .tfrecord run_grid_postprocessing.py writes for an index."""
    with_bary = conf["analysis"]["modelling"]["baryonified"]
    return filenames.get_filename_tfrecords(
        tfr_dir, index, tag=conf["survey"]["name"], simset="grid", with_bary=with_bary
    )


def check(chain, conf, dir_in, indices, tfr_dir=None):
    """The problems of a grid or validate run over the indices: the config passes the pipelines' checks, every file it
    names exists and every index has all its projected files. validate also needs each index's .tfrecord.

    Returns:
        list: the problems found, empty if none
    """
    from msfm.utils import configuration

    problems = missing_config_files(conf)
    if not problems:
        try:
            configuration.print_and_check_modeling_in_config(conf)
        except Exception as err:
            problems.append(f"{type(err).__name__}: {err}")

    inputs = grid_inputs(conf, dir_in, indices)
    missing = [file for file in inputs if not os.path.isfile(file)]
    if missing:
        problems.append(f"{len(missing)} of {len(inputs)} projected files are missing, e.g. {missing[0]}")

    if chain == "validate":
        tfrecords = [grid_tfrecord(conf, tfr_dir, index) for index in indices]
        bad = [t for t in tfrecords if not os.path.isfile(t) or os.path.getsize(t) < MIN_TFRECORD_BYTES]
        if bad:
            problems.append(f"{len(bad)} of {len(indices)} .tfrecords are missing or stubs, e.g. {bad[0]}")

    LOGGER.info(f"{chain}: {len(indices)} indices, {len(inputs)} projected files checked")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("chain", choices=["grid", "validate"])
    parser.add_argument("config", help="msfm config, e.g. configs/desy6/v2/v2.yaml")
    parser.add_argument("--dir_in", required=True, help="the projection root, which holds grid/")
    parser.add_argument("--indices", required=True, help="file of the indices to check, one per line")
    parser.add_argument("--tfr_dir", default=None, help="validate: the directory of the .tfrecords, ignored for grid")
    args = parser.parse_args(argv)
    if args.chain == "validate" and args.tfr_dir is None:
        parser.error("validate needs --tfr_dir")

    with open(args.indices) as f:
        indices = [int(line) for line in f if line.strip()]
    problems = check(args.chain, files.load_config(args.config), args.dir_in, indices, args.tfr_dir)
    for problem in problems:
        LOGGER.error(problem)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
