# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created November 2026
Author: Arne Thomsen

Where the data of a dataset lives on disk, written once per version next to its config, e.g.
configs/desy6/v2/locations.yaml beside configs/desy6/v2/v2.yaml, and read by msfm, y3-deep-lss and msi. A dataset
is one version of a survey with one config, configs/<survey>/<version>/<version>.yaml (dataset_config). The file
holds machine paths only, never a science choice, so moving a directory changes no config.

    projection    the CosmoGrid projection, input (holds grid/ and benchmarks/)
    data          msfm outputs: <data>/{tfrecords/grid, cls/grid_cls.h5, obs, validation, packed_logs}
    obs           the mock observations, when they are not under data: <obs>/obs
    runs          y3-deep-lss runs, read by msi: <runs>/<representation>/<probe>/<run>
    archive       a durable copy of data, laid out the same way; no script reads it
    runs_archive  a durable copy of runs, laid out the same way; no script reads it

Every key is optional, since a dataset may lack one, and every value must be an absolute path.

The resolved paths of a config are projection, data_dir, obs_dir, runs_dir, archive_dir and runs_archive_dir,
present when the dataset has them. Shell scripts read them through the CLI, and an environment variable may still
override one path for a one-off run:

    python -m msfm.utils.locations <msfm config>                  # every resolved path, key=value per line
    python -m msfm.utils.locations <msfm config> --key data_dir   # one path; exits 1 if the dataset has none
"""

import argparse
import os
import sys

from msfm.utils import input_output

FILE_NAME = "locations.yaml"
KEYS = ("projection", "data", "obs", "runs", "archive", "runs_archive")
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def dataset_config(dataset):
    """The msfm config of a dataset <survey>/<version>, e.g. desy6/v2 gives <repo>/configs/desy6/v2/v2.yaml."""
    dataset = os.path.normpath(dataset)
    return os.path.join(REPO_DIR, "configs", dataset, f"{os.path.basename(dataset)}.yaml")


def locations_file(config_path):
    """The locations file of an msfm config: the one next to it."""
    return os.path.join(os.path.dirname(os.path.abspath(config_path)), FILE_NAME)


def read(path):
    """Reads and validates a locations file. Returns the dict of its keys, paths normalized."""
    raw = input_output.read_yaml(path)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must map keys to paths")
    unknown = sorted(set(raw) - set(KEYS))
    if unknown:
        raise KeyError(f"{path}: unknown keys {unknown}, the keys are {list(KEYS)}")
    for key, value in raw.items():
        if not isinstance(value, str) or not os.path.isabs(value):
            raise ValueError(f"{path}: {key} must be an absolute path, got {value!r}")
    return {key: os.path.normpath(value) for key, value in raw.items()}


def load(config_path):
    """The resolved paths of an msfm config, from its locations file.

    Returns:
        dict: projection, data_dir, obs_dir, runs_dir, archive_dir and runs_archive_dir, each only if the dataset
            has it
    """
    path = locations_file(config_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} does not exist, so {config_path} has no data locations")
    roots = read(path)

    resolved = {}
    if "projection" in roots:
        resolved["projection"] = roots["projection"]
    for key in ("data", "runs", "archive", "runs_archive"):
        if key in roots:
            resolved[f"{key}_dir"] = roots[key]
    obs_root = roots.get("obs", roots.get("data"))
    if obs_root is not None:
        resolved["obs_dir"] = os.path.join(obs_root, "obs")
    return resolved


def get(config_path, key):
    """One resolved path of an msfm config. Raises a KeyError naming the locations file if the dataset has none."""
    resolved = load(config_path)
    if key not in resolved:
        raise KeyError(f"{locations_file(config_path)} gives no {key}; it resolves {sorted(resolved)}")
    return resolved[key]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="msfm config, e.g. configs/desy6/v2/v2.yaml")
    parser.add_argument("--key", default=None, help="print only this resolved path")
    args = parser.parse_args(argv)

    try:
        if args.key is not None:
            print(get(args.config, args.key))
        else:
            for key, value in load(args.config).items():
                print(f"{key}={value}")
    except (FileNotFoundError, KeyError, ValueError) as err:
        # str() of a KeyError quotes its message
        print(f"msfm.utils.locations: {err.args[0] if err.args else err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
