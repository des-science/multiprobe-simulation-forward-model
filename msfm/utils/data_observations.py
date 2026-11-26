# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created November 2026
Author: Arne Thomsen

The variants of the real data observation of a forward-model version, all defined in one registry file next to the
version's config, e.g. configs/desy6/v2/data_observations.yaml:

    base: v2.yaml             # the config the registry belongs to, relative to the registry
    observations:
      <label>:                # <survey.name>_<suffix>: the key of the network predictions and the name of the chain
        lens_weights: <key>   # a files: key of the base config, a file whose maglim/systematics group (on the patch of
                              # files.pixels) corrects the observed lens counts instead of the one of files.pixels

The default observation is the base config itself, labelled survey.name, and is not listed. A variant changes only how
the observed maps are made from the catalogs, never the forward model, so networks trained before it existed evaluate
it unchanged. Unlike the mocks (msfm.utils.mocks), whose maps are files, y3-deep-lss makes it on the fly from the
catalogs, like the default observation.

    python -m msfm.utils.data_observations labels <registry>
    python -m msfm.utils.data_observations check <registry> [--config <msfm config>]   # default: the base
"""

import argparse
import os
import sys
from dataclasses import dataclass

import h5py

from msfm.utils import files, input_output, logger, mocks

LOGGER = logger.get_logger(__file__)

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
REGISTRY_KEYS = {"base", "observations"}
OBSERVATION_KEYS = {"lens_weights"}


@dataclass(frozen=True)
class DataObservation:
    label: str
    # absolute path of the lens weight file
    lens_weights: str


def base_config(path):
    """Absolute path of the config the registry belongs to."""
    return os.path.join(os.path.dirname(os.path.abspath(path)), input_output.read_yaml(path)["base"])


def load(path):
    """Reads and validates a registry against its base config. Returns a list of DataObservation."""
    reg = input_output.read_yaml(path)
    if not isinstance(reg, dict) or set(reg) != REGISTRY_KEYS:
        raise KeyError(f"{path} must have exactly the keys {sorted(REGISTRY_KEYS)}")
    if not isinstance(reg["observations"], dict) or not reg["observations"]:
        raise ValueError(f"{path}: observations is empty")

    base = files.load_config(base_config(path))
    survey = base["survey"]["name"]
    observations = []
    for label, entry in reg["observations"].items():
        if not label.startswith(f"{survey}_") or not mocks.LABEL_PATTERN.match(label):
            raise ValueError(f"{path}: the label {label!r} is not {survey}_<suffix>, a plain file name stem")
        if not isinstance(entry, dict) or set(entry) != OBSERVATION_KEYS:
            raise KeyError(f"{path}: observations.{label} must have exactly the keys {sorted(OBSERVATION_KEYS)}")
        key = entry["lens_weights"]
        if key not in base["files"]:
            raise KeyError(f"{path}: observations.{label}.lens_weights = {key} is not a files: key of {reg['base']}")
        observations.append(DataObservation(label, os.path.join(REPO_DIR, base["files"][key])))
    return observations


def labels(path):
    return [obs.label for obs in load(path)]


def check(path, conf=None):
    """Checks that the registry's observations apply to the msfm config conf, by default the base: the same survey and
    pixel file, and a lens weight file holding every lens bin of conf on its data vector.

    Returns:
        list: the problems found, empty if none
    """
    try:
        observations = load(path)
    except Exception as err:
        return [f"{type(err).__name__}: {err}"]
    base = files.load_config(base_config(path))
    conf = base if conf is None else files.load_config(conf)

    problems = []
    for key in ("survey", "files"):
        sub = "name" if key == "survey" else "pixels"
        if conf[key][sub] != base[key][sub]:
            problems.append(f"{key}.{sub} is {conf[key][sub]}, but the registry's base has {base[key][sub]}")
    if problems:
        return problems

    data_vec_len = files.load_pixel_file(conf)[0].size
    for obs in observations:
        if not os.path.isfile(obs.lens_weights):
            problems.append(f"{obs.label}: no lens weight file {obs.lens_weights}")
            continue
        with h5py.File(obs.lens_weights, "r") as f:
            for z_bin in conf["survey"]["maglim"]["z_bins"]:
                name = f"maglim/systematics/data_vector/{z_bin}"
                if name not in f or f[name].shape != (data_vec_len,):
                    problems.append(f"{obs.label}: {obs.lens_weights} has no {name} of length {data_vec_len}")
        LOGGER.info(f"{obs.label}: lens weights {os.path.relpath(obs.lens_weights, REPO_DIR)}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("labels").add_argument("registry")
    check_parser = sub.add_parser("check")
    check_parser.add_argument("registry")
    check_parser.add_argument("--config", default=None, help="the msfm config to check against (default: the base)")
    args = parser.parse_args(argv)

    if args.command == "labels":
        print("\n".join(labels(args.registry)))
    else:
        problems = check(args.registry, args.config)
        for problem in problems:
            LOGGER.error(problem)
        sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
