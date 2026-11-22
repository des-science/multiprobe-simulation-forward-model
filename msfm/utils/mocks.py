# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created November 2026
Author: Arne Thomsen

The mock observations of a forward-model version, all defined in one registry file next to the version's config,
e.g. configs/desy6/v1/mock_observations.yaml:

    base: default.yaml        # the config every mock starts from, relative to the registry
    n_perms: 20               # permutations per mock, perm_0000 .. perm_<n_perms - 1>
    args:                     # forward-model flags of run_single_postprocessing.py, for every mock
      with_lensing: True
    mocks:
      <label>:                # written to <label>_obs_maps.h5
        input: <dir>          # the CosmoGrid cosmology, relative to the projection root
        config: {...}         # optional, merged onto the base config
        args: {...}           # optional, replaces the registry's args of the same name

A mock's config is the base with its config: block merged on top. Dicts merge recursively, and scalars and lists
replace the base's value. A key the base does not have is an error, so a misspelled key cannot pass silently.

The submission scripts read the registry through the CLI:
    python -m msfm.utils.mocks labels <registry>
    python -m msfm.utils.mocks n_perms <registry>
    python -m msfm.utils.mocks check <registry> --projection <root> [label ...]
"""

import argparse
import copy
import os
import re
import sys
from dataclasses import dataclass

from msfm.utils import input_output, logger

LOGGER = logger.get_logger(__file__)

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# the flags of run_single_postprocessing.py a registry may set, by kind: a switch, a number or a list of numbers
FORWARD_MODEL_ARGS = {
    "with_lensing": "flag",
    "with_clustering": "flag",
    "noiseless": "flag",
    "noise_only": "flag",
    "reduced_shear": "flag",
    "contaminate_survey_systematics": "flag",
    "bta": "float",
    "tomo_Aia": "floats",
    "tomo_bg_metacal": "floats",
    "tomo_bg": "floats",
    "tomo_cg": "floats",
}
REGISTRY_KEYS = {"base", "n_perms", "args", "mocks"}
MOCK_KEYS = {"input", "config", "args"}
# the label is a file name stem, and the Y3 labels contain = and ,
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_.,=+-]+$")


@dataclass(frozen=True)
class Mock:
    label: str
    input: str
    conf: dict
    args: dict
    n_perms: int
    overrides: tuple


def merge_onto(base, override, where="config"):
    """base with override merged on top: dicts recursively, everything else replaced.

    Raises:
        KeyError: for a key the base does not have
        TypeError: when a block would replace a value or a value a block

    Returns:
        tuple: (merged dict, the dotted keys override sets)
    """
    merged = copy.deepcopy(base)
    changed = []
    for key, value in override.items():
        path = f"{where}.{key}"
        if key not in merged:
            raise KeyError(f"{path} is not in the base config, so it would add a key instead of changing one")
        if isinstance(merged[key], dict) != isinstance(value, dict):
            raise TypeError(f"{path} would replace a {type(merged[key]).__name__} with a {type(value).__name__}")
        if isinstance(value, dict):
            merged[key], sub = merge_onto(merged[key], value, path)
            changed += sub
        else:
            merged[key] = copy.deepcopy(value)
            changed.append(path)
    return merged, changed


def _check_args(args, where):
    for name, value in args.items():
        kind = FORWARD_MODEL_ARGS.get(name)
        if kind is None:
            raise KeyError(f"{where}.{name} is not one of the forward-model flags {sorted(FORWARD_MODEL_ARGS)}")
        number = (int, float)
        ok = {
            "flag": isinstance(value, bool),
            "float": isinstance(value, number) and not isinstance(value, bool),
            "floats": isinstance(value, list)
            and all(isinstance(v, number) and not isinstance(v, bool) for v in value),
        }[kind]
        if not ok:
            raise TypeError(f"{where}.{name} = {value!r} is not a {kind}")


def load_registry(path):
    """Reads and validates a registry. Returns the dict."""
    reg = input_output.read_yaml(path)
    if not isinstance(reg, dict) or set(reg) != REGISTRY_KEYS:
        raise KeyError(f"{path} must have exactly the keys {sorted(REGISTRY_KEYS)}")
    if not isinstance(reg["n_perms"], int) or reg["n_perms"] < 1:
        raise ValueError(f"{path}: n_perms = {reg['n_perms']!r} is not a positive integer")
    _check_args(reg["args"] or {}, "args")
    if not isinstance(reg["mocks"], dict) or not reg["mocks"]:
        raise ValueError(f"{path}: mocks is empty")
    for label, entry in reg["mocks"].items():
        if not LABEL_PATTERN.match(label):
            raise ValueError(f"{path}: the label {label!r} is not a plain file name stem")
        entry = entry or {}
        if "input" not in entry or not set(entry) <= MOCK_KEYS:
            raise KeyError(f"{path}: mocks.{label} needs input and may have {sorted(MOCK_KEYS - {'input'})}")
        if os.path.isabs(entry["input"]) or ".." in entry["input"].split("/"):
            raise ValueError(f"{path}: mocks.{label}.input must lie below the projection root, got {entry['input']}")
        _check_args(entry.get("args") or {}, f"mocks.{label}.args")
    return reg


def labels(path):
    return list(load_registry(path)["mocks"])


def resolve(path, label):
    """The mock `label` of the registry at `path`. Returns a Mock."""
    reg = load_registry(path)
    if label not in reg["mocks"]:
        raise KeyError(f"{label} is not a mock of {path}, which has {list(reg['mocks'])}")
    entry = reg["mocks"][label] or {}
    base = input_output.read_yaml(os.path.join(os.path.dirname(os.path.abspath(path)), reg["base"]))
    conf, overrides = merge_onto(base, entry.get("config") or {})
    args = {**(reg["args"] or {}), **(entry.get("args") or {})}
    return Mock(label, entry["input"], conf, args, reg["n_perms"], tuple(overrides))


def check(path, projection, selected=None):
    """Resolves every selected mock and checks what its run would read: the config passes the pipelines' checks, every
    file it names exists, the projection has every permutation, and the bias table has the mock's key.

    Returns:
        list: the problems found, empty if none
    """
    from msfm.utils import configuration, filenames, files

    problems = []
    for label in selected or labels(path):
        try:
            mock = resolve(path, label)
            configuration.print_and_check_modeling_in_config(mock.conf)
            method, bias, _, systematics = files.get_shape_noise(mock.conf)
        except Exception as err:
            problems.append(f"{label}: {type(err).__name__}: {err}")
            continue

        for key, rel in mock.conf["files"].items():
            if isinstance(rel, str) and "/" in rel and not os.path.exists(os.path.join(REPO_DIR, rel)):
                problems.append(f"{label}: files.{key} = {rel} does not exist")

        cosmo_dir = os.path.join(projection, mock.input)
        with_bary = mock.conf["analysis"]["modelling"]["baryonified"]
        perm_dirs = [os.path.join(cosmo_dir, f"perm_{i:04d}") for i in range(mock.n_perms)]
        missing = [
            d for d in perm_dirs if not os.path.isfile(filenames.get_filename_full_maps(d, with_bary=with_bary))
        ]
        if missing:
            problems.append(
                f"{label}: {len(missing)} of {mock.n_perms} permutations have no maps file, e.g. {missing[0]}"
            )

        if method == "count" and bias == "fixed" and "tomo_bg_metacal" not in mock.args:
            key = files.metacal_bias_key(perm_dirs[0])
            try:
                files.read_metacal_bias(key, conf=mock.conf)
            except KeyError:
                problems.append(f"{label}: the bias table {mock.conf['files']['metacal_bias']} has no key {key}")

        LOGGER.info(
            f"{label}: {mock.input}, {mock.n_perms} perms, shape noise {method}/{bias}, systematics {systematics}, "
            f"overrides {list(mock.overrides) or 'none'}, args {mock.args}"
        )
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("labels", "n_perms"):
        sub.add_parser(command).add_argument("registry")
    check_parser = sub.add_parser("check")
    check_parser.add_argument("registry")
    check_parser.add_argument("--projection", required=True, help="the projection root the inputs are relative to")
    check_parser.add_argument("labels", nargs="*", help="the mocks to check, default all")
    args = parser.parse_args(argv)

    if args.command == "labels":
        print("\n".join(labels(args.registry)))
    elif args.command == "n_perms":
        print(load_registry(args.registry)["n_perms"])
    else:
        problems = check(args.registry, args.projection, args.labels or None)
        for problem in problems:
            LOGGER.error(problem)
        sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
