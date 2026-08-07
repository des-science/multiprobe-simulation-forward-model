"""Resolve every repo-relative path in every config and report whether it exists.

Nothing in msfm globs or walks data/ -- every input is reached through an explicit `files:` or
`dirs:` key, resolved as os.path.join(repo_dir, value). That makes the configs the single source of
truth for the data layout, and it makes a moved or renamed file a silent failure for any config
that was not updated with it.

This script is the gate for that: it loads every configs/**/*.yaml, resolves each relative entry
against the repo root, and prints OK/MISSING per (config, section, key). Run it before and after
any change to data/ and compare the OK counts -- the paths may change, the count must not.

Some targets are already MISSING at HEAD and are expected to stay that way (see EXPECTED_MISSING).
Use --strict to exit non-zero when the missing set drifts from that list.

Usage:
    python dev/scripts/data_audit/check_config_paths.py            # full listing
    python dev/scripts/data_audit/check_config_paths.py --summary  # counts only
    python dev/scripts/data_audit/check_config_paths.py --strict   # non-zero on drift
"""

import os
import sys
import glob

import yaml

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# Targets that do not exist at HEAD and are not expected to. They are dangling keys in retired
# configs, plus `peak_binning`, which every live v16/v17 config still names although data/peaks/
# has never existed here -- it is read only by msfm/apps/deprecated/run_peaks.py and v18 dropped
# the key. Do NOT "fix" these by creating the files; that would hide real regressions.
EXPECTED_MISSING = {
    "data/CosmoGridV1_param_dirs.npy",
    "data/CosmoGridV1_perm_dirs.npy",
    "data/DESY3_noise.h5",
    # note the reversed name: this is a typo in configs/deprecated/v8/linear_bias_octant.yaml, a
    # different string from the real DESY3_pixels_v11_octant_512.h5, so it was left frozen
    "data/DESY3_pixels_v11_512_octant.h5",
    "data/deprecated/nside256/DESY3_noise.h5",
    "data/desy3_noise_1024.h5",
    "data/desy3_pixels_fiducial_1024.h5",
    "data/peaks/binning/v6_linear_bias.h5",
    "data/peaks/binning/v7_linear_bias.h5",
    "data/peaks/binning/v9_linear_bias.h5",
    "data/peaks/binning/v10_linear_bias.h5",
    "data/peaks/binning/v11_extended.h5",
}


def iter_path_entries(conf):
    """Yield (section, key, value) for every path-like entry of a config.

    The `dirs:` block nests one level deep in a few retired configs (dirs.connections.san), so the
    walk descends into dict values rather than assuming a flat mapping.

    Args:
        conf (dict): Parsed config.

    Yields:
        tuple: (section, key, value) with value a string.
    """
    for section in ("files", "dirs"):
        block = conf.get(section)
        if not isinstance(block, dict):
            continue
        for key, value in block.items():
            if isinstance(value, str):
                yield section, key, value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, str):
                        yield section, f"{key}.{sub_key}", sub_value


def main(argv):
    summary_only = "--summary" in argv
    strict = "--strict" in argv

    rows, missing = [], set()
    for config in sorted(glob.glob(os.path.join(REPO_DIR, "configs/**/*.yaml"), recursive=True)):
        rel_config = os.path.relpath(config, REPO_DIR)
        try:
            with open(config) as f:
                conf = yaml.safe_load(f)
        except yaml.YAMLError as err:
            print(f"UNPARSED {rel_config}: {err}")
            missing.add(rel_config)
            continue
        if not isinstance(conf, dict):
            continue

        for section, key, value in iter_path_entries(conf):
            # absolute values point at scratch/store or another cluster, not at the repo
            if os.path.isabs(value):
                continue
            exists = os.path.exists(os.path.join(REPO_DIR, value))
            rows.append((exists, rel_config, section, key, value))
            if not exists:
                missing.add(value)

    if not summary_only:
        for exists, rel_config, section, key, value in rows:
            status = "OK     " if exists else "MISSING"
            print(f"{status}  {rel_config:52s}  {section}.{key:24s}  {value}")

    n_ok = sum(1 for row in rows if row[0])
    print(f"\n{n_ok}/{len(rows)} entries resolve, {len(missing)} distinct missing targets")

    unexpected = sorted(missing - EXPECTED_MISSING)
    if unexpected:
        print("\nUNEXPECTED missing targets (regressions):")
        for value in unexpected:
            print(f"  {value}")
    stale = sorted(EXPECTED_MISSING - missing)
    if stale:
        print("\nEXPECTED_MISSING entries no longer referenced (prune them from this script):")
        for value in stale:
            print(f"  {value}")

    if strict and unexpected:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
