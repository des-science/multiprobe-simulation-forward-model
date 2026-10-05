#!/usr/bin/env bash
# Zero-dependency snapshot of every data/ path named by a config, and whether it resolves.
#
# The companion check_config_paths.py is the real check, but it needs PyYAML, which lives inside
# the TensorFlow container and not on the login node. This script runs anywhere, so a before/after
# pair around a refactor is guaranteed to have been taken under identical conditions.
#
# Usage:
#   dev/scripts/data_audit/gate.sh > /tmp/gate.before      # before the change
#   dev/scripts/data_audit/gate.sh > /tmp/gate.after       # after
#   diff /tmp/gate.before /tmp/gate.after
#
# Pass criterion: the OK count is unchanged (every target that resolved still resolves) and every
# line in the diff is a path rewrite you intended. Several targets are ALREADY missing at HEAD --
# data/peaks/**, the nside-1024 pair, the v5/v6/v7 grid .npy -- so a non-empty MISSING set is
# expected and is not a regression by itself.
set -u
cd "$(dirname "$0")/../../.." || exit 1

grep -rhoE 'data/[A-Za-z0-9._/+-]+' configs/ --include=*.yaml |
    sed 's/"$//' | sort -u |
    while read -r f; do
        if [ -e "$f" ]; then echo "OK      $f"; else echo "MISSING $f"; fi
    done | sort
