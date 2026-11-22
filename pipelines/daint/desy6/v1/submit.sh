#!/bin/bash
# Submits the DES Y6 v1 tfrecord production and its validation on daint: a packed array (one exclusive GH200 node per
# element, $SLOTS cosmologies per node on its CPUs) and the merge after it (afterok). Port of
# ../../../perlmutter/desy3/v18/packed/submit.sh, with the same interface.
#
# Usage:
#   DRY_RUN=1 ./submit.sh grid            # preflight + print the sbatch calls, submit nothing
#   ./submit.sh grid                      # fresh run: submit every index
#   ./submit.sh --rerun grid              # submit only the indices whose .tfrecord is missing or a stub
#   ./submit.sh validate                  # dev/scripts/validation/grid_tfrecords.py on every .tfrecord, then its merge
#   ./submit.sh --rerun validate          # only the indices without a scan
#   ./submit.sh obs <label>|all           # the mock observations of $MOCKS, one array and one merge per mock
#   ./submit.sh --rerun obs <label>|all   # only the permutations without an output, for a mock not merged yet
#
# Before any sbatch, a login-node preflight checks that every submitted index has all its projected files under
# $DIR_IN and that every file the config names exists (files.metacal_bias needs step sc_bias first). For validate it
# also checks that the index's .tfrecord exists. For obs it runs `python -m msfm.utils.mocks check` on every selected
# mock before submitting any: config, files, permutations and bias-table key.
#
# The mocks are defined in $MOCKS alone (msfm/utils/mocks.py); this script only knows their labels.
#
# The input projection ($DIR_IN) and the output root ($DATA_ROOT) are those of the locations.yaml next to $CONFIG
# (msfm/utils/locations.py); every output directory below lies under $DATA_ROOT.
#
# Overrides (environment):
#   CONFIG DIR_IN DATA_ROOT                      the msfm config, the projection and the output root
#   DIR_OUT PACKED_LOGS JOB_NAME                 per-chain paths and names, see below (obs ignores JOB_NAME)
#   MOCKS                                        obs: the mock registry
#   TFR_DIR GRID_CLS REPRODUCE_DIR               validate: the .tfrecords, the merged Cls and a --debug rerun of grid
#   SLOTS OMP                                    tasks per node and cores per task. OMP must divide 72 (one Grace)
#   NODE_TIME MERGE_TIME PARTITION               per-element wall clock, merge wall clock, partition (normal)
#   ARRAY=<spec>                                 submit only these array elements, e.g. ARRAY=0 runs one node
#   ARRAY_THROTTLE=<n>                           at most n elements at once (default 25), staggers the noise-file loads
#   EXTRA_ARGS                                   appended to the app's arguments, e.g. --debug
#   SKIP_MERGE=1                                 do not submit the merge
#   MERGE_AFTER=<jobid>                          submit only the merge, after that array (e.g. when sbatch refused it)
#   DRY_RUN=1                                    preflight and print, submit nothing

set -euo pipefail
cd "$(dirname "$0")"

REPO=$(cd ../../../.. && pwd)
# absolute, since Slurm runs the batch script from a spool copy
DRIVER=$REPO/pipelines/common/run_packed.py
NODE_SCRIPT=$REPO/pipelines/daint/common/packed_node.slurm
export TASK_SCRIPT=$REPO/pipelines/daint/common/packed_task.sh

# login-node environment of the preflight and of the msfm CLIs (locations, mocks)
UENV=pytorch/v2.9.1:v2
PY=$HOME/dlss/torch_env/bin/python

CONFIG=${CONFIG:-$REPO/configs/desy6/v1/default.yaml}
MOCKS=${MOCKS:-$REPO/configs/desy6/v1/mock_observations.yaml}
# the projection (input) and the config's output root come from the locations.yaml next to $CONFIG
LOCATIONS=$(uenv run --view=default "$UENV" -- "$PY" -m msfm.utils.locations "$CONFIG") \
    || { echo "cannot resolve the locations of $CONFIG" >&2; exit 1; }
location() { sed -n "s|^$1=||p" <<< "$LOCATIONS"; }
DIR_IN=${DIR_IN:-$(location projection)}
DATA_ROOT=${DATA_ROOT:-$(location data_dir)}
if [ -z "$DIR_IN" ] || [ -z "$DATA_ROOT" ]; then
    echo "the locations of $CONFIG give no projection or data_dir; set DIR_IN and DATA_ROOT" >&2
    exit 1
fi
PACKED_LOGS=${PACKED_LOGS:-$DATA_ROOT/packed_logs}
PARTITION=${PARTITION:-normal}
ARRAY_THROTTLE=${ARRAY_THROTTLE:-25}
# a GH200 node: 4 Grace CPUs x 72 cores
NODE_CPUS=288
EXTRA_ARGS=${EXTRA_ARGS:-}
DRY_RUN=${DRY_RUN:-}
ARRAY_OVERRIDE=${ARRAY:-}
SKIP_MERGE=${SKIP_MERGE:-}
MERGE_AFTER=${MERGE_AFTER:-}

# SLOTS x OMP: 24 x 12 = 6 tasks per Grace, 0.031 node-hours per cosmology against 0.047 at 12 x 24, 10.5 GB peak
# per task; NODE_TIME is twice the measured 45 min (README.md)
chain_cfg() {
    case "$1" in
        grid)
            APP=$REPO/msfm/apps/run_grid_postprocessing.py
            JOB_NAME=${JOB_NAME:-tfr_grid_desy6_v1}
            DIR_OUT=${DIR_OUT:-$DATA_ROOT/tfrecords/grid}
            SIMSET=grid
            N_TOTAL=2500
            SLOTS=${SLOTS:-24}
            OMP=${OMP:-12}
            NODE_TIME=${NODE_TIME:-01:30:00}
            MERGE_TIME=${MERGE_TIME:-06:00:00}
            APP_ARGS="--n_files=$N_TOTAL --dir_out=$DIR_OUT --config=$CONFIG --dir_in=$DIR_IN --cosmogrid_version=1.1"
            ;;
        validate)
            # reads one .tfrecord (6.7 GB) per task, so I/O-bound: many tasks with few cores each
            APP=$REPO/dev/scripts/validation/grid_tfrecords.py
            JOB_NAME=${JOB_NAME:-tfr_validate_desy6_v1}
            DIR_OUT=${DIR_OUT:-$DATA_ROOT/validation}
            TFR_DIR=${TFR_DIR:-$DATA_ROOT/tfrecords/grid}
            GRID_CLS=${GRID_CLS:-$DATA_ROOT/cls/grid_cls.h5}
            REPRODUCE_DIR=${REPRODUCE_DIR:-$DATA_ROOT/validation/reproduce/debug}
            N_TOTAL=2500
            SLOTS=${SLOTS:-72}
            OMP=${OMP:-4}
            NODE_TIME=${NODE_TIME:-01:00:00}
            MERGE_TIME=${MERGE_TIME:-01:00:00}
            APP_ARGS="--config=$CONFIG --dir_tfrecords=$TFR_DIR --dir_out=$DIR_OUT --dir_in=$DIR_IN --grid_cls=$GRID_CLS"
            # the merge compares the rerun if it exists at submission
            if [ -d "$REPRODUCE_DIR" ]; then APP_ARGS+=" --reproduce_dir=$REPRODUCE_DIR"; fi
            ;;
        obs)
            # one mock of $MOCKS: its permutations are the indices, and the merge writes <label>_obs_maps.h5. One node;
            # fiducial_bench took 3:42 and its merge 26 s (README.md), so both wall clocks have a wide margin
            APP=$REPO/msfm/apps/run_single_postprocessing.py
            JOB_NAME=obs_desy6_v1_$LABEL
            DIR_OUT=${DIR_OUT:-$DATA_ROOT/obs}
            N_TOTAL=$N_PERMS
            SLOTS=${SLOTS:-24}
            OMP=${OMP:-12}
            NODE_TIME=${NODE_TIME:-00:30:00}
            MERGE_TIME=${MERGE_TIME:-00:10:00}
            APP_ARGS="--mocks=$MOCKS --mock=$LABEL --dir_in=$DIR_IN --dir_out=$DIR_OUT"
            ;;
        fiducial)
            echo "fiducial: DES Y6 has no fiducial projection, and none is planned" >&2
            return 1
            ;;
        *) echo "unknown chain '$1'" >&2; return 1 ;;
    esac
    if (( 72 % OMP != 0 )); then
        echo "OMP=$OMP does not divide the 72 cores of a Grace, so a task would span two NUMA domains" >&2
        return 1
    fi
}

preflight() {
    local index_file=$1 chain=$2
    echo "[preflight] projected files under $DIR_IN and the config's files, on the login node"
    uenv run --view=default "$UENV" -- "$PY" -I - "$REPO" "$CONFIG" "$DIR_IN" "$index_file" "$chain" "${TFR_DIR:-}" \
        <<'EOF'
import os
import sys

import h5py
import yaml

repo, config, dir_in, index_file, chain, tfr_dir = sys.argv[1:]
with open(config) as f:
    conf = yaml.safe_load(f)
problems = []

# every file the config names, relative to the repo (the files: section also holds non-path values)
for key, path in conf["files"].items():
    if isinstance(path, str) and "/" in path and not os.path.exists(os.path.join(repo, path)):
        problems.append(f"files.{key} = {path} does not exist")

# the grid app reads <dir_in>/grid/<path_par>/perm_XXXX/projected_probes_maps_v11dmb.h5, the index-th cosmology
with h5py.File(os.path.join(repo, conf["files"]["meta_info"]), "r") as f:
    path_par = [p.decode().strip("/") for p in f["parameters/grid"]["path_par"]]
n_perms = conf["analysis"]["grid"]["n_perms_per_cosmo"]
indices = [int(line) for line in open(index_file) if line.strip()]
missing = []
for index in indices:
    for i_perm in range(n_perms):
        file = os.path.join(dir_in, "grid", path_par[index], f"perm_{i_perm:04d}", "projected_probes_maps_v11dmb.h5")
        if not os.path.isfile(file):
            missing.append(file)
if missing:
    problems.append(f"{len(missing)} of {len(indices) * n_perms} projected files are missing, e.g. {missing[0]}")

# validate reads the grid app's output, <survey>_grid_dm{b,o}_XXXX.tfrecord, and a short stub is a failed write
if chain == "validate":
    model = "dmb" if conf["analysis"]["modelling"]["baryonified"] else "dmo"
    tfrecords = [os.path.join(tfr_dir, f"{conf['survey']['name']}_grid_{model}_{i:04d}.tfrecord") for i in indices]
    bad = [t for t in tfrecords if not os.path.isfile(t) or os.path.getsize(t) < 2**20]
    if bad:
        problems.append(f"{len(bad)} of {len(indices)} .tfrecords are missing or stubs, e.g. {bad[0]}")

print(f"[preflight] {len(indices)} indices x {n_perms} perms checked")
for problem in problems:
    print(f"[preflight] {problem}", file=sys.stderr)
sys.exit(1 if problems else 0)
EOF
}

# indices whose output is complete: grid's .tfrecord (a task that dies mid-run can leave a short stub, real files are
# GBs), validate's scan (written under a temporary name, then renamed)
done_indices() {
    case "$1" in
        grid)
            find "$DIR_OUT" -maxdepth 1 -name "*_${SIMSET}_dm?_*.tfrecord" -size +1M -printf '%f\n' 2>/dev/null \
                | sed -E 's/.*_dm._0*([0-9]+)\.tfrecord$/\1/' | sort -n -u
            ;;
        validate)
            find "$DIR_OUT/scan" -maxdepth 1 -name 'index_*.npz' -printf '%f\n' 2>/dev/null \
                | sed -E 's/^index_0*([0-9]+)\.npz$/\1/' | sort -n -u
            ;;
        obs)
            # the merge deletes the per-permutation files, so a merged mock counts as complete
            if [ -f "$DIR_OUT/${LABEL}_obs_maps.h5" ]; then seq 0 $((N_TOTAL - 1)); return; fi
            find "$DIR_OUT" -maxdepth 1 -name "${LABEL}_obs_maps_*.h5" -size +1k -printf '%f\n' 2>/dev/null \
                | sed -E 's/.*_obs_maps_0*([0-9]+)\.h5$/\1/' | sort -n -u
            ;;
    esac
}

# the registry CLI, in the preflight environment
mocks_cli() {
    uenv run --view=default "$UENV" -- "$PY" -m msfm.utils.mocks "$@"
}

# obs: every selected mock is checked before any is submitted, then each gets its own array and merge
submit_obs() {
    local selection=$1 rerun=$2
    local all labels label
    all=$(mocks_cli labels "$MOCKS") || { echo "obs: cannot read the mocks of $MOCKS" >&2; return 1; }
    if [ "$selection" = all ] && [ -n "$MERGE_AFTER" ]; then
        echo "obs: MERGE_AFTER names one array, so it takes one mock, not all" >&2
        return 1
    elif [ "$selection" = all ]; then
        labels=$all
    elif [ -n "$selection" ] && grep -qxF -- "$selection" <<< "$all"; then
        labels=$selection
    else
        echo "obs: '$selection' is not a mock of $MOCKS, which has: $(echo $all) (or: all)" >&2
        return 1
    fi
    N_PERMS=$(mocks_cli n_perms "$MOCKS")
    echo "[preflight] the mocks $(echo $labels) of $MOCKS against $DIR_IN, on the login node"
    # shellcheck disable=SC2086
    mocks_cli check "$MOCKS" --projection "$DIR_IN" $labels || { echo "[preflight] failed, nothing submitted" >&2; return 1; }
    for label in $labels; do
        LABEL=$label
        submit_chain obs "$rerun"
    done
}

submit_chain() {
    local chain=$1 rerun=$2
    local tag=$chain
    [ "$chain" = obs ] && tag="obs $LABEL"
    chain_cfg "$chain"
    local index_file=$PACKED_LOGS/${JOB_NAME}_indices.txt
    mkdir -p "$PACKED_LOGS" "$DIR_OUT"

    if [ -n "$MERGE_AFTER" ]; then
        # only the merge; the index file stays as the running array reads it
        export SLOTS OMP APP JOB_NAME PACKED_LOGS DRIVER INDEX_FILE=$index_file FUNCTION=main
        export APP_ARGS="$APP_ARGS $EXTRA_ARGS"
        submit_merge "$chain" "$tag" "$MERGE_AFTER"
        return
    fi

    if [ -n "$rerun" ]; then
        local present
        present=$(done_indices "$chain")
        seq 0 $((N_TOTAL - 1)) | grep -vxF -f <(echo "$present") > "$index_file" || true
    else
        seq 0 $((N_TOTAL - 1)) > "$index_file"
    fi

    local n_idx n_elements
    n_idx=$(grep -c . "$index_file" || true)
    n_elements=$(( (n_idx + SLOTS - 1) / SLOTS ))
    echo "[$tag] ${rerun:+RERUN }$n_idx indices, $SLOTS per node x $OMP cores -> $n_elements node jobs ($NODE_TIME each)"
    if [ "$n_idx" -eq 0 ]; then echo "[$tag] nothing to do"; return 0; fi

    # element k processes lines [k*SLOTS, (k+1)*SLOTS) of the index file, so ARRAY only checks those
    local array_spec=${ARRAY_OVERRIDE:-0-$((n_elements - 1))}
    local check_file=$index_file
    if [ -n "$ARRAY_OVERRIDE" ]; then
        check_file=$PACKED_LOGS/${JOB_NAME}_indices_checked.txt
        : > "$check_file"
        local element
        for element in $(echo "$ARRAY_OVERRIDE" | tr ',' ' '); do
            local lo=${element%-*} hi=${element#*-}
            sed -n "$((lo * SLOTS + 1)),$(((hi + 1) * SLOTS))p" "$index_file" >> "$check_file"
        done
        echo "[$tag] ARRAY override: elements $array_spec of 0-$((n_elements - 1))"
    fi
    # obs ran its own preflight on every selected mock in submit_obs
    if [ "$chain" != obs ]; then preflight "$check_file" "$chain"; fi

    export SLOTS OMP APP JOB_NAME PACKED_LOGS DRIVER INDEX_FILE=$index_file FUNCTION=main
    export APP_ARGS="$APP_ARGS $EXTRA_ARGS"
    local packed_cmd=(sbatch --parsable --partition="$PARTITION" --array="$array_spec%$ARRAY_THROTTLE"
        --time="$NODE_TIME" --job-name="${JOB_NAME}_packed" --output="$PACKED_LOGS/%x_%A_%a.out" --export=ALL
        "$NODE_SCRIPT")
    local packed_id
    if [ -n "$DRY_RUN" ]; then
        echo "DRY RUN + ${packed_cmd[*]}"
        echo "DRY RUN   APP_ARGS=$APP_ARGS"
        packed_id=DRYRUN
    else
        packed_id=$("${packed_cmd[@]}")
    fi
    echo "[$tag] packed array: $packed_id"
    submit_merge "$chain" "$tag" "$packed_id"
}

# merge: the whole node, one process, only after every element of the array succeeded. The single-cosmology merge
# places each permutation by its index, so it gets all of them instead of the default 0
submit_merge() {
    local chain=$1 tag=$2 packed_id=$3
    if [ "$chain" = obs ]; then
        export TASKS
        TASKS=$(seq -s, 0 $((N_TOTAL - 1)))
    fi
    local merge_cmd=(sbatch --parsable --partition="$PARTITION" --dependency="afterok:$packed_id" --time="$MERGE_TIME"
        --job-name="${JOB_NAME}_merge" --output="$PACKED_LOGS/%x_%j.out"
        --export="ALL,FUNCTION=merge,SLOTS=1,OMP=$NODE_CPUS" "$NODE_SCRIPT")
    local merge_id
    if [ -n "$SKIP_MERGE" ]; then
        echo "[$tag] SKIP_MERGE set; submit the merge later with MERGE_AFTER=$packed_id and the same arguments"
    elif [ -n "$DRY_RUN" ]; then
        echo "DRY RUN + ${merge_cmd[*]}"
    elif ! merge_id=$("${merge_cmd[@]}"); then
        echo "[$tag] the merge was NOT submitted; once sbatch accepts it, rerun with MERGE_AFTER=$packed_id" >&2
        return 1
    else
        echo "[$tag] merge job: $merge_id (afterok:$packed_id)"
    fi
}

RERUN=
[ "${1:-}" = --rerun ] && { RERUN=1; shift; }
case "${1:-}" in
    grid|fiducial|validate) submit_chain "$1" "$RERUN" ;;
    obs) submit_obs "${2:-}" "$RERUN" ;;
    *) echo "usage: [DRY_RUN=1] $0 [--rerun] grid|validate|obs <label>|obs all" >&2; exit 1 ;;
esac
