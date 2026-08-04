#!/bin/bash
# Clean-start submission of the v17 baseline postprocessing on regular QOS (whole CPU nodes),
# packing many tasks per node. This is the single source of truth for the run: it holds all
# per-chain configuration and exports it to the generic executors packed_node.slurm /
# merge_node.slurm. Nothing runs until this script is invoked.
#
# Usage:
#   DRY_RUN=1 ./submit.sh both        # print the sbatch calls, touch nothing
#   ./submit.sh grid|fiducial|both    # fresh run (archives any old done.dat first)
#   ./submit.sh --rerun grid|both     # resubmit only the indices missing from done.dat
#
# Why packing: regular QOS is node-exclusive, so a bare esub jobarray would burn a whole
# 128-core node on one ~15 GB task. The forward model is memory-bound, not compute-bound
# (see run_*_postprocessing.py resources()), so on a node we pack by MEMORY and split the
# 256 hyperthreads across the slots. Measured peak RSS: grid 14.6 GB, fiducial 17.2 GB on a
# ~503 GB node -> the SLOTS below leave comfortable headroom.

set -euo pipefail
cd "$(dirname "$0")"

# --- fixed paths -----------------------------------------------------------------------------
REPO=/global/homes/a/athomsen/multiprobe-simulation-forward-model
CONFIG=$REPO/configs/v17/baseline.yaml
DIR_IN=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary
DATA_ROOT=/pscratch/sd/a/athomsen/v11desy3/v17/baseline/tfrecords
LOG_DIR=/pscratch/sd/a/athomsen/run_files/v17/esub_logs
PACKED_LOGS=/pscratch/sd/a/athomsen/run_files/v17/packed_logs

# --- per-chain configuration (single source of truth) ---------------------------------------
# SLOTS      : tasks packed per node   (memory-bound: SLOTS * peak_RSS + headroom <= ~503 GB)
# OMP        : threads per task        (~256 / SLOTS; compute barely scales, so this is slack)
# N_TOTAL    : number of .tfrecord files == esub tasks (must match --n_files)
# NODE_TIME  : walltime cap per packed node job. Generous headroom over the measured max task
#              time (grid 1h45, fidu 1h22) since NERSC charges actual elapsed, not the cap, and
#              a node killed at the cap fails the element and blocks the afterok merge.
# MERGE_TIME : walltime cap for the single merge job (unmeasured; kept generous)
chain_cfg() {
    case "$1" in
        grid)
            APP=$REPO/msfm/apps/run_grid_postprocessing.py
            JOB_NAME=tfr_grid_v17
            DIR_OUT=$DATA_ROOT/grid
            N_TOTAL=2500; SLOTS=24; OMP=10; NODE_TIME=06:00:00; MERGE_TIME=08:00:00
            ;;
        fiducial)
            APP=$REPO/msfm/apps/run_fiducial_postprocessing.py
            JOB_NAME=tfr_fidu_v17
            DIR_OUT=$DATA_ROOT/fiducial
            N_TOTAL=1000; SLOTS=20; OMP=12; NODE_TIME=04:00:00; MERGE_TIME=04:00:00
            ;;
        *) echo "unknown chain '$1'" >&2; return 1 ;;
    esac
}

DRY_RUN=${DRY_RUN:-}
run() { if [ -n "$DRY_RUN" ]; then echo "DRY RUN + $*"; else "$@"; fi; }

submit_chain() {
    local chain=$1 rerun=$2
    chain_cfg "$chain"
    local done_file=$LOG_DIR/${JOB_NAME}_done.dat
    local index_file=$PACKED_LOGS/${JOB_NAME}_indices.txt

    mkdir -p "$LOG_DIR" "$PACKED_LOGS" "$DIR_OUT"

    # --- build the index list (0-based esub task indices) ------------------------------------
    if [ -n "$rerun" ]; then
        # missing = all - already-done (successful appends in done.dat)
        seq 0 $((N_TOTAL - 1)) \
            | grep -vxF -f <(grep -E '^[0-9]+$' "$done_file" 2>/dev/null | sort -n -u) \
            > "$index_file" || true
    else
        # fresh run: archive any stale done.dat so the merge count and reruns start clean
        if [ -s "$done_file" ]; then
            run mv "$done_file" "$done_file.$(date +%Y%m%dT%H%M%S).bak"
        fi
        seq 0 $((N_TOTAL - 1)) > "$index_file"
    fi

    local n_idx n_elements
    n_idx=$(wc -l < "$index_file")
    n_elements=$(( (n_idx + SLOTS - 1) / SLOTS ))
    echo "[$chain] ${rerun:+RERUN }$n_idx indices, $SLOTS/node, ${OMP} threads/task -> $n_elements node jobs (${NODE_TIME} each)"
    if [ "$n_idx" -eq 0 ]; then echo "[$chain] nothing to do"; return 0; fi

    # --- packed array ------------------------------------------------------------------------
    # everything the generic packed_node.slurm needs travels through the environment
    export CHAIN=$chain SLOTS OMP APP JOB_NAME LOG_DIR PACKED_LOGS INDEX_FILE=$index_file
    export APP_ARGS="--n_files=$N_TOTAL --dir_out=$DIR_OUT --config=$CONFIG --dir_in=$DIR_IN --cosmogrid_version=1.1"
    local packed_id
    if [ -n "$DRY_RUN" ]; then
        echo "DRY RUN + sbatch --parsable --array=0-$((n_elements - 1)) --time=$NODE_TIME" \
             "--job-name=${JOB_NAME}_packed --export=ALL packed_node.slurm"
        packed_id=DRYRUN
    else
        packed_id=$(sbatch --parsable \
            --array=0-$((n_elements - 1)) \
            --time="$NODE_TIME" \
            --job-name="${JOB_NAME}_packed" \
            --export=ALL \
            packed_node.slurm)
    fi
    echo "[$chain] packed array: $packed_id"

    # --- merge (afterok: only runs if every packed element succeeded => all files present) ---
    export MERGE_JOB=${JOB_NAME}_merge
    if [ -n "$DRY_RUN" ]; then
        echo "DRY RUN + sbatch --parsable --dependency=afterok:$packed_id --time=$MERGE_TIME" \
             "--job-name=$MERGE_JOB --export=ALL merge_node.slurm"
    else
        local merge_id
        merge_id=$(sbatch --parsable \
            --dependency="afterok:$packed_id" \
            --time="$MERGE_TIME" \
            --job-name="$MERGE_JOB" \
            --export=ALL \
            merge_node.slurm)
        echo "[$chain] merge job: $merge_id (afterok:$packed_id)"
    fi
}

# --- arg parsing -----------------------------------------------------------------------------
RERUN=
[ "${1:-}" = --rerun ] && { RERUN=1; shift; }
case "${1:-}" in
    grid)     submit_chain grid "$RERUN" ;;
    fiducial) submit_chain fiducial "$RERUN" ;;
    both)     submit_chain grid "$RERUN"; submit_chain fiducial "$RERUN" ;;
    *)        echo "usage: [DRY_RUN=1] $0 [--rerun] grid|fiducial|both" >&2; exit 1 ;;
esac
