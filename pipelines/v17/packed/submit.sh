#!/bin/bash
# Clean-start submission of the v17 baseline postprocessing on regular QOS (whole CPU nodes),
# packing many tasks per node -- WITHOUT esub. This is the single source of truth for the run: it
# holds all per-chain configuration and exports it to the generic executors in
# ../../common/packed/ (packed_node.slurm / merge_node.slurm, which call run_packed.py -- a
# ~30-line import-and-call driver shared by every pipeline version). Nothing runs until this
# script is invoked.
#
# Usage:
#   DRY_RUN=1 ./submit.sh both            # print the sbatch calls, touch nothing
#   ./submit.sh grid|fiducial|both        # fresh run: submit every index
#   ./submit.sh --rerun grid|fiducial|both  # submit only indices whose .tfrecord is missing
#
# Why no esub: esub's value is the done.dat / rerun_missing bookkeeping for a farm of
# independent shared-QOS jobarray tasks. On packed full nodes that layer buys nothing, so we
# drop it. Completion is tracked by the output .tfrecord files themselves: --rerun resubmits any
# index whose .tfrecord is absent (delete a failed task's 0-byte/short stub first, see README).
#
# Why packing: regular QOS is node-exclusive, so a bare jobarray would burn a whole 128-core node
# on one ~14 GB task. We pack many tasks per node, each pinned to a NUMA domain. The binding limit
# is memory BANDWIDTH, not capacity: the healpy SHTs saturate a domain's controllers at ~3 tasks,
# so we pack 3/domain (24 slots). Denser fits in RAM but costs ~20% more node-hours (see chain_cfg).

set -euo pipefail
cd "$(dirname "$0")"

# --- fixed paths -----------------------------------------------------------------------------
REPO=/global/homes/a/athomsen/multiprobe-simulation-forward-model
# the generic executors, shared with v18 and any future version -- see ../../common/packed/README.md
COMMON_PACKED=$REPO/pipelines/common/packed
# absolute path to the driver, exported to the slurm scripts. It must be absolute: Slurm copies
# the batch script into /var/spool/slurmd/<job>, so the executors cannot locate it themselves.
DRIVER=$COMMON_PACKED/run_packed.py
# CONFIG / DATA_ROOT default to the v17 baseline but can be overridden from the environment for a
# parallel run (e.g. the B-mode Cls study: CONFIG_OVERRIDE=configs/v17/baseline_bmode.yaml with a
# separate DATA_ROOT_OVERRIDE so the completed baseline tfrecords are never touched). JOB_SUFFIX is
# appended to every job/log/index-file name so the two runs never collide in squeue or on disk.
CONFIG=${CONFIG_OVERRIDE:-$REPO/configs/v17/baseline.yaml}
DIR_IN=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary
DATA_ROOT=${DATA_ROOT_OVERRIDE:-/pscratch/sd/a/athomsen/v11desy3/v17/baseline/tfrecords}
PACKED_LOGS=/pscratch/sd/a/athomsen/run_files/v17/packed_logs
JOB_SUFFIX=${JOB_SUFFIX:-}

# --- per-chain configuration (single source of truth) ---------------------------------------
# SLOTS      : tasks packed per node. A MULTIPLE OF 8 so packed_node.slurm places SLOTS/8 tasks on
#              each of the 8 NUMA domains evenly. Set to the memory-BANDWIDTH knee (3 tasks/domain
#              = 24), NOT the RAM ceiling: a 4th task/domain fits but raises node-hours ~20%.
# OMP        : threads per task, ~= 32 / (SLOTS/8) rounded down (24 slots -> 3/domain -> 10),
#              splitting each domain's 32 threads across its tasks. Also bounds TF/BLAS/OpenMP.
# N_TOTAL    : number of .tfrecord files == number of indices (must match --n_files)
# SIMSET     : filename tag used to detect already-produced files for --rerun
# NODE_TIME  : walltime cap per packed node job -- generous headroom over the measured node time
#              (grid ~85 min, fiducial ~38 min): NERSC charges actual elapsed, not the cap, and a
#              node killed at the cap fails the element and blocks the afterok merge.
# MERGE_TIME : walltime cap for the single merge job (grid ~2h47, fiducial ~32 min measured)
chain_cfg() {
    case "$1" in
        grid)
            APP=$REPO/msfm/apps/run_grid_postprocessing.py
            JOB_NAME=tfr_grid_v17$JOB_SUFFIX
            DIR_OUT=$DATA_ROOT/grid
            SIMSET=grid
            # space-separated indices whose CosmoGrid INPUT maps are known permanently corrupt (no
            # good copy) -- never submitted, so no task can fail on them and block the afterok merge.
            # None known for grid. Override at call time with SKIP_INDICES="..." (or "" to force all).
            SKIP_INDICES_DEFAULT=""
            # 3/domain x ~13.5 GB = 40 GB (>20 GB headroom). Benchmarked 24 vs 32 slots under the
            # same fleet contention: 24-slot 85.5 min/node -> ~150 nh; 32-slot 142.8 min -> ~188 nh
            # (~20% dearer). The first v17 run used 32/8 before this was measured -- see README.
            N_TOTAL=2500; SLOTS=24; OMP=10; NODE_TIME=04:00:00; MERGE_TIME=08:00:00
            ;;
        fiducial)
            APP=$REPO/msfm/apps/run_fiducial_postprocessing.py
            JOB_NAME=tfr_fidu_v17$JOB_SUFFIX
            DIR_OUT=$DATA_ROOT/fiducial
            SIMSET=fiducial
            # perm_0209 of the cosmo_delta_bary_nu_p perturbation is a truncated input HDF5 (89 MB vs
            # 606 MB, "bad object header version number", no good copy) shared by every fiducial run,
            # so index 209 can only ever fail and DependencyNeverSatisfied-block the merge (it did in
            # the baseline run too). Never submit it. Override with SKIP_INDICES="..." (or "" to force).
            SKIP_INDICES_DEFAULT="209"
            # 3/domain x ~11.5 GB = 35 GB (huge headroom). 24-slot 37.5 min/node -> ~26 nh; 32-slot
            # doubled to 78 min -> ~42 nh (bandwidth-bound, same as grid). 3/domain is the knee.
            N_TOTAL=1000; SLOTS=24; OMP=10; NODE_TIME=04:00:00; MERGE_TIME=04:00:00
            ;;
        *) echo "unknown chain '$1'" >&2; return 1 ;;
    esac
}

DRY_RUN=${DRY_RUN:-}
# Staging overrides (optional; for benchmarking or partial reruns):
#   ARRAY=<spec>   submit only these array elements instead of the full 0-(N-1), e.g. ARRAY=0
#                  runs one node. The index_file is still built fresh over ALL indices, so the
#                  slices line up -- element k always processes indices [k*SLOTS, (k+1)*SLOTS).
#   SKIP_MERGE=1   do not submit the merge job (submit it yourself once every stage is done, via
#                  sbatch --dependency=afterok:<id1>:<id2> ... merge_node.slurm with this env).
ARRAY_OVERRIDE=${ARRAY:-}
SKIP_MERGE=${SKIP_MERGE:-}

submit_chain() {
    local chain=$1 rerun=$2
    chain_cfg "$chain"
    local index_file=$PACKED_LOGS/${JOB_NAME}_indices.txt

    mkdir -p "$PACKED_LOGS" "$DIR_OUT"

    # --- build the index list (0-based indices, one .tfrecord file each) ----------------------
    if [ -n "$rerun" ]; then
        # missing = all indices whose output .tfrecord is absent OR a stub. The file is the done
        # marker, but a task that dies mid-run can leave a 0-byte/short stub (the writer opens
        # before the work finishes -- this is how index 209 failed), so count only files > 1 MB as
        # done (real ones are GBs). Strip the 4-digit index out of DESy3_grid_dmb_0007.tfrecord -> 7.
        local present
        present=$(find "$DIR_OUT" -maxdepth 1 -name "*_${SIMSET}_dm?_*.tfrecord" -size +1M -printf '%f\n' 2>/dev/null \
            | sed -E 's/.*_dm._0*([0-9]+)\.tfrecord$/\1/' | sort -n -u)
        seq 0 $((N_TOTAL - 1)) \
            | grep -vxF -f <(echo "$present") \
            > "$index_file" || true
    else
        seq 0 $((N_TOTAL - 1)) > "$index_file"
    fi

    # drop known permanently-corrupt-input indices (see SKIP_INDICES_DEFAULT in chain_cfg). Applied to
    # both fresh and --rerun lists so these indices are never (re)submitted and can't block the merge.
    local skip=${SKIP_INDICES-$SKIP_INDICES_DEFAULT}
    if [ -n "$skip" ]; then
        local n_before; n_before=$(grep -c . "$index_file" || true)
        grep -vxF -f <(printf '%s\n' $skip) "$index_file" > "$index_file.tmp" && mv "$index_file.tmp" "$index_file"
        echo "[$chain] skipping known-bad-input indices ($skip): $n_before -> $(grep -c . "$index_file" || true) indices"
    fi

    local n_idx n_elements
    n_idx=$(grep -c . "$index_file" || true)
    n_elements=$(( (n_idx + SLOTS - 1) / SLOTS ))
    echo "[$chain] ${rerun:+RERUN }$n_idx indices, $SLOTS/node, ${OMP} threads/task -> $n_elements node jobs (${NODE_TIME} each)"
    if [ "$n_idx" -eq 0 ]; then echo "[$chain] nothing to do"; return 0; fi

    # --- packed array ------------------------------------------------------------------------
    # everything the generic packed_node.slurm needs travels through the environment
    export SLOTS OMP APP JOB_NAME PACKED_LOGS DRIVER INDEX_FILE=$index_file
    export APP_ARGS="--n_files=$N_TOTAL --dir_out=$DIR_OUT --config=$CONFIG --dir_in=$DIR_IN --cosmogrid_version=1.1"
    local array_spec=${ARRAY_OVERRIDE:-0-$((n_elements - 1))}
    [ -n "$ARRAY_OVERRIDE" ] && echo "[$chain] ARRAY override: submitting only elements $array_spec (of 0-$((n_elements - 1)))"
    local packed_id
    if [ -n "$DRY_RUN" ]; then
        echo "DRY RUN + sbatch --parsable --array=$array_spec --time=$NODE_TIME" \
             "--job-name=${JOB_NAME}_packed --output=$PACKED_LOGS/%x_%A_%a.out --export=ALL" \
             "$COMMON_PACKED/packed_node.slurm"
        packed_id=DRYRUN
    else
        packed_id=$(sbatch --parsable \
            --array="$array_spec" \
            --time="$NODE_TIME" \
            --job-name="${JOB_NAME}_packed" \
            --output="$PACKED_LOGS/%x_%A_%a.out" \
            --export=ALL \
            "$COMMON_PACKED/packed_node.slurm")
    fi
    echo "[$chain] packed array: $packed_id"

    # --- merge (afterok: only runs if every packed element succeeded => all files present) ---
    export MERGE_JOB=${JOB_NAME}_merge
    if [ -n "$SKIP_MERGE" ]; then
        echo "[$chain] SKIP_MERGE set -- not submitting the merge. When every stage is done, run:"
        echo "  sbatch --dependency=afterok:<packed_id>[:<packed_id2>...] --time=$MERGE_TIME \\"
        echo "         --job-name=$MERGE_JOB --output=$PACKED_LOGS/%x_%A.out --export=ALL \\"
        echo "         $COMMON_PACKED/merge_node.slurm"
        return 0
    fi
    if [ -n "$DRY_RUN" ]; then
        echo "DRY RUN + sbatch --parsable --dependency=afterok:$packed_id --time=$MERGE_TIME" \
             "--job-name=$MERGE_JOB --output=$PACKED_LOGS/%x_%A.out --export=ALL" \
             "$COMMON_PACKED/merge_node.slurm"
    else
        local merge_id
        merge_id=$(sbatch --parsable \
            --dependency="afterok:$packed_id" \
            --time="$MERGE_TIME" \
            --job-name="$MERGE_JOB" \
            --output="$PACKED_LOGS/%x_%A.out" \
            --export=ALL \
            "$COMMON_PACKED/merge_node.slurm")
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
