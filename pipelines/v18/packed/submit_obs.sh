#!/bin/bash
# Packed regular-QOS replacement for the retired ../../deprecated/v18_esub/obs_commands.sh: runs
# run_single_postprocessing.py over a small, fixed index range (one CosmoGrid permutation
# directory, 20 indices per arm) for every v18 benchmark/mock arm. v18 is packed-only (see
# ../../README.md), so this is the only place these arms are defined -- there is no esub sibling
# to keep in sync with, unlike v17 (see ../../v17/packed/submit_single_mock.sh).
#
# Why not the default esub/shared chain: see ../../common/packed/README.md.
#
# Usage:
#   DRY_RUN=1 ./submit_obs.sh <arm>|all      # print the sbatch calls, touch nothing
#   ./submit_obs.sh <arm>|all                # fresh run: submit every index of the given arm(s)
#   ./submit_obs.sh --rerun <arm>|all        # submit only indices whose obs_maps file is missing
#
# arms: reference dmo box_size particle_count redshift_resolution sc_in_place sc_no_sys sc_gatti
#       eta_shell grid_debug
#
# Reuses ../../common/packed/{run_packed.py,packed_node.slurm,merge_node.slurm} unchanged -- no
# msfm Python was touched. merge_node.slurm's default --tasks=0 is for the grid/fiducial apps'
# glob-based merge(); run_single_postprocessing.merge() needs the FULL index range instead (see
# that file's header), so TASKS is exported explicitly below.

set -euo pipefail
cd "$(dirname "$0")"

REPO=/global/homes/a/athomsen/multiprobe-simulation-forward-model
COMMON_PACKED=$REPO/pipelines/common/packed
DRIVER=$COMMON_PACKED/run_packed.py

APP=$REPO/msfm/apps/run_single_postprocessing.py
DIR_OUT=/pscratch/sd/a/athomsen/dlss/data/v18/baseline/obs
PACKED_LOGS=/pscratch/sd/a/athomsen/run_files/v18/packed_logs

FIDUCIAL_BENCH=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench
DEFAULT_CONFIG=$REPO/configs/v18/default.yaml

# arm_cfg <arm> sets DIR_IN / MSFM_CONFIG / SUFFIX_OUT / EXTRA_ARGS / JOB_NAME, ported 1:1 from the
# retired ../../deprecated/v18_esub/obs_commands.sh (same dir_in/config/suffix/extra-flags per
# arm). N_TOTAL=20 for every arm (matches that file's --tasks="0>20" --n_jobs=20 everywhere).
arm_cfg() {
    EXTRA_ARGS=""
    SUFFIX_OUT=""
    MSFM_CONFIG=$DEFAULT_CONFIG
    DIR_IN=$FIDUCIAL_BENCH
    case "$1" in
        reference)
            JOB_NAME=postproc_v18_bench_fidu
            ;;
        dmo)
            SUFFIX_OUT=_dmo
            MSFM_CONFIG=$REPO/configs/v18/obs/dmo.yaml
            JOB_NAME=postproc_v18_bench_fidu_dmo
            ;;
        box_size)
            DIR_IN=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/box_size
            JOB_NAME=postproc_v18_bench_box
            ;;
        particle_count)
            DIR_IN=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/particle_count
            JOB_NAME=postproc_v18_bench_particle
            ;;
        redshift_resolution)
            DIR_IN=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/redshift_resolution
            JOB_NAME=postproc_v18_bench_redshift
            ;;
        sc_in_place)
            SUFFIX_OUT=_source_clustering_in_place
            MSFM_CONFIG=$REPO/configs/v18/obs/sc_in_place.yaml
            JOB_NAME=postproc_v18_sc_in_place
            ;;
        sc_no_sys)
            SUFFIX_OUT=_source_clustering_no_sys
            MSFM_CONFIG=$REPO/configs/v18/obs/sc_no_sys.yaml
            JOB_NAME=postproc_v18_sc_no_sys
            ;;
        sc_gatti)
            SUFFIX_OUT=_source_clustering_gatti
            MSFM_CONFIG=$REPO/configs/v18/obs/sc_gatti.yaml
            EXTRA_ARGS="--tomo_bg_metacal 1 1 1 1"
            JOB_NAME=postproc_v18_sc_gatti
            ;;
        eta_shell)
            DIR_IN=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/tests/test_eta_ia/CosmoGrid/bary/benchmarks/fiducial_bench
            SUFFIX_OUT="_Aia=0.5,eta=1_shell"
            EXTRA_ARGS="--tomo_Aia 0.5 0.5 0.5 0.5"
            JOB_NAME="postproc_v18_Aia=0.5,eta=1_shell"
            ;;
        grid_debug)
            DIR_IN=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/grid/cosmo_114996
            EXTRA_ARGS="--tomo_Aia 0.0 0.0 0.0 0.0"
            JOB_NAME=postproc_v18_grid
            ;;
        *) echo "unknown arm '$1'" >&2; return 1 ;;
    esac
}

ARMS="reference dmo box_size particle_count redshift_resolution sc_in_place sc_no_sys sc_gatti eta_shell grid_debug"
N_TOTAL=20
# same worst-case-3-per-domain arithmetic as v17/packed/submit_single_mock.sh: ceil(20/8)=3 tasks
# on the busiest domains, so SLOTS=24/OMP=10 (not an assumed even 20/8=2.5) is what's actually
# safe -- see ../../common/packed/README.md.
SLOTS=24
OMP=10
NODE_TIME=01:00:00
MERGE_TIME=00:15:00

DRY_RUN=${DRY_RUN:-}
RERUN=
[ "${1:-}" = --rerun ] && { RERUN=1; shift; }

submit_arm() {
    local arm=$1
    arm_cfg "$arm"
    local index_file=$PACKED_LOGS/${JOB_NAME}_indices.txt
    local cosmo_name; cosmo_name=$(basename "$DIR_IN")

    mkdir -p "$PACKED_LOGS" "$DIR_OUT"

    if [ -n "$RERUN" ]; then
        # missing = all indices whose intermediate obs_maps file is absent. Only valid before a
        # successful merge (which deletes these files) -- a rerun after a completed merge has
        # nothing to detect missing indices from and should be a fresh run instead.
        local present
        present=$(find "$DIR_OUT" -maxdepth 1 -name "${cosmo_name}${SUFFIX_OUT}_obs_maps_*.h5" -size +1k -printf '%f\n' 2>/dev/null \
            | sed -E "s/.*_obs_maps_0*([0-9]+)\.h5\$/\1/" | sort -n -u)
        seq 0 $((N_TOTAL - 1)) | grep -vxF -f <(echo "$present") > "$index_file" || true
    else
        seq 0 $((N_TOTAL - 1)) > "$index_file"
    fi

    local n_idx; n_idx=$(grep -c . "$index_file" || true)
    if [ "$n_idx" -eq 0 ]; then
        echo "[$arm] nothing to do"
        return 0
    fi
    local n_elements=$(( (n_idx + SLOTS - 1) / SLOTS ))
    echo "[$arm] ${RERUN:+RERUN }$n_idx indices, $SLOTS/node, ${OMP} threads/task -> $n_elements node job(s) (${NODE_TIME} each)"

    export SLOTS OMP APP JOB_NAME PACKED_LOGS DRIVER INDEX_FILE=$index_file
    export APP_ARGS="--dir_in=$DIR_IN --dir_out=$DIR_OUT --suffix_out=$SUFFIX_OUT --with_lensing --with_clustering --msfm_config=$MSFM_CONFIG $EXTRA_ARGS"

    local array_spec="0-$((n_elements - 1))"
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
    echo "[$arm] packed array: $packed_id"

    export MERGE_JOB=${JOB_NAME}_merge
    export TASKS
    TASKS=$(seq -s, 0 $((N_TOTAL - 1)))
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
        echo "[$arm] merge job: $merge_id (afterok:$packed_id)"
    fi
}

case "${1:-}" in
    all) for a in $ARMS; do submit_arm "$a"; done ;;
    "") echo "usage: [DRY_RUN=1] $0 [--rerun] <arm>|all" >&2; echo "arms: $ARMS" >&2; exit 1 ;;
    *) submit_arm "$1" ;;
esac
