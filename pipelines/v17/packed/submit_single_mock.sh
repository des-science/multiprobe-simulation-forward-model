#!/bin/bash
# Packed regular-QOS run of run_single_postprocessing.py over a small, fixed index range -- e.g.
# an exploratory mock (one CosmoGrid permutation directory, tens of indices), not a full
# grid/fiducial production chain. Sibling of submit.sh in this directory, simplified to a single
# "chain" and sized so everything fits on ONE node in ONE array element.
#
# Why not the default esub/shared chain: see ../../common/packed/README.md -- shared QOS is a
# saturated 70-node pool (ETA can be weeks), regular QOS has idle capacity but is node-exclusive,
# so we hand-pack many tasks onto one node instead of burning a whole node per ~2 GB task.
#
# This duplicates exactly one arm also defined as an esub command in ../esub/obs_commands.sh
# (job_name="postproc_v17_sc_fixed_sys", the "source clustering with the DES Y3 imaging
# systematics" block) -- v17 keeps both forms deliberately (see ../packed/README.md), so if you
# change dir_in/dir_out/suffix_out/msfm_config here, check that block too, and vice versa.
#
# Usage:
#   DRY_RUN=1 ./submit_single_mock.sh        # print the sbatch calls, touch nothing
#   ./submit_single_mock.sh                  # fresh run: submit every index
#   ./submit_single_mock.sh --rerun          # submit only indices whose obs_maps file is missing
#
# Reuses ../../common/packed/{run_packed.py,packed_node.slurm,merge_node.slurm} unchanged -- no
# msfm Python was touched. merge_node.slurm's merge() needs the FULL index range (not the dummy
# --tasks=0 the grid/fiducial chains use), since run_single_postprocessing.merge() reads exactly
# the {cosmo_name}{suffix_out}_obs_maps_{index:04}.h5 files named by the indices it is given (see
# ../../common/packed/merge_node.slurm's header) -- so TASKS is exported explicitly below.

set -euo pipefail
cd "$(dirname "$0")"

REPO=/global/homes/a/athomsen/multiprobe-simulation-forward-model
COMMON_PACKED=$REPO/pipelines/common/packed
DRIVER=$COMMON_PACKED/run_packed.py

# --- this run's configuration (mirrors ../esub/obs_commands.sh's sc_fixed_sys block, see above) -
APP=$REPO/msfm/apps/run_single_postprocessing.py
JOB_NAME=postproc_v17_sc_fixed_sys
DIR_IN=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench
DIR_OUT=/pscratch/sd/a/athomsen/dlss/data/v17/baseline/obs
SUFFIX_OUT=_source_clustering_fixed_sys
MSFM_CONFIG=$REPO/configs/v17/mocks/sc_fixed_sys.yaml
PACKED_LOGS=/pscratch/sd/a/athomsen/run_files/v17/packed_logs

# N_TOTAL matches the esub --tasks="0>20" range this replaces. SLOTS >= N_TOTAL so everything
# fits in a single node/array element; OMP=10 is the same bandwidth-knee value ../packed/submit.sh
# uses for the grid/fiducial chains (3 tasks/domain -> 32/3 threads each), since
# postprocess_shape_noise is the same memory-bandwidth-heavy SHT workload.
N_TOTAL=20
SLOTS=24
OMP=10
# matches run_single_postprocessing.resources() main_time=1 (hour) for a single index
NODE_TIME=01:00:00
MERGE_TIME=00:15:00

DRY_RUN=${DRY_RUN:-}
RERUN=${1:-}
if [ -n "$RERUN" ] && [ "$RERUN" != --rerun ]; then
    echo "usage: [DRY_RUN=1] $0 [--rerun]" >&2
    exit 1
fi

mkdir -p "$PACKED_LOGS" "$DIR_OUT"

cosmo_name=$(basename "$DIR_IN")
index_file=$PACKED_LOGS/${JOB_NAME}_indices.txt

if [ "$RERUN" = --rerun ]; then
    # missing = all indices whose intermediate obs_maps file is absent. Only valid before a
    # successful merge (which deletes these files) -- a rerun after a completed merge has nothing
    # to detect missing indices from and should be a fresh run instead.
    present=$(find "$DIR_OUT" -maxdepth 1 -name "${cosmo_name}${SUFFIX_OUT}_obs_maps_*.h5" -size +1k -printf '%f\n' 2>/dev/null \
        | sed -E "s/.*_obs_maps_0*([0-9]+)\.h5\$/\1/" | sort -n -u)
    seq 0 $((N_TOTAL - 1)) | grep -vxF -f <(echo "$present") > "$index_file" || true
else
    seq 0 $((N_TOTAL - 1)) > "$index_file"
fi

n_idx=$(grep -c . "$index_file" || true)
if [ "$n_idx" -eq 0 ]; then
    echo "nothing to do"
    exit 0
fi
n_elements=$(( (n_idx + SLOTS - 1) / SLOTS ))
echo "${RERUN:+RERUN }$n_idx indices, $SLOTS/node, ${OMP} threads/task -> $n_elements node job(s) (${NODE_TIME} each)"

export SLOTS OMP APP JOB_NAME PACKED_LOGS DRIVER INDEX_FILE=$index_file
export APP_ARGS="--dir_in=$DIR_IN --dir_out=$DIR_OUT --suffix_out=$SUFFIX_OUT --with_lensing --with_clustering --msfm_config=$MSFM_CONFIG"

array_spec="0-$((n_elements - 1))"
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
echo "packed array: $packed_id"

# the merge always needs the FULL 0..N_TOTAL-1 range: a fresh run produces all of them, and a
# --rerun's previously-successful indices are still sitting untouched in DIR_OUT
export MERGE_JOB=${JOB_NAME}_merge
export TASKS
TASKS=$(seq -s, 0 $((N_TOTAL - 1)))
if [ -n "$DRY_RUN" ]; then
    echo "DRY RUN + sbatch --parsable --dependency=afterok:$packed_id --time=$MERGE_TIME" \
         "--job-name=$MERGE_JOB --output=$PACKED_LOGS/%x_%A.out --export=ALL" \
         "$COMMON_PACKED/merge_node.slurm"
else
    merge_id=$(sbatch --parsable \
        --dependency="afterok:$packed_id" \
        --time="$MERGE_TIME" \
        --job-name="$MERGE_JOB" \
        --output="$PACKED_LOGS/%x_%A.out" \
        --export=ALL \
        "$COMMON_PACKED/merge_node.slurm")
    echo "merge job: $merge_id (afterok:$packed_id)"
fi
