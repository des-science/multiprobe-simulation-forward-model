#!/bin/bash
#SBATCH --account=m5030
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --time=02:30:00
#SBATCH --licenses=cfs,scratch
#SBATCH --output=/pscratch/sd/a/athomsen/run_files/v17/packed_logs/%x_%A_%a.out

# Packed whole-node replacement for the shared-QOS esub jobarrays of the v17 baseline
# postprocessing (prepared 2026-07-12 while the shared pool was saturated; see README.md).
# Each node job runs 16 postprocessing tasks in parallel (measured ~15 GB and 16 hyperthreads
# per task -> 240 GB / 256 HT of a 512 GB CPU node). Submit via switch_to_regular.sh, which
# passes CHAIN (grid|fiducial) and INDEX_FILE (one missing index per line) through --export.

set -uo pipefail

REPO=/global/homes/a/athomsen/multiprobe-simulation-forward-model
export PATH=/global/common/software/des/athomsen/dlss15/bin:$PATH

SLOTS=16
# per-task threading identical to the shared-QOS runs (SLURM_CPUS_PER_TASK=16 there)
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 \
       VECLIB_MAXIMUM_THREADS=16 NUMEXPR_NUM_THREADS=16
# 16 unpinned processes; kernel balancing beats per-process spread-binding across the whole node
export OMP_PLACES=threads OMP_PROC_BIND=false
# esub --mode=run must not size its threading off the whole-node allocation
unset SLURM_CPUS_PER_TASK

case "${CHAIN:?set CHAIN=grid|fiducial via --export}" in
    grid)
        APP=$REPO/msfm/apps/run_grid_postprocessing.py
        APP_ARGS="--n_files=2500 --dir_out=/pscratch/sd/a/athomsen/v11desy3/v17/baseline/tfrecords/grid"
        JOB_NAME=tfr_grid_v17
        ;;
    fiducial)
        APP=$REPO/msfm/apps/run_fiducial_postprocessing.py
        APP_ARGS="--n_files=1000 --dir_out=/pscratch/sd/a/athomsen/v11desy3/v17/baseline/tfrecords/fiducial"
        JOB_NAME=tfr_fidu_v17
        ;;
    *)
        echo "unknown CHAIN=$CHAIN" && exit 1
        ;;
esac

LOG_DIR=/pscratch/sd/a/athomsen/run_files/v17/esub_logs
PACKED_LOGS=/pscratch/sd/a/athomsen/run_files/v17/packed_logs
# esub appends every index yielded by main() here; rerun accounting relies on it
DONE_FILE=$LOG_DIR/${JOB_NAME}_done.dat

# this element's slice of the missing-index list
start=$((SLURM_ARRAY_TASK_ID * SLOTS + 1))
mapfile -t indices < <(sed -n "${start},$((start + SLOTS - 1))p" "${INDEX_FILE:?set INDEX_FILE via --export}")
if [ ${#indices[@]} -eq 0 ]; then
    echo "no indices for array element $SLURM_ARRAY_TASK_ID"
    exit 0
fi
echo "array element $SLURM_ARRAY_TASK_ID processing indices: ${indices[*]}"

run_one() {
    local idx=$1
    esub "$APP" $APP_ARGS \
        --config=$REPO/configs/v17/baseline.yaml \
        --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
        --cosmogrid_version="1.1" \
        --mode=run --function=main --tasks="$idx" \
        --job_name=$JOB_NAME --log_dir=$LOG_DIR \
        > "$PACKED_LOGS/${JOB_NAME}_index${idx}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        # keep the esub done-index bookkeeping consistent (guarded: esub run mode may
        # already have appended; O_APPEND single-line writes are atomic)
        grep -qxF "$idx" "$DONE_FILE" 2>/dev/null || echo "$idx" >> "$DONE_FILE"
        echo "index $idx done"
    else
        echo "index $idx FAILED with rc=$rc (log: $PACKED_LOGS/${JOB_NAME}_index${idx}.log)"
    fi
    return $rc
}

fail=0
pids=()
for idx in "${indices[@]}"; do
    run_one "$idx" &
    pids+=($!)
done
for p in "${pids[@]}"; do
    wait "$p" || fail=1
done
exit $fail
