#!/bin/bash
#SBATCH --account=m5030
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --licenses=cfs,scratch
#SBATCH --output=/pscratch/sd/a/athomsen/run_files/v17/packed_logs/%x_%A_%a.out
# Generic packed executor. --time, --array and --job-name are supplied by submit.sh,
# along with every variable read below via --export=ALL. Each array element takes its own
# SLOTS-sized slice of $INDEX_FILE and runs $SLOTS run_packed.py tasks in parallel, one per
# index. No esub: run_packed.py imports the app and calls its main() directly.
#
# CRITICAL -- each task is pinned to one of the node's 8 NUMA domains with numactl. This does
# two things (both measured to matter, see README "Why pinning is mandatory"):
#   1. Bounds each task's cpuset to 32 CPUs, so libraries that auto-size their thread pools to
#      the visible CPU count -- TensorFlow above all (TFRecordWriter + the verify-parse) -- make
#      32-thread pools instead of 256-thread ones. Unpinned, 24-32 tasks each spawned ~536
#      threads (load avg ~750 on 256 HW threads) and the run went ~4x slower.
#   2. Keeps each task's memory local to its domain (the SHTs are memory-bandwidth heavy), so
#      the 8 memory controllers are used evenly and accesses don't cross the die.
# SLOTS is a multiple of 8, so `position % 8` puts SLOTS/8 tasks on each domain evenly.

set -uo pipefail
export PATH=/global/common/software/des/athomsen/dlss15/bin:$PATH
# DRIVER is exported by submit.sh (an absolute path). We can't derive it from BASH_SOURCE here:
# Slurm copies this batch script into /var/spool/slurmd/<job> before running it.
: "${SLOTS:?} ${OMP:?} ${APP:?} ${APP_ARGS:?} ${JOB_NAME:?} ${PACKED_LOGS:?} ${INDEX_FILE:?} ${DRIVER:?}"

# per-task threading. numactl caps the cpuset to a 32-thread domain; these caps then split that
# domain's threads across the SLOTS/8 tasks sharing it. TF_* are essential: OMP_NUM_THREADS does
# NOT bound TensorFlow's intra/inter-op pools, which are the ones that exploded when unpinned.
export OMP_NUM_THREADS=$OMP OPENBLAS_NUM_THREADS=$OMP MKL_NUM_THREADS=$OMP \
       VECLIB_MAXIMUM_THREADS=$OMP NUMEXPR_NUM_THREADS=$OMP \
       TF_NUM_INTRAOP_THREADS=$OMP TF_NUM_INTEROP_THREADS=1
export OMP_PLACES=threads OMP_PROC_BIND=false
# the app must not size its threading off the whole-node allocation
unset SLURM_CPUS_PER_TASK

# this element's slice of the index list
start=$((SLURM_ARRAY_TASK_ID * SLOTS + 1))
mapfile -t indices < <(sed -n "${start},$((start + SLOTS - 1))p" "$INDEX_FILE")
if [ ${#indices[@]} -eq 0 ]; then
    echo "no indices for array element $SLURM_ARRAY_TASK_ID"
    exit 0
fi
echo "array element $SLURM_ARRAY_TASK_ID ($JOB_NAME) processing indices: ${indices[*]}"

run_one() {
    local idx=$1 dom=$2
    # --cpunodebind confines CPUs to the domain; --preferred keeps memory local but allows
    # spillover so a momentarily tight domain can't OOM-kill the task
    numactl --cpunodebind="$dom" --preferred="$dom" \
        python "$DRIVER" --app "$APP" --function=main --tasks="$idx" $APP_ARGS \
        > "$PACKED_LOGS/${JOB_NAME}_index${idx}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "index $idx (numa $dom) done"
    else
        echo "index $idx (numa $dom) FAILED with rc=$rc (log: $PACKED_LOGS/${JOB_NAME}_index${idx}.log)"
    fi
    return $rc
}

fail=0
pids=()
pos=0
for idx in "${indices[@]}"; do
    run_one "$idx" $((pos % 8)) &
    pids+=($!)
    pos=$((pos + 1))
done
for p in "${pids[@]}"; do
    wait "$p" || fail=1
done
# non-zero if any task failed -> the array element fails -> the afterok merge stays blocked. A
# failed task leaves no .tfrecord or a 0-byte stub; `submit.sh --rerun` treats both as missing.
exit $fail
