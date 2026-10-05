#!/bin/bash
# Container-side fan-out for packed_node.slurm. Launched via `srun --environment=tensorflow`,
# so this runs INSIDE the tensorflow container. Starts one `esub --mode=run` process per
# cosmology index in $INDICES (set by packed_node.slurm), OMP threads each, and waits for all.
#
# esub --mode=run executes main() locally (no SLURM submission, resources() irrelevant); each
# index writes its own .tfrecord, so the concurrent processes never clash. Finished indices are
# appended to the esub <job>_done.dat so the merge / rerun bookkeeping stays consistent.
set -uo pipefail

source ~/dlss/tf_env/bin/activate

APP=$REPO/msfm/apps/run_grid_postprocessing.py
CONFIG=$REPO/configs/v18/default.yaml
JOB_NAME=tfr_grid_v18
DONE_FILE=$LOG_DIR/${JOB_NAME}_done.dat

# per-cosmology threading (mirrors the shared-QOS SLURM_CPUS_PER_TASK). The whole-node
# allocation must NOT size esub's threading off itself, so unset SLURM_CPUS_PER_TASK.
export OMP_NUM_THREADS=${OMP:-8} OPENBLAS_NUM_THREADS=${OMP:-8} MKL_NUM_THREADS=${OMP:-8} \
       VECLIB_MAXIMUM_THREADS=${OMP:-8} NUMEXPR_NUM_THREADS=${OMP:-8}
export OMP_PLACES=threads OMP_PROC_BIND=false
unset SLURM_CPUS_PER_TASK

run_one() {
    local idx=$1
    esub "$APP" \
        --n_files=2500 \
        --config="$CONFIG" \
        --dir_in="$DIR_IN" \
        --dir_out="$DIR_OUT" \
        --cosmogrid_version="1.1" \
        --cluster=clariden \
        --mode=run --function=main --tasks="$idx" \
        --job_name=$JOB_NAME --log_dir="$LOG_DIR" \
        > "$PACKED_LOGS/${JOB_NAME}_index${idx}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        # O_APPEND single-line writes are atomic; guard against a double append
        grep -qxF "$idx" "$DONE_FILE" 2>/dev/null || echo "$idx" >> "$DONE_FILE"
        echo "index $idx done"
    else
        echo "index $idx FAILED rc=$rc (see $PACKED_LOGS/${JOB_NAME}_index${idx}.log)"
    fi
    return $rc
}

fail=0
pids=()
for idx in $INDICES; do
    run_one "$idx" &
    pids+=($!)
done
for p in "${pids[@]}"; do
    wait "$p" || fail=1
done
exit $fail
