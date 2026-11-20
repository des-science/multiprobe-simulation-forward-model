#!/bin/bash
# One task of packed_node.slurm, run by srun inside the tensorflow container (SLURM_PROCID = 0 .. SLOTS-1).
#
# main: processes the index on line SLURM_ARRAY_TASK_ID * SLOTS + SLURM_PROCID + 1 of $INDEX_FILE, pinned with taskset
#       to cores [PROCID * OMP, (PROCID + 1) * OMP). The range must lie inside one NUMA node, so the task's memory stays
#       on its own Grace by first touch. Unpinned, TensorFlow sizes its thread pools to all 288 CPUs (measured ~4x slower
#       on Perlmutter). Logs to $PACKED_LOGS/${JOB_NAME}_index<idx>.log.
# merge: one process on the whole node, logging to the job's output file.

set -uo pipefail
: "${SLOTS:?} ${OMP:?} ${APP:?} ${APP_ARGS:?} ${JOB_NAME:?} ${PACKED_LOGS:?} ${DRIVER:?}"
FUNCTION=${FUNCTION:-main}
rank=${SLURM_PROCID:?}

# OMP_NUM_THREADS does not bound TensorFlow's pools, hence the TF_* variables. The GPUs stay untouched
export OMP_NUM_THREADS=$OMP OPENBLAS_NUM_THREADS=$OMP MKL_NUM_THREADS=$OMP NUMEXPR_NUM_THREADS=$OMP \
       VECLIB_MAXIMUM_THREADS=$OMP TF_NUM_INTRAOP_THREADS=$OMP TF_NUM_INTEROP_THREADS=1
export OMP_PLACES=threads OMP_PROC_BIND=false
export CUDA_VISIBLE_DEVICES=
unset SLURM_CPUS_PER_TASK

if [ "$FUNCTION" = merge ]; then
    exec python "$DRIVER" --app "$APP" --function=merge --tasks="${TASKS:-0}" $APP_ARGS
fi

line=$(( ${SLURM_ARRAY_TASK_ID:-0} * SLOTS + rank + 1 ))
idx=$(sed -n "${line}p" "${INDEX_FILE:?}")
if [ -z "$idx" ]; then
    echo "rank $rank: no index on line $line of $INDEX_FILE, nothing to do"
    exit 0
fi

first=$(( rank * OMP ))
last=$(( first + OMP - 1 ))
numa=
for dir in /sys/devices/system/node/node[0-9]*; do
    # CPU-less nodes (the GPUs' HBM) have an empty cpulist
    cpulist=$(cat "$dir/cpulist")
    for range in ${cpulist//,/ }; do
        lo=${range%-*}
        hi=${range#*-}
        if (( first >= lo && last <= hi )); then
            numa=${dir##*node}
        fi
    done
done
if [ -z "$numa" ]; then
    echo "rank $rank: cores $first-$last of index $idx do not lie inside one NUMA node" >&2
    exit 1
fi

log=$PACKED_LOGS/${JOB_NAME}_index${idx}.log
echo "rank $rank: index $idx on cores $first-$last (NUMA node $numa), log $log"
taskset -c "$first-$last" python "$DRIVER" --app "$APP" --function=main --tasks="$idx" $APP_ARGS > "$log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
    echo "rank $rank: index $idx done"
else
    echo "rank $rank: index $idx FAILED with rc=$rc, see $log"
fi
exit $rc
