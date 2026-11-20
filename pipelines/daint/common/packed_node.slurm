#!/bin/bash
#SBATCH --account=lp184
#SBATCH --nodes=1
# Generic packed executor for daint: one exclusive GH200 node runs $SLOTS msfm tasks in parallel on its CPUs.
# --partition, --time, --array, --job-name and --output come from the caller (e.g. ../desy6/v1/submit.sh), and every
# variable below arrives via --export=ALL.
#
# One srun step starts $SLOTS copies of packed_task.sh inside the tensorflow container. Each copy takes line
# SLURM_ARRAY_TASK_ID * SLOTS + SLURM_PROCID of $INDEX_FILE and pins itself to cores [PROCID * OMP, (PROCID + 1) * OMP),
# so with OMP dividing 72 every task sits inside one Grace NUMA domain. --cpu-bind=none because SLURM's own binding
# of sub-node steps fails on Alps (the step is CANCELLED and only slurm-<jid>.out says why); the task pins itself.
#
# FUNCTION=merge runs the app's merge() once, on the whole node (SLOTS=1, no pinning).

set -uo pipefail
: "${SLOTS:?} ${OMP:?} ${APP:?} ${APP_ARGS:?} ${JOB_NAME:?} ${PACKED_LOGS:?} ${DRIVER:?} ${TASK_SCRIPT:?}"
FUNCTION=${FUNCTION:-main}
[ "$FUNCTION" = main ] && : "${INDEX_FILE:?}"

if (( SLOTS * OMP > SLURM_CPUS_ON_NODE )); then
    echo "SLOTS * OMP = $((SLOTS * OMP)) exceeds the $SLURM_CPUS_ON_NODE CPUs of this node" >&2
    exit 1
fi
echo "$JOB_NAME $FUNCTION on $(hostname): array element ${SLURM_ARRAY_TASK_ID:-none}, $SLOTS tasks x $OMP cores"

# --kill-on-bad-exit=0 and --wait=0: one failed index must not take down the others on the node
srun --ntasks="$SLOTS" --cpus-per-task="$OMP" --cpu-bind=none --kill-on-bad-exit=0 --wait=0 \
    --environment=tensorflow bash "$TASK_SCRIPT"
rc=$?
# non-zero if any task failed, so the afterok merge stays blocked and submit.sh --rerun picks the index up
echo "$JOB_NAME $FUNCTION finished with rc=$rc"
exit $rc
