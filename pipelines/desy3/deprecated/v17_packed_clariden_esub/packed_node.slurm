#!/bin/bash
# Packed whole-node postprocessing for Clariden (CSCS) Grace-Hopper nodes.
#
# Clariden has NO shared QOS -- all partitions are OverSubscribe=EXCLUSIVE, so every allocation
# is a whole 288-core / ~870 GB node and esub's jobarray mode (one task per node) would waste
# ~36x. This is the Clariden analogue of pipelines/v17/packed/packed_node.slurm (Perlmutter
# regular-QOS): each array element owns one exclusive node and runs SLOTS cosmologies in parallel
# via `esub --mode=run` (which does NOT submit and does NOT touch resources()), OMP threads each.
#
# Node budget (measured v17, cosmo_000001, 8 threads: ~0.59 h and ~15 GB peak per cosmology):
#   SLOTS * OMP <= 288 cores   and   SLOTS * ~15 GB <= ~870 GB
#   default SLOTS=36, OMP=8  -> 288 threads, ~540 GB (leaves headroom); ~36 cosmos / ~0.6 h / node.
#
# Submit (indices 0..N_TOTAL-1, ceil(N_TOTAL/SLOTS) elements; 2500/36 -> 70 elements = 0..69):
#   sbatch --array=0-69 packed_node.slurm
#   # retune, e.g. more concurrency / fewer threads (memory permitting), fewer array elements:
#   SLOTS=48 OMP=6 sbatch --array=0-52 --export=ALL,SLOTS=48,OMP=6 packed_node.slurm

#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --job-name=tfr_grid_v17_packed
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/run_files/v17/packed_logs/%x_%A_%a.out

set -uo pipefail

# ---- tunables (override via --export=ALL,SLOTS=..,OMP=..) ----
SLOTS=${SLOTS:-36}          # cosmologies run concurrently per node
export OMP=${OMP:-8}        # threads per cosmology; keep SLOTS*OMP <= 288
N_TOTAL=${N_TOTAL:-2500}    # total grid cosmologies -> indices 0..N_TOTAL-1

REPO=/users/athomsen/dlss/repos/multiprobe-simulation-forward-model
export REPO
export DIR_IN=/iopsstor/scratch/cscs/athomsen/v11desy3
export DIR_OUT=/iopsstor/scratch/cscs/athomsen/v11desy3/v17/baseline/tfrecords/grid
export LOG_DIR=/iopsstor/scratch/cscs/athomsen/run_files/v17/esub_logs
export PACKED_LOGS=/iopsstor/scratch/cscs/athomsen/run_files/v17/packed_logs
mkdir -p "$LOG_DIR" "$PACKED_LOGS" "$DIR_OUT"

# this array element's contiguous slice of indices [start, start+SLOTS), clamped to N_TOTAL
start=$((SLURM_ARRAY_TASK_ID * SLOTS))
indices=()
for ((i = start; i < start + SLOTS && i < N_TOTAL; i++)); do indices+=("$i"); done
if [ ${#indices[@]} -eq 0 ]; then
    echo "no indices for array element $SLURM_ARRAY_TASK_ID (start=$start >= N_TOTAL=$N_TOTAL)"
    exit 0
fi
echo "array element $SLURM_ARRAY_TASK_ID (SLOTS=$SLOTS OMP=$OMP) -> indices ${indices[*]}"
export INDICES="${indices[*]}"

# the batch script runs on the host; launch the container for the fan-out worker
srun -N1 --ntasks=1 --environment=tensorflow \
    bash "$REPO/pipelines/v17/packed_clariden/packed_worker.sh"
