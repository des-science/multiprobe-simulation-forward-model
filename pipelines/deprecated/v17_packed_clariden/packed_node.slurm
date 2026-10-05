#!/bin/bash
# Packed whole-node v17 grid postprocessing for Clariden (CSCS) -- esub-free.
#
# Clariden has NO shared QOS (all partitions are OverSubscribe=EXCLUSIVE), so every allocation is
# a whole 288-core / ~870 GB Grace-Hopper node, billed per node. We therefore pack ~36 cosmologies
# onto each node ourselves. Unlike the esub variant (../packed_clariden_esub/), here SLURM does the
# fan-out and CPU pinning directly: `srun --ntasks=SLOTS --cpus-per-task=OMP` starts SLOTS copies of
# run_slot.sh, one per SLURM_PROCID, each running one cosmology via run_cosmo.py (a direct call into
# the app's main(); no esub, no jobarray/resources(), no _done.dat).
#
# Index mapping is fully deterministic:  idx = SLURM_ARRAY_TASK_ID * SLOTS + SLURM_PROCID.
# So array element k owns the contiguous block [k*SLOTS, (k+1)*SLOTS).
#
# Node budget (measured v17, cosmo_000001, 8 threads: ~0.59 h and ~15 GB peak per cosmology):
#   SLOTS * OMP <= 288 cores   and   SLOTS * ~15 GB <= ~870 GB
#   default SLOTS=36, OMP=8  -> 288 threads, ~540 GB; ~36 cosmos / ~0.6 h / node.
#
# Submit (indices 0..N_TOTAL-1, ceil(N_TOTAL/SLOTS) elements; 2500/36 -> 70 elements = 0..69):
#   sbatch --array=0-69 packed_node.slurm
#   # retune concurrency/threads (fewer array elements): ceil(2500/48)=53
#   sbatch --array=0-52 --export=ALL,SLOTS=48,OMP=6 packed_node.slurm
#   # rerun one failed node (its block is known from the mapping): e.g. element 12
#   sbatch --array=12 packed_node.slurm

#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --job-name=tfr_grid_v17
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/run_files/v17/packed_logs/%x_%A_%a.out

set -uo pipefail

# ---- tunables (override via --export=ALL,SLOTS=..,OMP=..) ----
SLOTS=${SLOTS:-36}          # cosmologies per node = srun tasks; keep SLOTS*OMP <= 288
OMP=${OMP:-8}               # cores per cosmology
N_TOTAL=${N_TOTAL:-2500}    # total grid cosmologies -> indices 0..N_TOTAL-1

REPO=/users/athomsen/dlss/repos/multiprobe-simulation-forward-model
DIR_IN=/iopsstor/scratch/cscs/athomsen/v11desy3
DIR_OUT=/iopsstor/scratch/cscs/athomsen/v11desy3/v17/baseline/tfrecords/grid
LOG_DIR=/iopsstor/scratch/cscs/athomsen/run_files/v17/packed_logs
mkdir -p "$LOG_DIR" "$DIR_OUT"

# run_slot.sh reads these; --cpus-per-task pins OMP cores per task and sets SLURM_PROCID 0..SLOTS-1
export SLOTS OMP N_TOTAL REPO DIR_IN DIR_OUT LOG_DIR

echo "array element $SLURM_ARRAY_TASK_ID: SLOTS=$SLOTS OMP=$OMP -> indices" \
     "$((SLURM_ARRAY_TASK_ID * SLOTS))..$((SLURM_ARRAY_TASK_ID * SLOTS + SLOTS - 1)) (clamped to $N_TOTAL)"

srun -N1 --ntasks="$SLOTS" --cpus-per-task="$OMP" --environment=tensorflow \
    bash "$REPO/pipelines/v17/packed_clariden/run_slot.sh"
