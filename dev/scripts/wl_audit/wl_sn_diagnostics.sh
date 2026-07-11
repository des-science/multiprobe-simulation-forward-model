#!/bin/bash
#SBATCH --account=a0158
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --job-name=wl_audit
#SBATCH --output=/iopsstor/scratch/cscs/athomsen/deep_lss/runs/wl_audit/slurm-%j.out

export OMP_NUM_THREADS=32

REPOS="/users/athomsen/dlss/repos"
MYSCRATCH="/iopsstor/scratch/cscs/athomsen"

OUT_DIR="$MYSCRATCH/deep_lss/runs/wl_audit"
mkdir -p "$OUT_DIR"

srun -N1 --ntasks-per-node=1 --environment=tensorflow \
    bash -c "source ~/dlss/tf_env/bin/activate && python $REPOS/multiprobe-simulation-forward-model/dev/scripts/wl_audit/wl_sn_diagnostics.py \
        --config=$REPOS/multiprobe-simulation-forward-model/configs/v16/rot_in_place.yaml \
        --fiducial_perm_dir=$MYSCRATCH/deep_lss/data/projected/fiducial/cosmo_fiducial/perm_0000 \
        --grid_cls_file=$MYSCRATCH/deep_lss/data/v16/rot_in_place/cls/grid_cls.h5 \
        --out_dir=$OUT_DIR"
