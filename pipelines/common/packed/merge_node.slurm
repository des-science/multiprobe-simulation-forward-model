#!/bin/bash
#SBATCH --account=m5030
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --licenses=cfs,scratch
#SBATCH --output=/pscratch/sd/a/athomsen/run_files/packed_logs/%x_%A.out
# Shared across pipeline versions -- every submit.sh below passes --output=$PACKED_LOGS/... on
# the sbatch command line, which overrides the fallback above.
#
# Single whole-node merge, submitted with --time/--job-name/--output and every variable below via
# --export=ALL. Runs the app's merge() in-process.
#
# TASKS defaults to "0", a dummy value: the grid/fiducial apps' merge() globs all .tfrecord files
# in --dir_out and ignores `indices` entirely (it batches per cosmology, bins the Cls and writes
# {grid,fiducial}_cls.h5 from the glob, not from the tasks list). run_single_postprocessing.py's
# merge() is different -- it reads exactly the per-index obs_maps_{index:04}.h5 files named by the
# indices it's given, so callers merging THAT app's output (see submit_single_mock.sh,
# submit_obs.sh) must export TASKS as the full comma-separated index list before calling sbatch.
#
# The afterok on the packed array guarantees the full file set is present before this runs.

set -uo pipefail
export PATH=/global/common/software/des/athomsen/dlss15/bin:$PATH
# DRIVER is exported by submit.sh (an absolute path); it can't be derived from BASH_SOURCE
# because Slurm copies this batch script into /var/spool/slurmd/<job> before running it.
: "${APP:?} ${APP_ARGS:?} ${JOB_NAME:?} ${DRIVER:?}"
TASKS=${TASKS:-0}

# merge is single-process over a tf.data pipeline (or a handful of small h5 files); give it the
# whole node
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32 \
       VECLIB_MAXIMUM_THREADS=32 NUMEXPR_NUM_THREADS=32
export OMP_PLACES=threads OMP_PROC_BIND=false
unset SLURM_CPUS_PER_TASK

echo "merging $JOB_NAME (tasks: $TASKS)"
python "$DRIVER" --app "$APP" --function=merge --tasks="$TASKS" $APP_ARGS
