#! /bin/bash 
#
#SBATCH --output=/global/u2/a/athomsen/multiprobe-simulation-forward-model/esub_logs/ind_debug%a.o 
#SBATCH --error=/global/u2/a/athomsen/multiprobe-simulation-forward-model/esub_logs/ind_debug%a.e 
#SBATCH --job-name=ind_debug 
#SBATCH --account=des 
#SBATCH --constraint=cpu 
#SBATCH --qos=debug 
#SBATCH --array=1-2 
#SBATCH --nodes=1 
#SBATCH --time=0:01:00 

echo $SLURM_ARRAY_TASK_ID