#! /bin/bash 
#
#SBATCH --output=/pscratch/sd/a/athomsen/run_files/v9/esub_logs/tfr_fidu_v9_rerun_missing.o 
#SBATCH --error=/pscratch/sd/a/athomsen/run_files/v9/esub_logs/tfr_fidu_v9_rerun_missing.e 
#SBATCH --job-name=tfr_fidu_v9_rerun_missing 
#SBATCH --account=des 
#SBATCH --constraint=cpu 
#SBATCH --qos=shared 
#SBATCH --licenses=cfs 
#SBATCH --licenses=scratch 
#SBATCH --array=1-1 
#SBATCH --time=4:00:00 
#SBATCH --mem-per-cpu=1952 
#SBATCH --cpus-per-task=8 

srun --ntasks=1 bash; source /global/u2/a/athomsen/multiprobe-simulation-forward-model/pipelines/v9/perlmutter_setup.sh;  python -m esub.submit --job_name=tfr_fidu_v9 --tasks='0>1000' --source_file=../pipelines/v9/perlmutter_setup.sh --main_memory=1952 --main_time=4 --main_scratch=2000 --function=rerun_missing --executable=/global/u2/a/athomsen/multiprobe-simulation-forward-model/submissions/../msfm/apps/run_fiducial_preprocessing.py --n_jobs=1 --log_dir=/pscratch/sd/a/athomsen/run_files/v9/esub_logs --system=slurm --main_name=main --batchsize=100000 --max_njobs=1000 --main_n_cores=8 --main_gpu=0 --main_gpu_memory=1000 --esub_verbosity=3 --main_mode=jobarray --mode=jobarray --n_files=1000 --config=../configs/v9/linear_bias.yaml --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3 --dir_out=/pscratch/sd/a/athomsen/v11desy3/v9/linear_bias/tfrecords/fiducial --cosmogrid_version=1.1 