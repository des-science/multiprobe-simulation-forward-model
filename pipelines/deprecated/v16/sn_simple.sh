salloc --account=m5030 --constraint=cpu --qos=shared_interactive --time 01:00:00 --ntasks=1 --cpus-per-task=8 --mem-per-cpu=1952
source /global/homes/a/athomsen/multiprobe-simulation-forward-model/pipelines/v16/perlmutter_setup.sh
conda activate dlss15
cd /global/homes/a/athomsen/multiprobe-simulation-forward-model/submissions/v16

esub ../../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../../configs/v16/debug/no_sc.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v16/debug/tfrecords/grid \
    --verbosity=debug \
    --cosmogrid_version="1.1" \
    --max_sleep=0 \
    --mode=run --function=main --n_jobs=1 --max_njobs=1000 --tasks="0" \
    --job_name=tfr_grid_v16 --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=m5030,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../../configs/v16/no_sc.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v16/debug/tfrecords/grid \
    --cosmogrid_version="1.1" \
    --max_sleep=0 \
    --mode=run --function=main --n_jobs=1 --max_njobs=1000 --tasks="0" \
    --job_name=tfr_grid_v16 --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=m5030,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"


esub ../../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../../configs/v16/no_sc.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v16/no_sc/tfrecords/grid \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=all --n_jobs=2500 --max_njobs=1000 --tasks="0>2500" --keep_submit_files \
    --job_name=tfr_grid_v16 --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=m5030,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../../configs/v16/no_sc.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v16/no_sc/tfrecords/grid \
    --cosmogrid_version="1.1" \
    --max_sleep=0 \
    --mode=run --function=merge --n_jobs=1 --max_njobs=1000 --tasks="0" \
    --job_name=tfr_grid_v16 --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=m5030,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# prior
esub ../../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../../configs/v16/debug/prior_sc.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v16/debug/tfrecords/grid \
    --cosmogrid_version="1.1" \
    --max_sleep=0 \
    --mode=run --function=main --n_jobs=1 --max_njobs=1000 --tasks="0" \
    --job_name=tfr_grid_v16 --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=m5030,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../../configs/v16/debug/rot_in_place.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v16/debug/tfrecords/grid \
    --verbosity=debug \
    --cosmogrid_version="1.1" \
    --max_sleep=0 \
    --mode=run --function=main --n_jobs=1 --max_njobs=1000 --tasks="0" \
    --job_name=tfr_grid_v16 --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=m5030,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"


esub ../../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../../configs/v16/debug/clustering.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v16/debug/tfrecords/grid \
    --verbosity=debug \
    --cosmogrid_version="1.1" \
    --max_sleep=0 \
    --mode=run --function=main --n_jobs=1 --max_njobs=1000 --tasks="1251" \
    --job_name=tfr_grid_v16 --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=m5030,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"
