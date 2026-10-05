# debug

esub ../../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../../configs/v16/default.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v16/debug/tfrecords/grid \
    --cosmogrid_version="1.1" \
    --max_sleep=0 \
    --verbosity=debug \
    --mode=run --function=main --n_jobs=1 --max_njobs=1000 --tasks="0" \
    --job_name=tfr_grid_v16 --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=m5030,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"
    
esub ../../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../../configs/v16/default.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v16/extended/tfrecords/grid \
    --cosmogrid_version="1.1" \
    --max_sleep=0 \
    --mode=jobarray --function=main --n_jobs=1 --max_njobs=1000 --tasks="0" \
    --job_name=tfr_grid_v16 --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=m5030,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_power_spectra_noise.py \
    --n_noise=1000000 \
    --n_noise_per_index=1000 \
    --config=../../configs/v16/default.yaml \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v16/extended/cls \
    --max_sleep=0 \
    --mode=run --function=main --system=slurm \
    --job_name=white_noise --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --additional_slurm_args="--account=m5030,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"


esub ../../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../../configs/v16/default.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v16/default/tfrecords/grid \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=all --n_jobs=619 --max_njobs=1000 --tasks="1880>2500" \
    --job_name=tfr_grid_v16 --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=m5030,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# fiducial example
esub ../../msfm/apps/run_fiducial_postprocessing.py \
    --n_files=1000 --no_derivatives \
    --config=../../configs/v16/simple.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v16/default/tfrecords/fiducial \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=all --n_jobs=1 --max_njobs=1000 --tasks="0" --keep_submit_files \
    --job_name=tfr_fidu_v16 --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=m5030,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_fiducial_postprocessing.py \
    --n_files=1000 --no_derivatives \
    --config=../../configs/v16/simple.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v16/default/tfrecords/fiducial \
    --max_sleep=0 \
    --cosmogrid_version="1.1" \
    --mode=run --function=main --n_jobs=1 --max_njobs=1000 --tasks="0" --keep_submit_files \
    --job_name=tfr_fidu_v16 --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=m5030,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"


esub ../../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../../configs/v16/default.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v16/default/tfrecords/grid \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=merge --n_jobs=2500 --max_njobs=1000 --tasks="0>2500" --keep_submit_files \
    --job_name=tfr_grid_v16 --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=m5030,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"