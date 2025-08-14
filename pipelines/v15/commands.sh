# debug

esub ../../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../../configs/v15/debug/extended_debug.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/tfrecords/grid \
    --cosmogrid_version="1.1" --max_sleep=0 \
    --mode=run --function=main --n_jobs=1 --tasks="0" \
    --job_name=tfr_grid_v15_debug --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"