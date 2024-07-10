find . -mindepth 1 -maxdepth 1 -type d -exec mkdir -p /pscratch/sd/a/athomsen/v11desy3/fiducial/{} \;

esub ../msfm/apps/run_fiducial_preprocessing.py \
    --n_files=1000 \
    --config=../configs/v9/linear_bias.yaml \
    --dir_in=/pscratch/sd/a/athomsen/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v9/linear_bias/tfrecords/fiducial \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=main --n_jobs=10 --max_njobs=1000 --tasks="0>10" --keep_submit_files \
    --job_name=tfr_fidu_v9 --log_dir=/pscratch/sd/a/athomsen/run_files/v9/esub_logs \
    --system=slurm --source_file=../pipelines/v9/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"
