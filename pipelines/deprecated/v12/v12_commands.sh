# debug
esub ../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../configs/v12/nonlinear.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3/CosmoGrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v12/debug \
    --cosmogrid_version="1.1" --max_sleep=0 \
    --mode=jobarray --function=main --n_jobs=1 --max_njobs=1000 --tasks="0" --keep_submit_files \
    --job_name=tfr_grid_v12_debug --log_dir=/pscratch/sd/a/athomsen/run_files/v12/esub_logs \
    --system=slurm --source_file=../pipelines/v12/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch" \
    --main_time=0.1

esub ../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../configs/v12/nonlinear.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3/CosmoGrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v12/debug \
    --cosmogrid_version="1.1" --max_sleep=0 \
    --mode=jobarray --function=main --n_jobs=1 --max_njobs=1000 --tasks="0" --keep_submit_files \
    --job_name=tfr_grid_v12_debug_16 --log_dir=/pscratch/sd/a/athomsen/run_files/v12/esub_logs/debug \
    --system=slurm --source_file=../pipelines/v12/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch" \
    --main_time=0.1 --main_n_cores=16

esub check_cores.py \
    --mode=jobarray --function=main --n_jobs=1 --tasks="0" --keep_submit_files \
    --job_name=check_cores_np_tf_hp --log_dir=/pscratch/sd/a/athomsen/run_files/v12/esub_logs/debug \
    --system=slurm --source_file=../pipelines/v12/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch" \
    --main_time=0.05 --main_n_cores=8

esub ../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../configs/v12/debug/nonlinear.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3/CosmoGrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v12/debug \
    --cosmogrid_version="1.1" \
    --mode=run --function=main --n_jobs=1 --max_njobs=1000 --tasks="0" \
    --job_name=tfr_grid_v12 --log_dir=/pscratch/sd/a/athomsen/run_files/v12/esub_logs/debug \
    --system=slurm --source_file=../pipelines/v12/perlmutter_setup.sh \
    --max_sleep=0 \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# production
esub ../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../configs/v12/nonlinear.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3/CosmoGrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v12/nonlinear/tfrecords/grid \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=all --n_jobs=2500 --max_njobs=1000 --tasks="0>2500" --keep_submit_files \
    --job_name=tfr_grid_v12 --log_dir=/pscratch/sd/a/athomsen/run_files/v12/esub_logs \
    --system=slurm --source_file=../pipelines/v12/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../configs/v12/nonlinear.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/v11desy3/CosmoGrid/v11desy3 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v12/nonlinear/tfrecords/grid \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=rerun_missing --n_jobs=2500 --max_njobs=1000 --tasks="0>2500" --keep_submit_files \
    --job_name=tfr_grid_v12 --log_dir=/pscratch/sd/a/athomsen/run_files/v12/esub_logs \
    --system=slurm --source_file=../pipelines/v12/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

rsync -ahv --prune-empty-dirs \
    --include={"*/","*.yaml","*.h5","*.npy","*.pt"} \
    --exclude={"*","debug","wandb/","wandb/**"} \
    /pscratch/sd/a/athomsen/run_files \
    /global/cfs/cdirs/des/athomsen/deep_lss
