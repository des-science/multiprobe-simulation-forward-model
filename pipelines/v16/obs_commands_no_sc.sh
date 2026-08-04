conda activate dlss15
cd /global/homes/a/athomsen/multiprobe-simulation-forward-model/submissions/v16

# reference
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v16/no_sc/obs \
    --suffix_out="_bsc=fit" \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v16/default.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v16_bench_bsc=fit" --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"


# rotate in place
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v16/no_sc/obs \
    --suffix_out="_rot" \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v16/no_sc.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v16_bench_rot" --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"


# source clustering bias
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v16/no_sc/obs \
    --suffix_out="_bsc=1" \
    --with_lensing --with_clustering \
    --tomo_bg_metacal 1 1 1 1 \
    --msfm_config=../../configs/v16/no_sc.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v16_bsc=1" \
    --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v16/no_sc/obs \
    --suffix_out="_bsc=0" \
    --with_lensing --with_clustering \
    --tomo_bg_metacal 0 0 0 0 \
    --msfm_config=../../configs/v16/no_sc.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v16_bsc=0" \
    --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"


# debug
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v16/no_sc/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v16/default.yaml \
    --mode=run --function=main --tasks="0" --n_jobs=1 \
    --job_name="postproc_v16_bench_debug" --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --max_sleep=0 \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"
