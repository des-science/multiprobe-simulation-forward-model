# fiducial cosmology
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/fiducial/cosmo_fiducial \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v14/extended.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v14_fidu" --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# grid cosmology close to Buzzard
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/grid/cosmo_114996 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/obs \
    --with_lensing --with_clustering \
    --tomo_Aia 0.0 0.0 0.0 0.0 \
    --msfm_config=../../configs/v14/extended.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v14_grid" --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# source clustering
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/fiducial/cosmo_fiducial \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/obs \
    --suffix_out="_source_clustering_bgs_low" \
    --with_lensing \
    --tomo_bg_metacal 1.03 0.96 1.01 1.03 \
    --msfm_config=../../configs/v14/extended.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v14_sc_bgs_low" \
    --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/fiducial/cosmo_fiducial \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/obs \
    --suffix_out="_source_clustering_bgs_high" \
    --with_lensing \
    --tomo_bg_metacal 1.22 1.17 1.33 1.47 \
    --msfm_config=../../configs/v14/extended.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v14_sc_bgs_high" \
    --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# noise
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/fiducial/cosmo_fiducial \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/obs \
    --suffix_out="_noise" \
    --with_clustering \
    --noise_only \
    --msfm_config=../../configs/v14/extended.yaml \
    --mode=run --function=main --tasks="0" --n_jobs=1 \
    --job_name="postproc_v14_noise" --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --max_sleep=0 \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# no magnification
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/fiducial/cosmo_fiducial \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/obs \
    --suffix_out="_no_mag" \
    --with_clustering \
    --noiseless \
    --msfm_config=../../configs/v14/extended.yaml \
    --mode=run --function=main --tasks="0" --n_jobs=1 \
    --job_name="postproc_v14_no_mag" --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --max_sleep=0 \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# with magnification
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/fiducial/cosmo_fiducial \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/obs \
    --suffix_out="_mag" \
    --with_clustering \
    --noiseless \
    --tomo_cg 0.42 0.30 1.76 1.94 \
    --msfm_config=../../configs/v14/extended.yaml \
    --mode=run --function=main --tasks="0" --n_jobs=1 \
    --job_name="postproc_v14_mag" --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --max_sleep=0 \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# survey property maps
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/obs \
    --with_clustering \
    --noise_only \
    --suffix_out="_noise" \
    --msfm_config=../../configs/v14/extended.yaml \
    --mode=run --function=main --tasks="0" --n_jobs=1 \
    --job_name="postproc_v14_bench_fidu_noise" --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/obs \
    --with_clustering \
    --noiseless \
    --suffix_out="_survey_ref_noiseless" \
    --msfm_config=../../configs/v14/extended.yaml \
    --mode=run --function=main --tasks="0" --n_jobs=1 \
    --job_name="postproc_v14_bench_fidu" --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/obs \
    --with_clustering \
    --noiseless \
    --contaminate_survey_systematics \
    --suffix_out="_survey_sys_noiseless" \
    --msfm_config=../../configs/v14/extended.yaml \
    --mode=run --function=main --tasks="0" --n_jobs=1 \
    --job_name="postproc_v14_bench_fidu" --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"
