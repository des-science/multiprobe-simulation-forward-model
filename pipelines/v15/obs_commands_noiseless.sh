# systematics shift tests #############################################################################################

# reference
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --msfm_config=../../configs/v15/extended.yaml \
    --with_lensing --with_clustering --noiseless \
    --suffix_out="_noiseless" \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_bench_fidu" --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --msfm_config=../../configs/v15/extended.yaml \
    --with_lensing --with_clustering --noise_only \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --suffix_out="_noise" \
    --job_name="postproc_v15_bench_fidu_noise" --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# box size
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/box_size \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --msfm_config=../../configs/v15/extended.yaml \
    --with_lensing --with_clustering --noiseless \
    --suffix_out="_noiseless" \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_bench_box" --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# particle count
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/particle_count \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --msfm_config=../../configs/v15/extended.yaml \
    --with_lensing --with_clustering --noiseless \
    --suffix_out="_noiseless" \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_bench_particle" --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# redshift resolution
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/redshift_resolution \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --msfm_config=../../configs/v15/extended.yaml \
    --with_lensing --with_clustering --noiseless \
    --suffix_out="_noiseless" \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_bench_redshift" --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# eta in shells
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/tests/test_eta_ia/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --with_lensing --with_clustering --noiseless \
    --tomo_Aia 0.5 0.5 0.5 0.5 \
    --suffix_out="shell_Aia=0.5,eta=1_noiseless" \
    --msfm_config=../../configs/v15/extended.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_Aia=0.5,eta=1_shell" --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# dark matter only
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --suffix_out="_dmo" \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v15/extended_dmo.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_bench_fidu_dmo" --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"


# TODO 

# source clustering
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --suffix_out="_source_clustering_bgs_low" \
    --with_lensing --with_clustering \
    --tomo_bg_metacal 1.03 0.96 1.01 1.03 \
    --msfm_config=../../configs/v15/extended.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_sc_bgs_low" \
    --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --suffix_out="_source_clustering_bgs_high" \
    --with_lensing --with_clustering \
    --tomo_bg_metacal 1.22 1.17 1.33 1.47 \
    --msfm_config=../../configs/v15/extended.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_sc_bgs_high" \
    --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"



