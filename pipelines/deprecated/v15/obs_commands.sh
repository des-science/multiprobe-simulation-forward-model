# systematics shift tests #############################################################################################

# reference
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v15/extended.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_bench_fidu" --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# box size
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/box_size \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v15/extended.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_bench_box" --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# particle count
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/particle_count \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v15/extended.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_bench_particle" --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# redshift resolution
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/redshift_resolution \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v15/extended.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_bench_redshift" --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

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

# eta in shells, for comparison with the fiducial
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/tests/test_eta_ia/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --with_lensing --with_clustering \
    --tomo_Aia 0.5 0.5 0.5 0.5 \
    --suffix_out="_Aia=0.5,eta=1_shell" \
    --msfm_config=../../configs/v15/extended.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_Aia=0.5,eta=1_shell" --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"


# miscellaneous #################################################################################################

# Niall test
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/grid/cosmo_114996 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --with_clustering \
    --tomo_bg 1.38 1.54 1.68 1.78 \
    --suffix_out="_MAP_buzzard_mean" \
    --msfm_config=../../configs/v15/extended.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_niall_buzzard_mean" --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# fiducial cosmology
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/fiducial/cosmo_fiducial \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v15/extended.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_fidu" --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# grid cosmology close to Buzzard
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/grid/cosmo_114996 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --with_lensing --with_clustering \
    --tomo_Aia 0.0 0.0 0.0 0.0 \
    --msfm_config=../../configs/v15/extended.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_grid" --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# intrinsic alignments
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/grid/cosmo_114996 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --msfm_config=../../configs/v15/extended.yaml \
    --with_lensing --with_clustering \
    --tomo_Aia 0.0 0.0 0.0 0.0 \
    --suffix_out="_Aia=0" \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_Aia=0" --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/grid/cosmo_114996 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --msfm_config=../../configs/v15/extended.yaml \
    --with_lensing --with_clustering \
    --tomo_Aia 1.0 1.0 1.0 1.0 \
    --suffix_out="_Aia=1" \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_Aia=1" --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/grid/cosmo_114996 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --msfm_config=../../configs/v15/extended.yaml \
    --with_lensing --with_clustering \
    --tomo_Aia -1.0 -1.0 -1.0 -1.0 \
    --suffix_out="_Aia=-1" \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_Aia=-1" --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# eta in bins
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --with_lensing --with_clustering \
    --tomo_Aia 0.8014308 0.9186018 1.0741925 1.1903182 \
    --suffix_out="_eta_ia=1_bin" \
    --msfm_config=../../configs/v15/extended.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_eta_ia=1_bin" --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"


# eta in shells
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/tests/test_eta_ia/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --with_lensing --with_clustering \
    --tomo_Aia 1.0 1.0 1.0 1.0 \
    --suffix_out="_eta_ia=1_shell" \
    --msfm_config=../../configs/v15/extended.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_eta_ia=1_shell" --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# change noise seed
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v15/extended/obs \
    --with_lensing --with_clustering \
    --np_seed=1234 --suffix_out="_seed=1234" \
    --msfm_config=../../configs/v15/extended.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v15_bench_fidu_seed=1234" --log_dir=/pscratch/sd/a/athomsen/run_files/v15/esub_logs \
    --system=slurm --source_file=../../pipelines/v15/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"
