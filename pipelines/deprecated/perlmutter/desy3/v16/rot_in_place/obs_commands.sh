# systematics shift tests #############################################################################################

# reference
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v16/rot_in_place/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v16/rot_in_place.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v16_bench_fidu" --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# dark matter only
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v16/rot_in_place/obs \
    --suffix_out="_dmo" \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v16/rot_in_place_dmo.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v16_bench_fidu_dmo" --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# N-body benchmark runs

# box size
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/box_size \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v16/rot_in_place/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v16/rot_in_place.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v16_bench_box" --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# particle count
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/particle_count \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v16/rot_in_place/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v16/rot_in_place.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v16_bench_particle" --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# redshift resolution
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/redshift_resolution \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v16/rot_in_place/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v16/rot_in_place.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v16_bench_redshift" --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# forward model modifications

# source clustering
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v16/rot_in_place/obs \
    --suffix_out="_source_clustering_fixed" \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v16/sc_fixed.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v16_sc_fixed" \
    --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v16/rot_in_place/obs \
    --suffix_out="_source_clustering_gatti" \
    --with_lensing --with_clustering \
    --tomo_bg_metacal 1 1 1 1 \
    --msfm_config=../../configs/v16/sc_gatti.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v16_sc_gatti" \
    --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"


# eta in shells, for comparison with the fiducial
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/tests/test_eta_ia/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v16/rot_in_place/obs \
    --with_lensing --with_clustering \
    --tomo_Aia 0.5 0.5 0.5 0.5 \
    --suffix_out="_nla_per_shell" \
    --msfm_config=../../configs/v16/rot_in_place.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v16__nla_per_shell" --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# debug ###############################################################################################################

# grid cosmology close to Buzzard
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/grid/cosmo_114996 \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v16/rot_in_place/obs \
    --with_lensing --with_clustering \
    --tomo_Aia 0.0 0.0 0.0 0.0 \
    --msfm_config=../../configs/v16/rot_in_place.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v16_grid" --log_dir=/pscratch/sd/a/athomsen/run_files/v16/esub_logs \
    --system=slurm --source_file=../../pipelines/v16/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"
