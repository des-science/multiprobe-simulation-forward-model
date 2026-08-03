# this is meant for Perlmutter, not Euler
#
# NOTE: unlike v17, these obs products CANNOT be reused from v16/rot_in_place -- v18 changes the
# shape-noise model itself, so everything that carries an sn map has to be regenerated. The
# variant-forward-model benchmarks below keep their v16 configs on purpose: they are comparison arms
# for a DIFFERENT shape-noise model (rotate in place, gatti) and are not meant to track v18.

# systematics shift tests #############################################################################################

# reference
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v18/baseline/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v18/baseline.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v18_bench_fidu" --log_dir=/pscratch/sd/a/athomsen/run_files/v18/esub_logs \
    --system=slurm --source_file=../../pipelines/v18/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# dark matter only (v16 variant config, see NOTE above)
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v18/baseline/obs \
    --suffix_out="_dmo" \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v16/rot_in_place_dmo.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v18_bench_fidu_dmo" --log_dir=/pscratch/sd/a/athomsen/run_files/v18/esub_logs \
    --system=slurm --source_file=../../pipelines/v18/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# N-body benchmark runs

# box size
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/box_size \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v18/baseline/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v18/baseline.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v18_bench_box" --log_dir=/pscratch/sd/a/athomsen/run_files/v18/esub_logs \
    --system=slurm --source_file=../../pipelines/v18/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# particle count
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/particle_count \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v18/baseline/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v18/baseline.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v18_bench_particle" --log_dir=/pscratch/sd/a/athomsen/run_files/v18/esub_logs \
    --system=slurm --source_file=../../pipelines/v18/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# redshift resolution
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/redshift_resolution \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v18/baseline/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v18/baseline.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v18_bench_redshift" --log_dir=/pscratch/sd/a/athomsen/run_files/v18/esub_logs \
    --system=slurm --source_file=../../pipelines/v18/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# forward model modifications

# source clustering (v16 variant configs, see NOTE above)
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v18/baseline/obs \
    --suffix_out="_source_clustering_fixed" \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v16/sc_fixed.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v18_sc_fixed" \
    --log_dir=/pscratch/sd/a/athomsen/run_files/v18/esub_logs \
    --system=slurm --source_file=../../pipelines/v18/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v18/baseline/obs \
    --suffix_out="_source_clustering_gatti" \
    --with_lensing --with_clustering \
    --tomo_bg_metacal 1 1 1 1 \
    --msfm_config=../../configs/v16/sc_gatti.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v18_sc_gatti" \
    --log_dir=/pscratch/sd/a/athomsen/run_files/v18/esub_logs \
    --system=slurm --source_file=../../pipelines/v18/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# eta in shells, for comparison with the fiducial
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/tests/test_eta_ia/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v18/baseline/obs \
    --with_lensing --with_clustering \
    --tomo_Aia 0.5 0.5 0.5 0.5 \
    --suffix_out="_Aia=0.5,eta=1_shell" \
    --msfm_config=../../configs/v18/baseline.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v18_Aia=0.5,eta=1_shell" --log_dir=/pscratch/sd/a/athomsen/run_files/v18/esub_logs \
    --system=slurm --source_file=../../pipelines/v18/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# debug ###############################################################################################################

# grid cosmology close to Buzzard
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/grid/cosmo_114996 \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v18/baseline/obs \
    --with_lensing --with_clustering \
    --tomo_Aia 0.0 0.0 0.0 0.0 \
    --msfm_config=../../configs/v18/baseline.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v18_grid" --log_dir=/pscratch/sd/a/athomsen/run_files/v18/esub_logs \
    --system=slurm --source_file=../../pipelines/v18/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"
