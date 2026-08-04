# DEPRECATED: v18 moved to packed-only submission (regular QOS, no esub) -- every arm below has a
# packed equivalent in ../../v18/packed/submit_obs.sh. Kept here purely as historical record.
#
# this is meant for Perlmutter, not Euler
#
# NOTE: unlike v17, these obs products CANNOT be reused from v16/rot_in_place -- v18 changes the
# shape-noise model itself, so everything that carries an sn map has to be regenerated. Every arm
# uses configs/v18/mocks/<arm>.yaml, each of which is configs/v18/default.yaml with exactly one
# block changed, so a comparison against the reference isolates that one change.

# systematics shift tests #############################################################################################

# reference
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v18/baseline/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v18/default.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v18_bench_fidu" --log_dir=/pscratch/sd/a/athomsen/run_files/v18/esub_logs \
    --system=slurm --source_file=../../pipelines/common/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# dark matter only
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v18/baseline/obs \
    --suffix_out="_dmo" \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v18/mocks/dmo.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v18_bench_fidu_dmo" --log_dir=/pscratch/sd/a/athomsen/run_files/v18/esub_logs \
    --system=slurm --source_file=../../pipelines/common/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# N-body benchmark runs

# box size
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/box_size \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v18/baseline/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v18/default.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v18_bench_box" --log_dir=/pscratch/sd/a/athomsen/run_files/v18/esub_logs \
    --system=slurm --source_file=../../pipelines/common/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# particle count
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/particle_count \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v18/baseline/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v18/default.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v18_bench_particle" --log_dir=/pscratch/sd/a/athomsen/run_files/v18/esub_logs \
    --system=slurm --source_file=../../pipelines/common/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# redshift resolution
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/redshift_resolution \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v18/baseline/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v18/default.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v18_bench_redshift" --log_dir=/pscratch/sd/a/athomsen/run_files/v18/esub_logs \
    --system=slurm --source_file=../../pipelines/common/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# forward model modifications

# source clustering. The v18 reference ALREADY has count based shape noise with the DES imprint, so
# the two arms here are its complements:
#
#   _source_clustering_in_place  in_place, i.e. no density modulation at all
#   _source_clustering_no_sys    count+fixed on a CLEAN source density, with the bias table fit
#                                against that clean model -> the imprint alone
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v18/baseline/obs \
    --suffix_out="_source_clustering_in_place" \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v18/mocks/sc_in_place.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v18_sc_in_place" \
    --log_dir=/pscratch/sd/a/athomsen/run_files/v18/esub_logs \
    --system=slurm --source_file=../../pipelines/common/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v18/baseline/obs \
    --suffix_out="_source_clustering_no_sys" \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v18/mocks/sc_no_sys.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v18_sc_no_sys" \
    --log_dir=/pscratch/sd/a/athomsen/run_files/v18/esub_logs \
    --system=slurm --source_file=../../pipelines/common/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# the Gatti calibrated density modulation, a different shape-noise model entirely. Its bias is
# overridden to 1 because that model calibrates the modulation itself
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v18/baseline/obs \
    --suffix_out="_source_clustering_gatti" \
    --with_lensing --with_clustering \
    --tomo_bg_metacal 1 1 1 1 \
    --msfm_config=../../configs/v18/mocks/sc_gatti.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v18_sc_gatti" \
    --log_dir=/pscratch/sd/a/athomsen/run_files/v18/esub_logs \
    --system=slurm --source_file=../../pipelines/common/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# eta in shells, for comparison with the fiducial
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/tests/test_eta_ia/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v18/baseline/obs \
    --with_lensing --with_clustering \
    --tomo_Aia 0.5 0.5 0.5 0.5 \
    --suffix_out="_Aia=0.5,eta=1_shell" \
    --msfm_config=../../configs/v18/default.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v18_Aia=0.5,eta=1_shell" --log_dir=/pscratch/sd/a/athomsen/run_files/v18/esub_logs \
    --system=slurm --source_file=../../pipelines/common/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# debug ###############################################################################################################

# grid cosmology close to Buzzard
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/grid/cosmo_114996 \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v18/baseline/obs \
    --with_lensing --with_clustering \
    --tomo_Aia 0.0 0.0 0.0 0.0 \
    --msfm_config=../../configs/v18/default.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v18_grid" --log_dir=/pscratch/sd/a/athomsen/run_files/v18/esub_logs \
    --system=slurm --source_file=../../pipelines/common/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"
