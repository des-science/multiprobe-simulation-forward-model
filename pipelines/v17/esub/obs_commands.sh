# this is meant for Perlmutter, not Euler
#
# NOTE: at the fiducial (bta = 0) the v17 forward model is bit-identical to v16/rot_in_place, so all
# of these obs products can be reused/symlinked from
# /pscratch/sd/a/athomsen/dlss/data/v16/rot_in_place/obs (Clariden: data/v16/rot_in_place/obs)
# instead of being regenerated. The commands below produce a self-contained v17 dataset; the
# variant-forward-model benchmarks (dmo, source clustering) keep their v16 configs, since no v17
# variants of those configs exist -- their outputs are bta-independent as well. The one exception is
# the _source_clustering_fixed_sys arm below, which is new and has nothing to reuse.

# systematics shift tests #############################################################################################

# reference
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v17/baseline/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v17/baseline.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v17_bench_fidu" --log_dir=/pscratch/sd/a/athomsen/run_files/v17/esub_logs \
    --system=slurm --source_file=../../pipelines/common/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# dark matter only (v16 variant config, see NOTE above)
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v17/baseline/obs \
    --suffix_out="_dmo" \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v16/rot_in_place_dmo.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v17_bench_fidu_dmo" --log_dir=/pscratch/sd/a/athomsen/run_files/v17/esub_logs \
    --system=slurm --source_file=../../pipelines/common/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# N-body benchmark runs

# box size
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/box_size \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v17/baseline/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v17/baseline.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v17_bench_box" --log_dir=/pscratch/sd/a/athomsen/run_files/v17/esub_logs \
    --system=slurm --source_file=../../pipelines/common/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# particle count
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/particle_count \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v17/baseline/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v17/baseline.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v17_bench_particle" --log_dir=/pscratch/sd/a/athomsen/run_files/v17/esub_logs \
    --system=slurm --source_file=../../pipelines/common/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# redshift resolution
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/redshift_resolution \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v17/baseline/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v17/baseline.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v17_bench_redshift" --log_dir=/pscratch/sd/a/athomsen/run_files/v17/esub_logs \
    --system=slurm --source_file=../../pipelines/common/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# forward model modifications

# source clustering (v16 variant configs, see NOTE above)
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v17/baseline/obs \
    --suffix_out="_source_clustering_fixed" \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v16/sc_fixed.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v17_sc_fixed" \
    --log_dir=/pscratch/sd/a/athomsen/run_files/v17/esub_logs \
    --system=slurm --source_file=../../pipelines/common/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# source clustering with the DES Y3 imaging systematics imprinted on the source density, i.e. the v18
# shape-noise model on the v17 lineage. configs/v17/mocks/sc_fixed_sys.yaml differs from
# configs/v17/baseline.yaml in nothing but the shape noise (same maps, same channels, same
# normalization), so this is a clean systematics shift test against the v17 training set. The bias
# table is the one fit against this same contaminated model (metacal_biases_desy3_v2_sys.h5,
# enforced by configuration.py), so b is not absorbing the systematics: bin 4 is 1.98 here against
# 3.06 in the clean fit.
# NOTE: configs/v18/default.yaml can NOT be used here -- it is delta-NLA and carries a ds channel.
# Also defined (deliberately duplicated, not derived) as a packed regular-QOS submission in
# ../packed/submit_single_mock.sh -- if you change dir_in/dir_out/suffix_out/msfm_config here,
# check that script too, and vice versa.
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v17/baseline/obs \
    --suffix_out="_source_clustering_fixed_sys" \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v17/mocks/sc_fixed_sys.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v17_sc_fixed_sys" \
    --log_dir=/pscratch/sd/a/athomsen/run_files/v17/esub_logs \
    --system=slurm --source_file=../../pipelines/common/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v17/baseline/obs \
    --suffix_out="_source_clustering_gatti" \
    --with_lensing --with_clustering \
    --tomo_bg_metacal 1 1 1 1 \
    --msfm_config=../../configs/v16/sc_gatti.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v17_sc_gatti" \
    --log_dir=/pscratch/sd/a/athomsen/run_files/v17/esub_logs \
    --system=slurm --source_file=../../pipelines/common/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# eta in shells, for comparison with the fiducial
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/tests/test_eta_ia/CosmoGrid/bary/benchmarks/fiducial_bench \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v17/baseline/obs \
    --with_lensing --with_clustering \
    --tomo_Aia 0.5 0.5 0.5 0.5 \
    --suffix_out="_Aia=0.5,eta=1_shell" \
    --msfm_config=../../configs/v17/baseline.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v17_Aia=0.5,eta=1_shell" --log_dir=/pscratch/sd/a/athomsen/run_files/v17/esub_logs \
    --system=slurm --source_file=../../pipelines/common/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# debug ###############################################################################################################

# grid cosmology close to Buzzard
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/grid/cosmo_114996 \
    --dir_out=/pscratch/sd/a/athomsen/dlss/data/v17/baseline/obs \
    --with_lensing --with_clustering \
    --tomo_Aia 0.0 0.0 0.0 0.0 \
    --msfm_config=../../configs/v17/baseline.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v17_grid" --log_dir=/pscratch/sd/a/athomsen/run_files/v17/esub_logs \
    --system=slurm --source_file=../../pipelines/common/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"
