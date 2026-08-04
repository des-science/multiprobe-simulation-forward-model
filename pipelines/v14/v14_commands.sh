# debug

esub ../../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../../configs/v14/extended.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/debug/extended/tfrecords/grid \
    --cosmogrid_version="1.1" --max_sleep=0 \
    --mode=run --function=main --n_jobs=1 --max_njobs=1000 --tasks="0" --keep_submit_files \
    --job_name=tfr_grid_v14_debug --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../../configs/v14/extended.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/tfrecords/grid \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=all --n_jobs=1 --max_njobs=1000 --tasks="0" --keep_submit_files \
    --job_name=tfr_grid_v14 --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../../configs/v14/extended.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/tfrecords/grid \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=merge --n_jobs=2500 --max_njobs=1000 --tasks="0>2500" --keep_submit_files \
    --job_name=tfr_grid_v14 --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_fiducial_only_postprocessing.py \
    --n_files=1000 \
    --config=../../configs/v14/simple.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/tfrecords/fiducial \
    --cosmogrid_version="1.1" --max_sleep=0 \
    --mode=jobarray --function=all --n_jobs=1000 --tasks="0>1000" \
    --job_name=tfr_fidu_v14 --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_fiducial_postprocessing.py \
    --n_files=1000 \
    --config=../../configs/v14/simple_debug.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/tfrecords/fiducial \
    --cosmogrid_version="1.1" --max_sleep=0 --no_derivatives \
    --mode=run --function=main --n_jobs=1 --max_njobs=1000 --tasks="0" \
    --job_name=tfr_fidu_v14 --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/grid/cosmo_008963 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/obs \
    --suffix_out="_test" \
    --msfm_config=../../configs/v14/extended.yaml \
    --mode=run --function=merge --tasks="0>5" --n_jobs=10 \
    --jobname=postproc_v14 --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/grid/cosmo_008963 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/obs \
    --suffix_out="_test" \
    --msfm_config=../../configs/v14/extended.yaml \
    --mode=jobarray --function=all --tasks="0>10" --n_jobs=10 \
    --jobname=postproc_v14 --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/fiducial/cosmo_fiducial \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/obs \
    --suffix_out="_source_clustering_bgs=0.5" \
    --with_lensing \
    --tomo_bg_metacal=[0.5,0.5,0.5,0.5] \
    --msfm_config=../../configs/v14/extended.yaml \
    --mode=jobarray --function=all --tasks="0>4" --n_jobs=4 \
    --jobname=postproc_v14_sc_bgs=0.5 --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/fiducial/cosmo_fiducial \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/obs \
    --suffix_out="_source_clustering_bgs=1.5" \
    --with_lensing \
    --tomo_bg_metacal 1.5 1.5 1.5 1.5 \
    --msfm_config=../../configs/v14/extended.yaml \
    --mode=jobarray --function=all --tasks="0>4" --n_jobs=4 \
    --jobname="postproc_v14_sc_bgs=1.5" --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_grid_postprocessing.py \
    --n_files=2500 \
    --config=../../configs/v14/extended.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/tfrecords/grid \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=merge --n_jobs=2500 --max_njobs=1000 --tasks="0>2500" --keep_submit_files \
    --job_name=tfr_grid_v14 --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_fiducial_postprocessing.py \
    --n_files=1000 --no_derivatives \
    --config=../../configs/v14/simple.yaml \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/tfrecords/fiducial \
    --cosmogrid_version="1.1" \
    --mode=jobarray --function=merge --n_jobs=1000 --max_njobs=1000 --tasks="0>1000" --keep_submit_files \
    --job_name=tfr_fidu_v14 --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_power_spectra_noise.py \
    --n_noise=100000 \
    --n_noise_per_index=1000 \
    --config=../../configs/v14/extended.yaml \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/cls \
    --mode=jobarray --function=all --system=slurm \
    --job_name=white_noise --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/grid/cosmo_000001 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/obs \
    --with_lensing --with_clustering --suffix_out="_debug" \
    --msfm_config=../../configs/v14/extended.yaml \
    --mode=run --function=main --tasks="0" --n_jobs=1 \
    --jobname="postproc_v14_debug" --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch" \
    --max_sleep=0

esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/grid/cosmo_114996 \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/obs \
    --with_lensing --with_clustering \
    --tomo_Aia 0.0 0.0 0.0 0.0 \
    --msfm_config=../../configs/v14/extended.yaml \
    --mode=jobarray --function=all --tasks="0>20" --n_jobs=20 \
    --jobname="postproc_v14" --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/fiducial/cosmo_fiducial \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/obs \
    \
    --suffix_out="_source_clustering_bgs_high" \
    --with_lensing \
    --tomo_bg_metacal 1.03 0.96 1.01 1.03 \
    --tomo_bg_metacal 1.22 1.17 1.33 1.47 \
    --msfm_config=../../configs/v14/extended.yaml \
    --mode=jobarray --function=all --tasks="0>4" --n_jobs=4 \
    --jobname="postproc_v14_sc_bgs_low" \
    \
    --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch" # --suffix_out="_source_clustering_bgs_low" \
# --jobname="postproc_v14_sc_bgs_high" \

esub run_mcmc_for_coverage_tests.py \
    --preds_file=/pscratch/sd/a/athomsen/run_files/v14/extended/combined/mutual_info/2025-04-30_02-27-42_deepsphere_default/preds_400000.h5 \
    --flow_dir=/pscratch/sd/a/athomsen/run_files/v14/extended/combined/mutual_info/2025-04-30_02-27-42_deepsphere_default/400000_steps_likelihood_sigmoid_7/likelihood_flow \
    --tasks="[2,3,6,9,95,98,108,109,137,142,145,173,176,183,194,218,265,312,414,417,476,491,493,524,525,566,568,569,582,604,708,727,732,883,886,887,897,898,899,913,925,937]" \
    --mode=jobarray --function=main --n_jobs=42 \
    --jobname=mcmc --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub run_mcmc_for_coverage_tests.py \
    --preds_file=/pscratch/sd/a/athomsen/run_files/v14/extended/combined/mutual_info/2025-04-30_02-27-42_deepsphere_default/preds_400000.h5 \
    --flow_dir=/pscratch/sd/a/athomsen/run_files/v14/extended/combined/mutual_info/2025-04-30_02-27-42_deepsphere_default/400000_steps_likelihood_sigmoid_7/likelihood_flow \
    --n_sims=1000 --tasks="0>1000" --n_jobs=1000 \
    --mode=jobarray --function=main --keep_submit_files --test \
    --jobname=mcmc --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub run_mcmc_for_coverage_tests.py \
    --preds_file=/pscratch/sd/a/athomsen/run_files/v14/extended/combined/mutual_info/2025-04-30_02-27-42_deepsphere_default/preds_400000.h5 \
    --flow_dir=/pscratch/sd/a/athomsen/run_files/v14/extended/combined/mutual_info/2025-04-30_02-27-42_deepsphere_default/400000_steps_likelihood_sigmoid_7/likelihood_flow \
    --n_sims=1000 \
    --mode=jobarray --function=all --n_jobs=1000 --test \
    --jobname=mcmc --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

esub run_mcmc_for_coverage_tests.py \
    --preds_file=/pscratch/sd/a/athomsen/run_files/v14/extended/combined/mutual_info/2025-04-30_02-27-42_deepsphere_default/preds_400000.h5 \
    --flow_dir=/pscratch/sd/a/athomsen/run_files/v14/extended/combined/mutual_info/2025-04-30_02-27-42_deepsphere_default/400000_steps_likelihood_sigmoid_7/likelihood_flow \
    --n_sims=1000 \
    --mode=run --function=merge --n_jobs=1000 \
    --jobname=mcmc --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# particle count
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/particle_count \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v14/extended.yaml \
    --mode=run --function=merge --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v14_bench_particle" --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"

# redshift resolution
esub ../../msfm/apps/run_single_postprocessing.py \
    --dir_in=/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/benchmarks/redshift_resolution \
    --dir_out=/pscratch/sd/a/athomsen/v11desy3/v14/extended/obs \
    --with_lensing --with_clustering \
    --msfm_config=../../configs/v14/extended.yaml \
    --mode=run --function=merge --tasks="0>20" --n_jobs=20 \
    --job_name="postproc_v14_bench_redshift" --log_dir=/pscratch/sd/a/athomsen/run_files/v14/esub_logs \
    --system=slurm --source_file=../../pipelines/v14/perlmutter_setup.sh \
    --additional_slurm_args="--account=des,--constraint=cpu,--qos=shared,--licenses=cfs,--licenses=scratch"
