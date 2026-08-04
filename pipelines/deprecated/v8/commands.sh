esub msfm/apps/run_datavectors.py --simset=grid --config=configs/v7/linear_bias.yaml --dir_in=/home/ipa/refreg/experiments/tomaszk/projects/220627_cosmogrid_desy3/001_perms/cosmogrid_desy3/CosmoGrid/DESY3 --dir_out=/cluster/work/refregier/athomsen/CosmoGrid/DESY3/v8/debug --mode=jobarray --function=all --n_jobs=10 --tasks="0>10" --job_name=dv_grid --system=slurm --from_san

esub msfm/apps/run_datavectors.py --simset=grid --config=configs/v7/linear_bias.yaml --dir_in=/home/ipa/refreg/experiments/tomaszk/projects/220627_cosmogrid_desy3/001_perms/cosmogrid_desy3/CosmoGrid/DESY3 --dir_out=/cluster/work/refregier/athomsen/CosmoGrid/DESY3/v8/debug --mode=run --function=main --n_jobs=1 --tasks="0" --job_name=dv_grid --system=slurm --from_san --debug

esub msfm/apps/run_datavectors.py --simset=grid --config=configs/v7/linear_bias.yaml --dir_in=/home/ipa/refreg/data/data_products/CosmoGrid/processed/CosmoGrid/v11desy3 --dir_out=/cluster/work/refregier/athomsen/CosmoGrid/DESY3/v8/debug --mode=run --function=main --n_jobs=1 --tasks="0" --job_name=dv_grid --system=slurm --from_san --debug

esub msfm/apps/run_datavectors.py --simset=fiducial --config=configs/v7/linear_bias.yaml --dir_in=/home/ipa/refreg/experiments/tomaszk/projects/220627_cosmogrid_desy3/001_perms/cosmogrid_desy3/CosmoGrid/DESY3 --dir_out=/cluster/work/refregier/athomsen/CosmoGrid/DESY3/v8/debug --mode=run --function=main --n_jobs=1 --tasks="1" --job_name=dv_pert --system=slurm --from_san --debug






esub msfm/apps/run_datavectors.py --simset=fiducial --config=configs/v8/linear_bias.yaml --dir_in=/home/ipa/refreg/data/data_products/CosmoGrid/processed/CosmoGrid/v11desy3 --dir_out=/cluster/work/refregier/athomsen/CosmoGrid/DESY3/v8/debug --mode=run --function=main --n_jobs=1 --tasks="0" --job_name=dv_fidu --system=slurm --from_san --debug
esub msfm/apps/run_datavectors.py --simset=fiducial --config=configs/v8/linear_bias.yaml --dir_in=/home/ipa/refreg/data/data_products/CosmoGrid/processed/CosmoGrid/v11desy3 --dir_out=/cluster/work/refregier/athomsen/CosmoGrid/DESY3/v8/debug --mode=run --function=main --n_jobs=1 --tasks="1" --job_name=dv_pert --system=slurm --from_san --debug --with_bary
esub msfm/apps/run_datavectors.py --simset=grid --config=configs/v8/linear_bias.yaml --dir_in=/home/ipa/refreg/data/data_products/CosmoGrid/processed/CosmoGrid/v11desy3 --dir_out=/cluster/work/refregier/athomsen/CosmoGrid/DESY3/v8/debug --mode=run --function=main --n_jobs=1 --tasks="0" --job_name=dv_grid --system=slurm --from_san --debug --with_bary --cosmogrid_version="1.1"


esub msfm/apps/run_grid_preprocessing.py --n_files=2500 --config=configs/v8/linear_bias.yaml --dir_in=/home/ipa/refreg/data/data_products/CosmoGrid/processed/CosmoGrid/v11desy3 --dir_out=/cluster/work/refregier/athomsen/CosmoGrid/DESY3/v8/debug --mode=run --function=main --n_jobs=1 --tasks="0" --job_name=grid_new --system=slurm --from_san --debug --with_bary
esub msfm/apps/run_grid_preprocessing.py --n_files=2500 --config=configs/v8/linear_bias.yaml --dir_in=/home/ipa/refreg/data/data_products/CosmoGrid/processed/CosmoGrid/v11desy3 --dir_out=/cluster/work/refregier/athomsen/CosmoGrid/DESY3/v8/debug --mode=run --function=main --n_jobs=1 --tasks="0" --job_name=grid_new --system=slurm --from_san --with_bary
esub msfm/apps/run_grid_preprocessing.py --n_files=2500 --config=configs/v7/linear_bias.yaml --dir_in=/home/ipa/refreg/experiments/tomaszk/projects/220627_cosmogrid_desy3/001_perms/cosmogrid_desy3/CosmoGrid/DESY3 --dir_out=/cluster/work/refregier/athomsen/CosmoGrid/DESY3/v8/debug --mode=run --function=main --n_jobs=1 --tasks="0" --job_name=grid_new --system=slurm --from_san --debug --with_bary --cosmogrid_version="1"

# fiducial
esub msfm/apps/run_fiducial_preprocessing.py --n_files=200 --config=configs/v8/linear_bias.yaml --dir_in=/home/ipa/refreg/data/data_products/CosmoGrid/processed/CosmoGrid/v11desy3 --dir_out=/cluster/work/refregier/athomsen/CosmoGrid/DESY3/v8/debug --mode=run --function=main --n_jobs=1 --tasks="0" --job_name=fidu_new --system=slurm --from_san --debug --with_bary
esub msfm/apps/run_fiducial_preprocessing.py --n_files=1000 --config=configs/v8/linear_bias.yaml --dir_in=/home/ipa/refreg/data/data_products/CosmoGrid/processed/CosmoGrid/v11desy3 --dir_out=/cluster/work/refregier/athomsen/CosmoGrid/DESY3/v8/debug/full --mode=jobarray --function=all --n_jobs=1 --tasks="0" --job_name=fidu_new --system=slurm --from_san --debug --with_bary


# grid
esub msfm/apps/run_grid_preprocessing.py --n_files=2500 --config=configs/v8/linear_bias.yaml --dir_in=/home/ipa/refreg/data/data_products/CosmoGrid/processed/CosmoGrid/v11desy3 --dir_out=/cluster/work/refregier/athomsen/CosmoGrid/DESY3/v8/debug --mode=run --function=main --n_jobs=1 --tasks="0" --job_name=grid_new --system=slurm --from_san --with_bary
esub msfm/apps/run_grid_preprocessing.py --n_files=2500 --config=configs/v8/linear_bias.yaml --dir_in=/home/ipa/refreg/data/data_products/CosmoGrid/processed/CosmoGrid/v11desy3 --dir_out=/cluster/work/refregier/athomsen/CosmoGrid/DESY3/v8/debug/full --mode=jobarray --function=main --n_jobs=1 --tasks="0" --job_name=grid_new --system=slurm --from_san --with_bary
