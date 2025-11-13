
# multiprobe-simulation-forward-model
[![arXiv](https://img.shields.io/badge/arXiv-2511.04681-b31b1b.svg)](https://arxiv.org/abs/2511.04681)

This repository provides a pipeline to forward model Dark Energy Survey Year 3 (DES Y3)-like weak lensing and galaxy clustering mocks from cosmological simulations:
- **Input:** Full-sky probe maps (weak lensing signal, intrinsic alignments, and dark matter density) from the [CosmoGridV1](http://www.cosmogrid.ai/) simulation suite [[Kacprzak & Fluri et al. 2022](https://arxiv.org/abs/2209.04662)] projected using [`cosmogridv11`](https://github.com/CosmoGridCollab/cosmogridv11)
- **Output:** Self-consistent DES Y3-like weak lensing (convergence) and galaxy clustering (linear bias) maps with realistic survey properties [[Thomsen et al. 2025](https://arxiv.org/abs/2511.04681)]
- **Deep Learning Integration**: The data storage and loading are designed to work in conjunction with the training pipeline for mutual information maximizing graph convolutional neural networks in [`y3-deep-lss`](https://github.com/des-science/y3-deep-lss)

![](data/figures/combined_moll+gnom.png)

## Installation

### Requirements
- Python >= 3.8
- Pre-existing installation of [`TensorFlow`](https://www.tensorflow.org/install) and [`TensorFlow-Probability`](https://www.tensorflow.org/probability) in the python environment (to ensure proper GPU support on HPC clusters)
- Python environment with pre-existing installations of [`TensorFlow`](https://www.tensorflow.org/install), and [`TensorFlow-Probability`](https://www.tensorflow.org/probability) (to ensure proper GPU support on HPC clusters)
- All additional dependencies are automatically installed from `pyproject.toml`

### Installation Steps

1. **Install this package in editable mode:**
```bash
pip install -e .
```

## Repository Structure

### `msfm`
- `msfm/apps` production scripts meant to be submitted using [`esub-epipe`](https://cosmo-gitlab.phys.ethz.ch/cosmo_public/esub-epipe). These are parallelized over the fiducial (plus finite-difference derivates) or grid cosmologies of the CosmoGridV1 to generate `.tfrecord` files of mock maps
- `msfm/utils` various helper functions 
- `msfm/fiducial_pipeline.py` and `msfm/grid_pipeline.py` contain the generators to efficiently read the `.tfrecord` files for neural network training

### `configs`
Configuration files specifying the constants used throughout the pipeline. These include information on the cosmological parameters like the priors, the definition of the redshift bins, relative paths, and analysis specific choices like the healpix resolution or linear bias parametrization.

### `data`
Data like CosmoGridV1 settings, the survey masks, and catalog ellipticities. These files are a prerequisite to running the pipeline. Note that `DESY3_noise_512.h5` is not stored on GitHub due to its filesize and has to be generated from the source galaxy catalog by running `notebooks/noise_file.ipynb`.

### `notebooks`
These notebooks are used to generate the contents of the `data` directory. 

### `pipelines`
Submission commands for [`esub-epipe`](https://cosmo-gitlab.phys.ethz.ch/cosmo_public/esub-epipe) to execute the scripts in `msfm/apps` in a distributed fashion on HPC CPU clusters.

## Companion Repositories
- Informative map-level neural summary statistics: [`y3-deep-lss`](https://github.com/des-science/y3-deep-lss)
- Simulation-based inference: [`multiprobe-simulation-inference`](https://github.com/des-science/multiprobe-simulation-inference)