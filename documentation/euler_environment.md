The python version is set via, which is usually included in the `.bash_profile`
```
module load gcc/8.2.0 python/3.10.4 git/2.31.1 hdf5/1.10.1
```
It's important that the virtual environment contains all of Euler's preinstalled packages like achieved with
```
python -m venv --system-site-packages dlss
```
The following packages should be included to a `requirements.txt` file
```
pip install esub-epipe
pip install icecream
```
The most important packages to be installed from source are
```
git clone git@github.com:des-science/multiprobe-simulation-forward-model.git
git clone git@github.com:des-science/y3-deep-lss.git
git clone git@github.com:des-science/multiprobe-simulation-inference.git
git clone https://cosmo-gitlab.phys.ethz.ch/cosmo_public/estats.git
```
where `DeepSphere` is missing on purpose, since everything after the `msfm` is better run on Perlmutter anyways. Note that `estats` has a `tensorflow` dependency that still kicks in when `tensorflow-gpu` is installed, as is usually the case on Euler. Therefore, the `tensorflow` dependency has to be commented out from within the cloned repo.
All of these packages are simply installed with
```
pip install -e .
```
