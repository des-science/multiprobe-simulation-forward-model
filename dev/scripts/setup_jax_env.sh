#!/bin/bash
# Build ~/dlss/jax_env inside the msfm_jax (JAX) container: jax[cuda] from the container,
# CPU-only TF + s2fft + msfm on top. Verifies the JAX-owns-GPU / TF-off-GPU split at the end.
set -eo pipefail

echo "===== container python: $(which python) ($(python --version 2>&1)) ====="

# uv on PATH + cache on SCRATCH (protect HOME quota)
source "$HOME/dlss/bin/env"
export UV_CACHE_DIR="${SCRATCH}/.cache/uv"
# cache (SCRATCH) and venv (HOME) are on different filesystems -> hardlinks unavailable; copy quietly
export UV_LINK_MODE=copy

echo "===== creating venv (inherits container jax[cuda] via --system-site-packages) ====="
# the mounted bashrc auto-activates jax_env if present; drop out of any partial venv first
command -v deactivate >/dev/null 2>&1 && deactivate || true
# start clean so a partially-built venv from an interrupted run can't block/poison the build
rm -rf ~/dlss/jax_env
# pin the container's own interpreter (not a possibly-activated venv python)
uv venv --python /usr/bin/python --system-site-packages --seed --relocatable --link-mode=copy ~/dlss/jax_env
source ~/dlss/jax_env/bin/activate

echo "===== installing CPU-only TensorFlow (fallback to plain tensorflow on aarch64) ====="
if uv pip install "tensorflow-cpu"; then
    echo "installed tensorflow-cpu"
else
    echo "tensorflow-cpu unavailable (likely no aarch64 wheel); falling back to plain tensorflow"
    uv pip install "tensorflow"
fi

echo "===== installing msfm (editable) ====="
uv pip install -e ~/dlss/repos/multiprobe-simulation-forward-model

echo "===== installing s2fft==1.4.0 (CPU-fallback build; pure-JAX path still uses the GPU jaxlib) ====="
# s2fft>=1.2 ships no linux-aarch64 wheel, so it builds from sdist. With nvcc visible its CMake links
# CUDA::cufft_static (absent in this container) and the build fails. Hiding the CUDA compiler makes
# CMake take its NO_CUDA_COMPILER branch (a CPU stub _s2fft, no cufft) -- fine here because sht.py only
# uses method="jax", which runs on the GPU through the container's jaxlib regardless of the C backend.
# plain pip + --no-deps so the container's CUDA jaxlib is NOT shadowed by a PyPI jax/jaxlib (runtime
# deps numpy+jax are already satisfied: numpy from TF, jax from the container). Build isolation still
# pulls jax/nanobind transiently for `from jax import ffi` during the CMake configure.
PATH_NO_CUDA="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v '/cuda' | paste -sd: -)"
env PATH="$PATH_NO_CUDA" CUDACXX="" CUDAToolkit_ROOT="" pip install --no-deps "s2fft==1.4.0"

echo
echo "############################ VERIFICATION ############################"
echo "----- JAX sees the GPU -----"
python -c "import jax; print('jax', jax.__version__, 'devices:', jax.devices())"

echo "----- TF version + GPUs visible BEFORE hiding -----"
python -c "import tensorflow as tf; print('tf', tf.__version__); print('tf GPUs (pre-hide):', tf.config.list_physical_devices('GPU'))"

echo "----- set_tf_cpu_only() hides the GPU from TF -----"
python -c "from msfm.utils import imports; imports.set_tf_cpu_only(); import tensorflow as tf; print('tf GPUs (post-hide):', tf.config.list_physical_devices('GPU'))"

echo "----- s2fft import -----"
python -c "import s2fft; print('s2fft ok')"

echo "----- msfm + sht s2fft backend (end-to-end backend selection) -----"
python -c "import msfm; from msfm.utils import sht; sht.set_backend('s2fft'); print('active SHT backend =', sht.get_backend())"

echo "----- tensorflow_probability must be ABSENT (apps do not need it) -----"
python -c "import importlib.util; print('tfp present:', importlib.util.find_spec('tensorflow_probability') is not None)"

echo "############################ DONE ############################"
