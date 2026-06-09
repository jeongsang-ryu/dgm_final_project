#!/usr/bin/env bash
# Source this: `source env/activate.sh`
# Sets up the dgm conda env with the env vars StyleGAN3 needs to JIT-compile its CUDA ops.
source /home/js/anaconda3/etc/profile.d/conda.sh
conda activate dgm
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/gcc"
export CXX="$CONDA_PREFIX/bin/g++"
# conda-linker rpath ordering (so libstdc++ from the env wins over the system one)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
