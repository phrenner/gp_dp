#!/bin/bash

#SBATCH -p parallel
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=23:00:00
#SBATCH --mem=128G

source /etc/profile

export OMP_NUM_THREADS=1

module add miniforge/20251003
module add openmpi/5.0.8-gcc
eval "$(mamba shell hook --shell bash)"
mamba activate
mamba activate /mmfs1/storage/users/renner/conda/pytorch

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export MPI4PY_RC_THREADS=0

mpirun --bind-to none python run_dpgp.py
