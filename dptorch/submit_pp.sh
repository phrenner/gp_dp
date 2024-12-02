#!/bin/bash

#SBATCH -p serial
#SBATCH -J post_process
#SBATCH --time=03:00:00
#SBATCH --mem=16G

source /etc/profile

export OMP_NUM_THREADS=1

module add openblas
module add binutils/2.26
module add miniconda/2023.06
source activate /storage/users/renner/conda/pytorch

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

python post_process.py RUN_DIR=AS_one_gp/model