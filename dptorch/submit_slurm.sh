#!/bin/bash

#SBATCH -p parallel
#SBATCH -C node_type=64core256G
#node_type=10Geth*   40core192G
#SBATCH --exclusive
#40core192G
#SBATCH --nodes=1
#SBATCH --time=23:00:00

source /etc/profile

export OMP_NUM_THREADS=1

module add openblas
module add binutils/2.26
module add miniconda/2023.06
source activate /storage/users/renner/conda/pytorch

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

mpirun python run_dpgp.py
