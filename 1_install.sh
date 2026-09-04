#!/bin/bash
set -e
eval "$(conda shell.bash hook)"

conda create --name rep python=3.9.23 -y
conda activate rep
conda install cmake -y
conda install conda-forge::openmpi -y
conda install -c conda-forge parallel -y

# Install all Python dependencies, including Tasmanian (required for Figure 2 only)
pip install -r requirements.txt

mkdir -p ~/.local/lib/
ln -sf $CONDA_PREFIX/lib/libtasmaniansparsegrid.so ~/.local/lib/
ln -sf $CONDA_PREFIX/lib/libtasmaniandream.so ~/.local/lib/
ln -sf $CONDA_PREFIX/lib/libtasmaniancaddons.so ~/.local/lib/

# Verify the installation
python -c "import torch, gpytorch, scipy; print('core OK')"
python -c "import Tasmanian; print('Tasmanian OK')"