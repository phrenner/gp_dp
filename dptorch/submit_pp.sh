#!/bin/bash

#SBATCH -p serial
#SBATCH -J post_process
#SBATCH --time=03:00:00
#SBATCH --mem=16G

source /etc/profile

# --- 1) Extract variables from config/config.yaml ---
# Path to your config file
CONFIG_PATH="config/config.yaml"

# Extracting MODEL_NAME and STARTING_POINT (e.g., Iter_289.pth)
# Using awk to get the value after the colon
MODEL_NAME=$(grep 'MODEL_NAME:' $CONFIG_PATH | awk '{print $2}')
ITER_FILE=Iter_8069.pth

# Define the relative run directory (excluding 'runs/')
# This is what will be passed to the python script: 10D_Fernandes_Phelan/model
PP_RUN_DIR="${MODEL_NAME}/model"

# --- 2) Copy the .pth file to the postprocess folder ---
# Source: runs/10D_Fernandes_Phelan/model/Iter_X.pth
# Destination: postprocess/10D_Fernandes_Phelan/model/
SOURCE_PATH="runs/${PP_RUN_DIR}/${ITER_FILE}"
DEST_DIR="postprocess/${PP_RUN_DIR}"

echo "Creating destination directory: $DEST_DIR"
mkdir -p "$DEST_DIR"

echo "Copying $SOURCE_PATH to $DEST_DIR"
if [ -f "$SOURCE_PATH" ]; then
    cp "$SOURCE_PATH" "$DEST_DIR/"
else
    echo "Error: Source file $SOURCE_PATH not found."
    exit 1
fi

# --- Environment Setup ---
export OMP_NUM_THREADS=1

module add miniforge/20251003
module add openmpi/5.0.8-gcc
eval "$(mamba shell hook --shell bash)"
mamba activate
mamba activate /mmfs1/storage/users/renner/conda/pytorch

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}


# --- 3) Execute Python with the dynamic RUN_DIR ---
python post_process.py RUN_DIR="$PP_RUN_DIR" CHECKPOINT_FILE="$ITER_FILE"
