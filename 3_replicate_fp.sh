#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate rep

# clean
rm figure_replication/Sec_4_5_Fernandes_Phelan/*.tex
rm figure_replication/Sec_4_5_Fernandes_Phelan/*.pdf
rm figure_replication/Sec_4_5_Fernandes_Phelan/data/2D/Iter_{0..2999}.pth

N_PROCS=16  #number of processes to use for parallelization;
cd dptorch
mpirun -np $N_PROCS python run_dpgp.py ++MODEL_NAME=2D_Fernandes_Phelan ++STARTING_POINT=NEW ++hydra.run.dir=runs/2D_Fernandes_Phelan/model  ++ENABLE_BAL=true ++ENABLE_HOWARD=true ++num_iterations=2999
python post_process_2DFP.py RUN_DIR=2D_Fernandes_Phelan/model
cp postprocess/2D_Fernandes_Phelan/model/simulation_2999.txt      ../figure_replication/Sec_4_5_Fernandes_Phelan/data/2D/
cp postprocess/2D_Fernandes_Phelan/model/V_func_error_0_2999.txt  ../figure_replication/Sec_4_5_Fernandes_Phelan/data/2D/
cp runs/2D_Fernandes_Phelan/model/Iter_2998.pth                   ../figure_replication/Sec_4_5_Fernandes_Phelan/data/2D/
cp runs/2D_Fernandes_Phelan/model/Iter_2999.pth                   ../figure_replication/Sec_4_5_Fernandes_Phelan/data/2D/

cd ../figure_replication


# Sections 4 and 5: Jupyter notebooks (remaining figures and tables)
NOTEBOOKS=(
    "Sec_4_5_Fernandes_Phelan/Plots_Table_2D_fernandes_phelan.ipynb"
)

for nb in "${NOTEBOOKS[@]}"; do
    echo "Executing $nb..."

    jupyter nbconvert --to notebook --execute --inplace "$nb"

    if [ $? -ne 0 ]; then
        echo "Error encountered while executing $nb. Stopping."
        exit 1
    fi

    echo "Successfully finished $nb"
    echo "-----------------------------------"
done

echo "All scripts and notebooks executed successfully!"

OUTPUT_DIR="figure_and_table_outputs"

rm -rf "$OUTPUT_DIR/3"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/3"

mv Sec_4_5_Fernandes_Phelan/*.pdf "$OUTPUT_DIR/3/"
mv Sec_4_5_Fernandes_Phelan/*.tex "$OUTPUT_DIR/3/"