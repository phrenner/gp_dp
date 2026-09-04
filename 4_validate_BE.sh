#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate rep

# clean
rm figure_replication/Sec_4_5_Fernandes_Phelan/*.tex
rm figure_replication/Sec_4_5_Fernandes_Phelan/*.pdf
rm figure_replication/Sec_5_overreporting/*.tex
rm figure_replication/Sec_5_overreporting/*.pdf
rm figure_replication/Sec_4_5_Fernandes_Phelan/data/10D/Iter_{4320..4330}.pth
rm figure_replication/Sec_4_5_Fernandes_Phelan/data/feas_set/Iter_{315..325}.pth
rm figure_replication/Sec_5_overreporting/data/2D/baseline/Iter_{5940..5950}.pth
rm figure_replication/Sec_5_overreporting/data/2D/overreporting/Iter_{4380..4390}.pth
rm figure_replication/Sec_5_overreporting/data/4D/baseline/Iter_{3778..3788}.pth
rm figure_replication/Sec_5_overreporting/data/4D/overreporting/Iter_{3000..3010}.pth

N_RESTARTS=1  #no. restarts in each optimization problem; 1 is just using the warm start; without BAL this is sufficient
N_ITS=10  #no. value function iterations to run; 10 is sufficient to validate that the Bellman equation holds for the warm start;
N_PROCS=16  #number of processes to use for parallelization;
cd dptorch
# mpirun -np $N_PROCS python run_dpgp.py ++MODEL_NAME=2D_Fernandes_Phelan ++STARTING_POINT=Iter_2999.pth ++hydra.run.dir=runs/2D_Fernandes_Phelan/restart  ++ENABLE_BAL=false ++ENABLE_HOWARD=false ++N_RESTARTS=$N_RESTARTS ++num_iterations=$N_ITS
mpirun -np $N_PROCS python run_dpgp.py ++MODEL_NAME=2D_limited_overreporting ++STARTING_POINT=Iter_4379.pth ++hydra.run.dir=runs/2D_limited_overreporting/restart  ++ENABLE_BAL=false ++ENABLE_HOWARD=false ++N_RESTARTS=$N_RESTARTS ++num_iterations=$N_ITS 
mpirun -np $N_PROCS python run_dpgp.py ++MODEL_NAME=2D_baseline_to_limited_or ++STARTING_POINT=Iter_5939.pth ++hydra.run.dir=runs/2D_baseline_to_limited_or/restart  ++ENABLE_BAL=false ++ENABLE_HOWARD=false ++N_RESTARTS=$N_RESTARTS ++num_iterations=$N_ITS 
mpirun -np $N_PROCS python run_dpgp.py ++MODEL_NAME=4D_limited_overreporting ++STARTING_POINT=Iter_2999.pth ++hydra.run.dir=runs/4D_limited_overreporting/restart  ++ENABLE_BAL=false ++ENABLE_HOWARD=false ++N_RESTARTS=$N_RESTARTS ++num_iterations=$N_ITS 
mpirun -np $N_PROCS python run_dpgp.py ++MODEL_NAME=4D_baseline_to_limited_or ++STARTING_POINT=Iter_3777.pth ++hydra.run.dir=runs/4D_baseline_to_limited_or/restart  ++ENABLE_BAL=false ++ENABLE_HOWARD=false ++N_RESTARTS=$N_RESTARTS ++num_iterations=$N_ITS 
mpirun -np $N_PROCS python run_dpgp.py ++MODEL_NAME=10D_Fernandes_Phelan ++STARTING_POINT=Iter_4319.pth ++hydra.run.dir=runs/10D_Fernandes_Phelan/restart  ++ENABLE_BAL=false ++ENABLE_HOWARD=false ++N_RESTARTS=$N_RESTARTS ++num_iterations=$N_ITS 

parallel -j $N_PROCS :::: post_processing_commands.txt 



cd ../figure_replication
NOTEBOOKS=(
    "Sec_5_overreporting/2D_simulation_and_error.ipynb"
    "Sec_5_overreporting/4D_simulation_and_error.ipynb"
)

for nb in "${NOTEBOOKS[@]}"; do
    echo "Executing $nb..."
    
    jupyter nbconvert --to notebook --execute --inplace "$nb"
    
    if [ $? -ne 0 ]; then
        echo "Error encountered while executing $nb. Stopping the sequence."
        exit 1
    fi
    
    echo "Successfully finished $nb"
    echo "-----------------------------------"
done

echo "All notebooks executed successfully!"

NOTEBOOKS=(
    "Sec_4_5_Fernandes_Phelan/Table_10D_fernandes_phelan.ipynb"
    "Sec_5_overreporting/Plot_Table_2D_overreporting.ipynb"
    "Sec_5_overreporting/Plot_Table_4D_overreporting.ipynb"
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

rm -rf "$OUTPUT_DIR/4"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/4"

mv Sec_4_5_Fernandes_Phelan/*.pdf "$OUTPUT_DIR/4/"
mv Sec_4_5_Fernandes_Phelan/*.tex "$OUTPUT_DIR/4/"
mv Sec_5_overreporting/*.pdf "$OUTPUT_DIR/4/"
mv Sec_5_overreporting/*.tex "$OUTPUT_DIR/4/"
