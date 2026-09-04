#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate rep

N_PROCS=64  #number of processes to use for parallelization;
cd dptorch
mpirun -np $N_PROCS python run_dpgp.py ++MODEL_NAME=2D_Fernandes_Phelan_feas_set ++STARTING_POINT=NEW ++hydra.run.dir=runs/2D_Fernandes_Phelan_feas_set/model ++num_iterations=314
mpirun -np $N_PROCS python run_dpgp.py ++MODEL_NAME=2D_baseline_to_limited_or    ++STARTING_POINT=NEW ++hydra.run.dir=runs/2D_baseline_to_limited_or/model    ++num_iterations=5939
mpirun -np $N_PROCS python run_dpgp.py ++MODEL_NAME=2D_limited_overreporting     ++STARTING_POINT=NEW ++hydra.run.dir=runs/2D_limited_overreporting/model     ++num_iterations=4379
mpirun -np $N_PROCS python run_dpgp.py ++MODEL_NAME=4D_baseline_to_limited_or    ++STARTING_POINT=NEW ++hydra.run.dir=runs/4D_baseline_to_limited_or/model    ++num_iterations=3777
mpirun -np $N_PROCS python run_dpgp.py ++MODEL_NAME=4D_limited_overreporting     ++STARTING_POINT=NEW ++hydra.run.dir=runs/4D_limited_overreporting/model     ++num_iterations=2999
mpirun -np $N_PROCS python run_dpgp.py ++MODEL_NAME=10D_Fernandes_Phelan         ++STARTING_POINT=NEW ++hydra.run.dir=runs/10D_Fernandes_Phelan/model         ++num_iterations=4319

# run post processing
python post_process.py      ++RUN_DIR=2D_Fernandes_Phelan_feas_set/model ++CHECKPOINT_FILE=Iter_314.pth
python post_process.py      ++RUN_DIR=2D_baseline_to_limited_or/model    ++CHECKPOINT_FILE=Iter_5939.pth
python post_process.py      ++RUN_DIR=2D_limited_overreporting/model     ++CHECKPOINT_FILE=Iter_4379.pth
python post_process.py      ++RUN_DIR=4D_baseline_to_limited_or/model    ++CHECKPOINT_FILE=Iter_3777.pth
python post_process.py      ++RUN_DIR=4D_limited_overreporting/model     ++CHECKPOINT_FILE=Iter_2999.pth
python post_process.py      ++RUN_DIR=10D_Fernandes_Phelan/model         ++CHECKPOINT_FILE=Iter_4319.pth

# --- 2D_Fernandes_Phelan (Section 4.5.1) ---
cp postprocess/2D_Fernandes_Phelan/model/simulation_2999.txt      ../figure_replication/Sec_4_5_Fernandes_Phelan/data/2D/
cp postprocess/2D_Fernandes_Phelan/model/V_func_error_0_2999.txt  ../figure_replication/Sec_4_5_Fernandes_Phelan/data/2D/
cp runs/2D_Fernandes_Phelan/model/Iter_2998.pth                   ../figure_replication/Sec_4_5_Fernandes_Phelan/data/2D/
cp runs/2D_Fernandes_Phelan/model/Iter_2999.pth                   ../figure_replication/Sec_4_5_Fernandes_Phelan/data/2D/

# --- 2D_Fernandes_Phelan_feas_set (Section 4.5.1, feasible set for Figure 5) ---
# The plotting notebook expects the historical filename V_func_all_314_0.txt;
# the post-processing writes the same data (discrete state 0) as V_set_0.txt, so rename while copying:
cp postprocess/2D_Fernandes_Phelan_feas_set/model/V_set_0.txt     ../figure_replication/Sec_4_5_Fernandes_Phelan/data/feas_set/V_func_all_314_0.txt
cp runs/2D_Fernandes_Phelan_feas_set/model/Iter_314.pth           ../figure_replication/Sec_4_5_Fernandes_Phelan/data/feas_set/

# --- 10D_Fernandes_Phelan (Section 4.5.2) ---
cp postprocess/10D_Fernandes_Phelan/model/simulation_4319.txt     ../figure_replication/Sec_4_5_Fernandes_Phelan/data/10D/
cp postprocess/10D_Fernandes_Phelan/model/V_func_error_4319.txt   ../figure_replication/Sec_4_5_Fernandes_Phelan/data/10D/
cp postprocess/10D_Fernandes_Phelan/model/V_raw_4319.txt          ../figure_replication/Sec_4_5_Fernandes_Phelan/data/10D/
cp runs/10D_Fernandes_Phelan/model/Iter_4318.pth                  ../figure_replication/Sec_4_5_Fernandes_Phelan/data/10D/
cp runs/10D_Fernandes_Phelan/model/Iter_4319.pth                  ../figure_replication/Sec_4_5_Fernandes_Phelan/data/10D/

# --- 2D_baseline_to_limited_or (Section 5.1, baseline) ---
cp postprocess/2D_baseline_to_limited_or/model/simulation_5939.txt    ../figure_replication/Sec_5_overreporting/data/2D/baseline/
cp postprocess/2D_baseline_to_limited_or/model/V_func_error_5939.txt  ../figure_replication/Sec_5_overreporting/data/2D/baseline/
cp runs/2D_baseline_to_limited_or/model/Iter_5938.pth                 ../figure_replication/Sec_5_overreporting/data/2D/baseline/
cp runs/2D_baseline_to_limited_or/model/Iter_5939.pth                 ../figure_replication/Sec_5_overreporting/data/2D/baseline/

# --- 2D_limited_overreporting (Section 5.1, limited overreporting) ---
cp postprocess/2D_limited_overreporting/model/simulation_4379.txt     ../figure_replication/Sec_5_overreporting/data/2D/overreporting/
cp postprocess/2D_limited_overreporting/model/V_func_error_4379.txt   ../figure_replication/Sec_5_overreporting/data/2D/overreporting/
cp runs/2D_limited_overreporting/model/Iter_4378.pth                  ../figure_replication/Sec_5_overreporting/data/2D/overreporting/
cp runs/2D_limited_overreporting/model/Iter_4379.pth                  ../figure_replication/Sec_5_overreporting/data/2D/overreporting/

# --- 4D_baseline_to_limited_or (Section 5.2, baseline) ---
cp postprocess/4D_baseline_to_limited_or/model/simulation_3777.txt    ../figure_replication/Sec_5_overreporting/data/4D/baseline/
cp postprocess/4D_baseline_to_limited_or/model/V_func_error_3777.txt  ../figure_replication/Sec_5_overreporting/data/4D/baseline/
cp runs/4D_baseline_to_limited_or/model/Iter_3776.pth                 ../figure_replication/Sec_5_overreporting/data/4D/baseline/
cp runs/4D_baseline_to_limited_or/model/Iter_3777.pth                 ../figure_replication/Sec_5_overreporting/data/4D/baseline/

# --- 4D_limited_overreporting (Section 5.2, limited overreporting) ---
cp postprocess/4D_limited_overreporting/model/simulation_2999.txt     ../figure_replication/Sec_5_overreporting/data/4D/overreporting/
cp postprocess/4D_limited_overreporting/model/V_func_error_2999.txt   ../figure_replication/Sec_5_overreporting/data/4D/overreporting/
cp runs/4D_limited_overreporting/model/Iter_2998.pth                  ../figure_replication/Sec_5_overreporting/data/4D/overreporting/
cp runs/4D_limited_overreporting/model/Iter_2999.pth                  ../figure_replication/Sec_5_overreporting/data/4D/overreporting/

cd ../figure_replication


NOTEBOOKS=(
    "Sec_4_5_Fernandes_Phelan/2D_run_simulation.ipynb"
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
    "Sec_4_5_Fernandes_Phelan/Plots_Table_2D_fernandes_phelan.ipynb"
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

rm -rf "$OUTPUT_DIR/5"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/5"

mv Sec_4_5_Fernandes_Phelan/*.pdf "$OUTPUT_DIR/5/"
mv Sec_4_5_Fernandes_Phelan/*.tex "$OUTPUT_DIR/5/"
mv Sec_5_overreporting/*.pdf "$OUTPUT_DIR/5/"
mv Sec_5_overreporting/*.tex "$OUTPUT_DIR/5/"