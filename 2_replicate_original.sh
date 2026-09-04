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

cd figure_replication
# Section 3: standalone Python scripts (Figures 1 and 2)
SEC3_SCRIPTS=(
    "analytical_example_1d.py"
    "analytical_example_2d.py"
)

for script in "${SEC3_SCRIPTS[@]}"; do
    echo "Executing Sec_3/$script..."

    (cd Sec_3 && python3 "$script")

    if [ $? -ne 0 ]; then
        echo "Error encountered while executing Sec_3/$script. Stopping."
        exit 1
    fi

    echo "Successfully finished Sec_3/$script"
    echo "-----------------------------------"
done

# Sections 4 and 5: Jupyter notebooks (remaining figures and tables)
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

rm -rf "$OUTPUT_DIR/2"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/2"

mv Sec_3/*.pdf "$OUTPUT_DIR/2/"
mv Sec_4_5_Fernandes_Phelan/*.pdf "$OUTPUT_DIR/2/"
mv Sec_4_5_Fernandes_Phelan/*.tex "$OUTPUT_DIR/2/"
mv Sec_5_overreporting/*.pdf "$OUTPUT_DIR/2/"
mv Sec_5_overreporting/*.tex "$OUTPUT_DIR/2/"
