# List of data roots to process
DATA_ROOTS=(
    "~/projects/aawr/data/offline_data/shelf_cabinet/data_0520"
    "~/projects/aawr/data/offline_data/complex/data_0520"
)

# Process each data root
for DATA_ROOT in "${DATA_ROOTS[@]}"; do
    echo "Processing $DATA_ROOT..."
    python reward_pipeline/scripts/calculate_mislabel.py "$DATA_ROOT"
    echo "Finished processing $DATA_ROOT"
    echo "----------------------------------------"
done