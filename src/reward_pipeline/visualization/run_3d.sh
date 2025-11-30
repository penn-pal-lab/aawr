#!/bin/bash

BASE_DIR="~/projects/aawr/data/"
NUM_SUCCESS=5
NUM_FAILURE=5
OUTPUT_DIR="./visuals"
DATASET_FILE="./dataset.txt"

mkdir -p $OUTPUT_DIR

# Read dataset.txt line by line
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines
    if [ -z "$line" ]; then
        continue
    fi
    
    # Determine RL_MODE from path (offline_data or online_data)
    if [[ $line == offline_data/* ]]; then
        RL_MODE="offline"
    elif [[ $line == online_data/* ]]; then
        RL_MODE="online"
    else
        echo "Warning: Cannot determine RL_MODE for $line, skipping..."
        continue
    fi
    
    INPUT_DIR="$BASE_DIR/$line"
    
    if [ ! -d "$INPUT_DIR" ]; then
        echo "Warning: Directory does not exist: $INPUT_DIR, skipping..."
        continue
    fi
    
    echo "========================================="
    echo "Processing: $line"
    echo "Mode: $RL_MODE"
    echo "Input: $INPUT_DIR"
    echo "========================================="
    
    python vis_3d_traj.py \
        --input_dir $INPUT_DIR \
        --num_success $NUM_SUCCESS \
        --num_failure $NUM_FAILURE \
        --light_bg \
        --seed 42
    
    INPUT_BASENAME=$(basename $INPUT_DIR)
    cp $INPUT_DIR/3D_traj/traj_static_$NUM_SUCCESS\_$NUM_FAILURE.png $OUTPUT_DIR/${RL_MODE}_${INPUT_BASENAME}_${NUM_SUCCESS}_${NUM_FAILURE}.png 
    cp $INPUT_DIR/3D_traj/traj_rotate_$NUM_SUCCESS\_$NUM_FAILURE.mp4 $OUTPUT_DIR/${RL_MODE}_${INPUT_BASENAME}_${NUM_SUCCESS}_${NUM_FAILURE}.mp4
    
    echo "Saved to: ${RL_MODE}_${INPUT_BASENAME}_${NUM_SUCCESS}_${NUM_FAILURE}.png"
    echo "Saved to: ${RL_MODE}_${INPUT_BASENAME}_${NUM_SUCCESS}_${NUM_FAILURE}.mp4"
    echo ""
    
done < "$DATASET_FILE"

echo "========================================="
echo "All datasets processed!"
echo "========================================="
