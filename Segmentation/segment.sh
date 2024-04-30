#!/bin/bash

# Function to process input paths
process_input_path() {
    local input_path=$1

    # Kept fixed
    dataset="999"
    config="2d"
    save_prob="--save_probabilities"
    # Folds to use, can be {0,1,2,3,4}. 1+ folds will be ensembled (by averaging predictions)

    # Kept fixed
    python name_handling.py "$input_path"

    for fold in {0..0}; do
        output_path="${input_path}/Segmentation/Fold_${fold}"
        mkdir -p "$output_path"
        nnUNetv2_predict_chrono -i "$input_path" -o "$output_path" -d "$dataset" -c "$config" -f "$fold" $save_prob
        python name_handling.py "$input_path" --revert_seg --segpath "$output_path"
    done

    python name_handling.py "$input_path" --revert
    
    python ensemble.py "$input_path" --alpha 0.90
}

# Add the paths to the nnUNet files
export nnUNet_raw="ChronoRoot_nnUNet/nnUNet_raw"
export nnUNet_preprocessed="ChronoRoot_nnUNet/nnUNet_preprocessed"
export nnUNet_results="ChronoRoot_nnUNet/nnUNet_results"

process_input_path DATA/Videos2023/rpi6_2023-09-25_15-59/1
process_input_path DATA/Videos2023/rpi6_2023-09-25_15-59/2
process_input_path DATA/Videos2023/rpi6_2023-09-25_15-59/3
process_input_path DATA/Videos2023/rpi6_2023-09-25_15-59/4
