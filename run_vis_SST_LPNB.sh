#!/bin/bash

# Directory containing the data files
DATA_DIR="/scratch/momenika/MITgcm/collection_runs_llc1080_LPNB"

THRESH_N=38454

# Loop over all files matching U.[N].data
for file in "$DATA_DIR"/U.[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].data; do
    # Extract the 10-digit number N
    N=$(basename "$file" | sed -E 's/^U\.([0-9]{10})\.data$/\1/')
    
    # Skip if N <= THRESH_N (force base-10 to avoid octal issues with leading zeros)
    if (( 10#$N <= 10#$THRESH_N )); then
      continue
    fi


    # Run the command
    python vis_llc.py \
        "$DATA_DIR/Theta.$N.data" \
        --nx 1080 \
        --level 1 \
        --cmap turbo \
        --vmin 0 \
        --vmax 36 \
        --startyr 2023 \
        --startmo 01 \
        --startdy 01 \
        --deltaT 600 \
        --rf "$PROJECT/llc_1080/llc_1080_grid_files/RF.data" \
        --nk 173 \
        --bathy "$PROJECT/llc1080_template/GEBCO2025_on_LLC1080_v13.bin" \
        --ice-shelf "$PROJECT/llc1080_template/BEDMACHINE_on_LLC1080_v13.bin" \
        --sea-ice "$DATA_DIR/SIheff.$N.data" \
        --no-show \
        --o "/scratch/momenika/comparison_1080/SST_LPNB/surface_temperature.$N.png"
done
