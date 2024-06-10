#!/bin/bash

# Function to revert changes for input paths
revert_input_path() {
    local input_path=$1
    python name_handling.py "$input_path" --revert
}

# Check if a filename is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_file>"
    exit 1
fi

# Get the filename from the first argument
filename=$1

# Loop through each line in the specified file and revert the path
while IFS= read -r line; do
    revert_input_path "$line"
done < "$filename"
