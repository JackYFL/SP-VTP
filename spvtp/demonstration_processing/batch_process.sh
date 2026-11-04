#!/bin/bash
# Batch process all tasks under ./data/
# Usage: cd spvtp && bash demonstration_processing/batch_process.sh

# Change to spvtp directory (script's parent's parent)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_DIR="./data"

for task_dir in "$DATA_DIR"/*/*; do
    # Skip non-directories
    [ -d "$task_dir" ] || continue

    # Extract relative path, e.g. scene1/put_fork1_to_bowl1
    rel_path="${task_dir#$DATA_DIR/}"

    input_dir="./data/${rel_path}"
    output_dir="./output_data/${rel_path}/recording_output_processed"
    output_zarr="${output_dir}/recording_output.zarr.zip"

    echo "========================================"
    echo "Processing: ${rel_path}"
    echo "  input_dir:  ${input_dir}"
    echo "  output_dir: ${output_dir}"
    echo "========================================"

    python demonstration_processing/process_demos_iphone_gopro.py \
        input_dir="${input_dir}" \
        output_dir="${output_dir}" \
        output_policy_replay_buffer_zarr="${output_zarr}"

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed processing ${rel_path}"
        # Continue with next task instead of stopping
    fi
done

echo "Batch processing complete."
