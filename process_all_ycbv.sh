#!/bin/bash

# Bash script to process all objects from 1 to 21 using svraster training and mesh extraction
# This script runs training and mesh extraction for both segmented and surface variants

# Base path where all object directories are located
BASE_PATH="/home/stefan/Projects/Grounded-SAM-2-zeroshop/dataset"

# Set to true or false to enable/disable processing of each variant
PROCESS_SURFACE=true
PROCESS_SEGMENTED=false
POSTPROCESSING=true

# Choose the parent folder for variants: 'mast3r-sfm' or 'vggt'
VARIANT_PARENT="mast3r-sfm"  # or 'vggt'

# Path to the training and mesh extraction scripts
TRAIN_SCRIPT="train.py"
EXTRACT_SCRIPT="extract_mesh.py"

# Config files
SEGMENTED_CONFIG="cfg/ycbv_segmented.yaml"
SURFACE_CONFIG="cfg/ycbv_surface.yaml"

# Check if scripts exist
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script not found at $TRAIN_SCRIPT"
    exit 1
fi

if [ ! -f "$EXTRACT_SCRIPT" ]; then
    echo "Error: Mesh extraction script not found at $EXTRACT_SCRIPT"
    exit 1
fi

# Check if config files exist
if [ ! -f "$SEGMENTED_CONFIG" ]; then
    echo "Error: Segmented config not found at $SEGMENTED_CONFIG"
    exit 1
fi

if [ ! -f "$SURFACE_CONFIG" ]; then
    echo "Error: Surface config not found at $SURFACE_CONFIG"
    exit 1
fi

# Check if the base path exists
if [ ! -d "$BASE_PATH" ]; then
    echo "Error: Base path not found at $BASE_PATH"
    exit 1
fi

echo "Starting svraster processing for objects 1 to 21..."
echo "Base path: $BASE_PATH"
echo "Training script: $TRAIN_SCRIPT"
echo "Mesh extraction script: $EXTRACT_SCRIPT"
echo "Segmented config: $SEGMENTED_CONFIG"
echo "Surface config: $SURFACE_CONFIG"
echo "========================================"

# Counter for successful and failed processing
success_count=0
failed_count=0
failed_objects=()

# Function to process a single variant (segmented or surface)
process_variant() {
    local obj_path="$1"
    local variant="$2"
    local config_file="$3"
    local obj_num="$4"

    local variant_path="$obj_path/train_pbr/$VARIANT_PARENT/$variant"

    echo "  Processing $variant variant..."
    echo "  Variant path: $variant_path"

    # Check if the variant directory exists
    if [ ! -d "$variant_path" ]; then
        echo "  Warning: $variant directory not found: $variant_path"
        return 1
    fi

    # Step 1: Training
    echo "  Step 1: Training $variant..."
    echo "  Running: python $TRAIN_SCRIPT --source_path $variant_path --model_path $variant_path --cfg_files $config_file"

    if ! python "$TRAIN_SCRIPT" --source_path "$variant_path" --model_path "$variant_path" --cfg_files "$config_file"; then
        echo "  ✗ Failed to train $variant for obj_$obj_num"
        return 1
    fi

    echo "  ✓ Training completed for $variant"

    # Step 2: Mesh extraction
    echo "  Step 2: Extracting mesh for $variant..."
    echo "  Running: python $EXTRACT_SCRIPT $variant_path --use_vert_color --progressive --bbox_scale 1.2 --use_clean"

    if ! python "$EXTRACT_SCRIPT" "$variant_path" --use_vert_color --progressive --bbox_scale 1.2 --use_clean; then
        echo "  ✗ Failed to extract mesh for $variant of obj_$obj_num"
        return 1
    fi

    echo "  ✓ Mesh extraction completed for $variant"

    # Post-processing mesh if enabled
    if [ "$POSTPROCESSING" = true ]; then
        echo "  Step 3: Post-processing mesh for $variant..."

        # Activate pymeshlab environment
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate postprocess

        bundler_file="$variant_path/images/scene.bundle.out"
        bundler_txt="$variant_path/images/scene.list.txt"
        object_info_json="$obj_path/scene/output/object_info.json"

        # Find mesh file for svraster (latest .ply in mesh/latest)
        mesh_dir="$variant_path/mesh/latest"
        mesh_file=""
        if [ -d "$mesh_dir" ]; then
            mesh_file=$(find "$mesh_dir" -maxdepth 1 -type f -name "*.ply" | head -n 1)
        fi

        if [ -z "$mesh_file" ]; then
            echo "  Warning: No mesh file found for post-processing in $mesh_dir"
            return 1
        fi

        if [ ! -f "$bundler_file" ] || [ ! -f "$bundler_txt" ]; then
            echo "  Warning: Bundler files not found for post-processing."
            return 1
        fi

        if [ ! -f "$object_info_json" ]; then
            echo "  Warning: object_info.json not found for post-processing."
            return 1
        fi

        echo "  Running: python postprocess.py --mesh \"$mesh_file\" --bundler \"$bundler_file\" --bundler_txt \"$bundler_txt\" --object_info_json \"$object_info_json\""
        if ! python postprocess.py --mesh "$mesh_file" --bundler "$bundler_file" --bundler_txt "$bundler_txt" --object_info_json "$object_info_json"; then
            echo "  ✗ Failed to post-process mesh for $variant of obj_$obj_num"
            return 1
        fi

        echo "  ✓ Post-processing completed for $variant"

        # Return to original environment
        conda activate svraster
    fi

    return 0
}


# Loop through objects 1 to 21
for i in {1..21}; do
    # Format the object number with leading zeros (6 digits)
    obj_num=$(printf "%06d" $i)
    obj_path="$BASE_PATH/obj_$obj_num"

    echo ""
    echo "Processing object $i (obj_$obj_num)..."
    echo "Object path: $obj_path"

    # Check if the object directory exists
    if [ ! -d "$obj_path" ]; then
        echo "Warning: Object directory not found: $obj_path"
        echo "Skipping obj_$obj_num"
        ((failed_count++))
        failed_objects+=("obj_$obj_num (directory not found)")
        continue
    fi

    # Track success for this object
    obj_success=true

    # Process surface variant if enabled
    if [ "$PROCESS_SURFACE" = true ]; then
        echo ""
        echo "--- Processing SURFACE variant for obj_$obj_num ---"
        if ! process_variant "$obj_path" "surface" "$SURFACE_CONFIG" "$obj_num"; then
            echo "Failed to process surface variant for obj_$obj_num"
            obj_success=false
        fi
    else
        echo "Skipping SURFACE variant for obj_$obj_num (PROCESS_SURFACE=false)"
    fi

    # Process segmented variant if enabled
    if [ "$PROCESS_SEGMENTED" = true ]; then
        echo ""
        echo "--- Processing SEGMENTED variant for obj_$obj_num ---"
        if ! process_variant "$obj_path" "segmented" "$SEGMENTED_CONFIG" "$obj_num"; then
            echo "Failed to process segmented variant for obj_$obj_num"
            obj_success=false
        fi
    else
        echo "Skipping SEGMENTED variant for obj_$obj_num (PROCESS_SEGMENTED=false)"
    fi

    # Update counters
    if [ "$obj_success" = true ]; then
        echo "✓ Successfully processed enabled variants for obj_$obj_num"
        ((success_count++))
    else
        echo "✗ Failed to process one or more enabled variants for obj_$obj_num"
        ((failed_count++))
        failed_objects+=("obj_$obj_num")
    fi

    echo "========================================"
done

echo ""
echo "========================================"
echo "SVRASTER PROCESSING COMPLETE"
echo "========================================"
echo "Total objects processed: $((success_count + failed_count))"
echo "Successful: $success_count"
echo "Failed: $failed_count"

if [ $failed_count -gt 0 ]; then
    echo ""
    echo "Failed objects:"
    for failed_obj in "${failed_objects[@]}"; do
        echo "  - $failed_obj"
    done
fi

echo ""
if [ $failed_count -eq 0 ]; then
    echo "All objects processed successfully!"
    echo "Training and mesh extraction completed for enabled variants."
    echo "Results are saved in the respective model paths under $VARIANT_PARENT:"
    echo "  - {object}/train_pbr/$VARIANT_PARENT/segmented/"
    echo "  - {object}/train_pbr/$VARIANT_PARENT/surface/"
    exit 0
else
    echo "Some objects failed to process. Check the logs above for details."
    exit 1
fi
