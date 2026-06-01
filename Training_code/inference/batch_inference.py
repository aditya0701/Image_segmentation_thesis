
import torch
import tifffile
import numpy as np
import os
import glob
from typing import Optional, Tuple, Union
from tqdm import tqdm
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
from scipy.ndimage import label as ndimage_label, generate_binary_structure
from skimage.morphology import remove_small_objects

# ============================
# CONFIGURATION
# ============================
MODEL_TYPE = "vit_l_lm"
CHECKPOINT_PATH = r"/work/scratch/rawat/models/microsam_training_output_large/checkpoints/vit_l_lm_neo_augmented_Original_v2/best.pt"
INPUT_FOLDER = r"/home/students/rawat/Dataset/Surface_Masks/Surface_Masks/airyscan_processed/"
OUTPUT_FOLDER = r"/home/students/rawat/czi_output/Annotations/Processed_labels/pred_test/test_annotations/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOB_REMOVAL_THRESHOLD = 1450
TILE_SHAPE = (896, 896)
HALO = (64, 64)

# Define the custom transform class that was used during training.
# This is necessary for torch.load (unpickling) to find the class definition.
class ChannelSafeIntensityAug:
    def __call__(self, x):
        # This transform operates on numpy arrays
        for c in range(2):
            if np.random.rand() > 0.5:
                x[c] = x[c] + np.random.randn(*x[c].shape) * 0.05
            if np.random.rand() > 0.5:
                factor = np.random.uniform(0.8, 1.2)
                x[c] = np.clip(x[c] * factor, 0, 1)
        return x

def preprocess_image(raw_data: np.ndarray) -> np.ndarray:
    """Normalizes and prepares a 3D image for inference."""
    print(f"  Original shape: {raw_data.shape}, dtype: {raw_data.dtype}")

    # --- Axis Transposition ---
    if raw_data.ndim == 4:
        if raw_data.shape[0] <= 4:  # (C, Z, Y, X) -> (Z, Y, X, C)
            raw_data = np.transpose(raw_data, (1, 2, 3, 0))
        elif raw_data.shape[1] <= 4:  # (Z, C, Y, X) -> (Z, Y, X, C)
            raw_data = np.transpose(raw_data, (0, 2, 3, 1))
    elif raw_data.ndim == 3: # (Z, Y, X) -> (Z, Y, X, 1)
        raw_data = raw_data[..., np.newaxis]
    
    print(f"  Processed shape (Z, Y, X, C): {raw_data.shape}")

    # --- Normalization ---
    low, high = np.percentile(raw_data, (0.00001, 99.9999))
    if high > low:
        raw_data = (raw_data.astype(np.float32) - low) / (high - low)
        raw_data = np.clip(raw_data, 0, 1)
        raw_data = raw_data * 255 # Scale to [0, 255]
    else:
        raw_data = np.zeros_like(raw_data, dtype=np.float32)

    # --- Convert to 3-Channel RGB ---
    Z, Y, X, C = raw_data.shape
    raw_data_rgb = np.zeros((Z, Y, X, 3), dtype=raw_data.dtype)
    raw_data_rgb[..., :min(C, 2)] = raw_data[..., :min(C, 2)]
    
    print(f"  Final RGB shape for inference: {raw_data_rgb.shape}")
    return raw_data_rgb

def postprocess_masks(predictions_3d: np.ndarray, blob_threshold: int) -> np.ndarray:
    """Applies 3D connected components and blob removal."""
    print("  Starting post-processing...")
    
    # --- 3D Connected Component Labeling ---
    binary_mask = predictions_3d > 0
    structure_3d = generate_binary_structure(3, 3)  # 26-connectivity
    stitched_labels, num_features = ndimage_label(binary_mask, structure=structure_3d)
    print(f"  Found {num_features} connected objects before cleaning.")

    # --- Blob Removal ---
    clean_labels = remove_small_objects(
        stitched_labels.astype(np.uint32), 
        min_size=blob_threshold, 
        connectivity=1
    )
    
    final_count = len(np.unique(clean_labels)) - 1
    print(f"  Found {final_count} objects after removing blobs smaller than {blob_threshold} voxels.")
    return clean_labels

def main():
    """
    Main function to run batch inference on a folder of images.
    """
    print("--- Starting Batch Inference Script ---")
    print(f"Using device: {DEVICE}")
    print(f"Using model: {MODEL_TYPE} with checkpoint: {CHECKPOINT_PATH}")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Find all .tif files in the input folder
    image_paths = glob.glob(os.path.join(INPUT_FOLDER, "*.tif"))
    if not image_paths:
        print(f"Error: No .tif files found in {INPUT_FOLDER}")
        return
        
    print(f"Found {len(image_paths)} images to process.")

    # 1. Initialize the model ONCE
    print("\nLoading model...")
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=MODEL_TYPE,
        checkpoint=CHECKPOINT_PATH,
        device=DEVICE,
        is_tiled=True,
    )
    print("Model loaded successfully.")

    # 2. Loop through each image
    for i, img_path in enumerate(image_paths):
        print(f"\n--- Processing image {i+1}/{len(image_paths)} ---")
        print(f"Input: {img_path}")
        
        base_name = os.path.basename(img_path)
        output_path = os.path.join(OUTPUT_FOLDER, base_name)

        # --- Pre-processing ---
        raw_data = tifffile.imread(img_path)
        image_rgb = preprocess_image(raw_data)
        
        # --- Slice-by-Slice Inference ---
        prediction_stack = []
        print("  Running slice-by-slice inference...")
        for z in tqdm(range(image_rgb.shape[0]), desc="  Processing Slices"):
            img_slice = image_rgb[z]
            pred_2d = automatic_instance_segmentation(
                predictor=predictor,
                segmenter=segmenter,
                input_path=img_slice,
                ndim=2,
                tile_shape=TILE_SHAPE,
                halo=HALO,
            )
            prediction_stack.append(pred_2d)
        
        predictions_3d = np.array(prediction_stack)
        
        # --- Post-processing ---
        final_masks = postprocess_masks(predictions_3d, BLOB_REMOVAL_THRESHOLD)
        
        # --- Save Final Output ---
        print(f"  Saving final result to: {output_path}")
        dtype = np.uint32 if final_masks.max() > 65535 else np.uint16
        tifffile.imwrite(output_path, final_masks.astype(dtype))
        print(f"✓ Saved: {base_name}")

    print("\n--- Batch Inference Complete ---")

if __name__ == "__main__":
    main()
