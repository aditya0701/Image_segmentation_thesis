import sys
import os
import glob
import argparse
import numpy as np
import tifffile
from tqdm import tqdm
from scipy import ndimage

# --- 1. CONFIGURATION ---
_parser = argparse.ArgumentParser()
_parser.add_argument("--dataset", type=str, default="all",
                     choices=["all", "dog", "original", "psf"])
_args = _parser.parse_args()

DATASET_CONFIG = {
    "all":      ("/home/students/rawat/Final_dataset/all",      "/home/students/rawat/training_data/training_data_All"),
    "dog":      ("/home/students/rawat/Final_dataset/DOG",      "/home/students/rawat/training_data/training_data_DOG"),
    "original": ("/home/students/rawat/Final_dataset/Original", "/home/students/rawat/training_data/training_data_Original"),
    "psf":      ("/home/students/rawat/Final_dataset/PSF",      "/home/students/rawat/training_data/training_data_PSF"),
}

INPUT_ROOT, SAVE_ROOT = DATASET_CONFIG[_args.dataset]
print(f"Dataset : {_args.dataset}")
print(f"Input   : {INPUT_ROOT}")
print(f"Output  : {SAVE_ROOT}")

# Define which sets to process
PROCESS_SETS = ["train", "validation"]

def find_image_mask_pairs(base_dir, set_name):
    """
    Finds all matching image variations for each unique mask using a "starts with" strategy.
    Both train and validation use images/ and masks/ subdirectories.
    """
    pairs = []

    # Both sets use the same images/ masks/ structure
    if set_name in ("train", "validation"):
        image_dir = os.path.join(base_dir, set_name, "images")
        mask_dir  = os.path.join(base_dir, set_name, "masks")
        image_files = glob.glob(os.path.join(image_dir, "*.tif")) + glob.glob(os.path.join(image_dir, "*.ome.tif"))
        mask_files  = glob.glob(os.path.join(mask_dir,  "*.tif")) + glob.glob(os.path.join(mask_dir,  "*.ome.tif"))
    else:
        return []

    # --- DEBUGGING ---
    print(f"-> Debug: Found {len(image_files)} image files and {len(mask_files)} mask files in '{set_name}'.")
    if not mask_files:
        print(f"-> Debug: No masks found in expected directory for '{set_name}'. Check path and file extensions.")
    # --- END DEBUGGING ---

    # 2. Iterate through each mask and use it as the 'source of truth'
    for mask_path in mask_files:
        mask_filename = os.path.basename(mask_path)
        
        # Create the base name by removing the known mask suffix and extension
        base_name_for_matching = mask_filename.replace("_mask.tif", "").replace("_mask.ome.tif", "")
        
        # --- DEBUGGING ---
        print(f"--> Debug: Processing mask '{mask_filename}'. Generated prefix for matching: '{base_name_for_matching}'")
        # --- END DEBUGGING ---

        # Find all image files that start with this exact base name
        matching_images = [
            p for p in image_files 
            if os.path.basename(p).startswith(base_name_for_matching)
        ]
        
        if matching_images:
            # The base name for saving should be the clean name from the mask
            pairs.append((matching_images, mask_path, base_name_for_matching, set_name))
        else:
            # This warning is now part of the standard flow, but good for debugging.
            print(f"⚠️ Warning: No images found for mask {mask_filename} (Searched for prefix: {base_name_for_matching})")
            
    return pairs

def apply_connected_component_labeling(mask_slice):
    """
    Separate touching/overlapping objects into individual labeled regions.
    """
    binary_mask = mask_slice > 0
    # Use 8-connectivity
    labeled_array, _ = ndimage.label(binary_mask, structure=np.ones((3, 3)))
    return labeled_array

def process_files(all_pairs, save_root):
    """
    Main processing loop. Duplicates mask slices for each image variation.
    """
    if not all_pairs:
        print("No matched image-mask pairs found to process.")
        return

    total_slices_saved = 0

    for image_paths, msk_path, base_name, set_name in tqdm(all_pairs, desc="Processing files"):
        try:
            # Load the single mask that corresponds to all image variations
            manual_masks = tifffile.imread(msk_path)
        except Exception as e:
            print(f"❌ Error reading mask {os.path.basename(msk_path)}: {e}")
            continue

        # Loop through each image variation (original, _dog, etc.)
        for img_path in image_paths:
            try:
                # Define output directories dynamically for each file
                img_output_dir = os.path.join(save_root, set_name, "images")
                msk_output_dir = os.path.join(save_root, set_name, "masks")
                os.makedirs(img_output_dir, exist_ok=True)
                os.makedirs(msk_output_dir, exist_ok=True)

                raw_data = tifffile.imread(img_path)
            except Exception as e:
                print(f"❌ Error reading image {os.path.basename(img_path)}: {e}")
                continue

            # B. HANDLE DIMENSIONS
            if raw_data.ndim == 4:
                if raw_data.shape[0] <= 5 and raw_data.shape[0] < raw_data.shape[1]:
                    raw_data = np.transpose(raw_data, (1, 2, 3, 0))
                elif raw_data.shape[1] <= 5:
                    raw_data = np.transpose(raw_data, (0, 2, 3, 1))
            elif raw_data.ndim == 3:
                raw_data = raw_data[..., np.newaxis]

            if raw_data.ndim != 4:
                print(f"⚠️ Skipping {os.path.basename(img_path)}: Unexpected shape {raw_data.shape}")
                continue
                
            # C. NORMALIZE
            low, high = np.percentile(raw_data, (0.00001, 99.99999))
            if high > low:
                raw_data = np.clip((raw_data.astype(np.float32) - low) / (high - low), 0, 1)
            else:
                raw_data = np.zeros_like(raw_data, dtype=np.float32)
                
            # D. RGB CONVERSION
            Z, Y, X, C = raw_data.shape
            raw_data_rgb = np.zeros((Z, Y, X, 3), dtype=np.float32)
            raw_data_rgb[..., :min(C, 3)] = raw_data[..., :min(C, 3)]

            # E. FILTER & SAVE SLICES
            # Get the specific filename of the image variation being processed
            img_variation_name = os.path.basename(img_path).replace(".ome.tif", "").replace(".tif", "")

            for z in range(Z):
                if np.any(manual_masks[z]):
                    # Filename for the slice now matches the specific image variation
                    fname = f"{img_variation_name}_slice_{z:04d}.tif"
                    img_save_path = os.path.join(img_output_dir, fname)
                    msk_save_path = os.path.join(msk_output_dir, fname)
                    
                    tifffile.imwrite(img_save_path, raw_data_rgb[z])
                    
                    # Apply labeling to the original mask slice and save it
                    labeled_slice = apply_connected_component_labeling(manual_masks[z])
                    tifffile.imwrite(msk_save_path, labeled_slice.astype(np.uint16))
                    
                    total_slices_saved += 1

    print("\n" + "-"*50)
    print("🎉 Processing Complete.")
    # Note: total_slices_saved now counts each duplicated mask slice
    print(f"✅ Total clean slices saved: {total_slices_saved}")
    print(f"📂 Data saved in: {save_root}")
    print("-"*50)

def main():
    """
    Main function to run the entire preprocessing pipeline.
    """
    print(f"Output will be saved to: {SAVE_ROOT}")
    
    all_pairs = []
    for set_name in PROCESS_SETS:
        print(f"\n--- Searching in '{set_name}' folder ---")
        set_pairs = find_image_mask_pairs(INPUT_ROOT, set_name)
        all_pairs.extend(set_pairs)
        print(f"Found {len(set_pairs)} matched pairs in '{set_name}'.")

    print("\n" + "="*50)
    print(f"Total found {len(all_pairs)} matched image-mask pairs. Starting processing...")
    print("="*50)
    
    process_files(all_pairs, SAVE_ROOT)

if __name__ == "__main__":
    main()
