import sys
import os
import glob
import numpy as np
import tifffile
from tqdm import tqdm  # Progress bar

# --- 1. CONFIGURATION ---
# Input Directories
IMAGE_DIR = r"/home/students/rawat/czi_output"
LABEL_DIR = r"/home/students/rawat/czi_output/Annotations"

# Output Directory (Where slices will be saved)
SAVE_ROOT = r"/home/students/rawat/training_data"
IMG_OUTPUT_DIR = os.path.join(SAVE_ROOT, "data", "train", "images")
LBL_OUTPUT_DIR = os.path.join(SAVE_ROOT, "data", "train", "labels")

# Ensure output directories exist
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
os.makedirs(LBL_OUTPUT_DIR, exist_ok=True)

print(f"Output will be saved to: {SAVE_ROOT}")

# --- 2. FILE MATCHING HELPER ---
def get_matched_pairs(image_dir, label_dir):
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
    pairs = []
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        
        # Determine Base Name (Handle .ome.tif and .tif)
        if filename.endswith(".ome.tif"):
            base_name = filename.replace(".ome.tif", "")
        else:
            base_name = filename.replace(".tif", "")
            
        # Construct Expected Label Name
        # Logic: Image "Name.tif" -> Label "Name_label.tif"
        label_name = f"{base_name}_label.tif"
        label_path = os.path.join(label_dir, label_name)
        
        if os.path.exists(label_path):
            pairs.append((img_path, label_path, base_name))
        else:
            print(f"⚠️ Warning: Label not found for {filename} (Expected: {label_name})")
            
    return pairs

# --- 3. PROCESSING LOOP ---
pairs = get_matched_pairs(IMAGE_DIR, LABEL_DIR)
print(f"Found {len(pairs)} matched image-label pairs. Starting processing...")

total_slices_saved = 0

for img_path, lbl_path, base_name in tqdm(pairs, desc="Processing files"):
    
    # A. LOAD DATA
    try:
        raw_data = tifffile.imread(img_path)
        manual_labels = tifffile.imread(lbl_path)
    except Exception as e:
        print(f"❌ Error reading {base_name}: {e}")
        continue

    # B. HANDLE DIMENSIONS (Your Snippet)
    # ---------------------------------------------------------
    ## Verify axis order from OME metadata
    with tifffile.TiffFile(img_path) as tif:
        if tif.ome_metadata:
            try:
                raw_axis_order = tif.series[0].axes
                # print(f"ℹ️ {base_name} Metadata Axis Order: {raw_axis_order}")
            except Exception:
                pass

    if raw_data.ndim == 4:
        if raw_data.shape[0] <= 4:  # (C, Z, Y, X)
            raw_data = np.transpose(raw_data, (1, 2, 3, 0))
        elif raw_data.shape[1] <= 4:  # (Z, C, Y, X)
            raw_data = np.transpose(raw_data, (0, 2, 3, 1))
    elif raw_data.ndim == 3:
         # Assume (Z, Y, X) if 3D, add dummy channel dim -> (Z, Y, X, 1)
         raw_data = raw_data[..., np.newaxis]
    else:
        print(f"Skipping {base_name}: Unexpected dimensions {raw_data.shape}")
        continue
    # ---------------------------------------------------------

    # C. RGB CONVERSION (Your Snippet)
    Z, Y, X, C = raw_data.shape
    
    # Create an empty 3-channel array
    raw_data_rgb = np.zeros((Z, Y, X, 3), dtype=raw_data.dtype)
    
    # Copy the first two channels (green, red)
    # Ensure we don't exceed channel count of input
    raw_data_rgb[..., :min(C, 2)] = raw_data[..., :min(C, 2)]

    # D. FILTER & SAVE SLICES
    # ---------------------------------------------------------
    # We loop through Z and check the mask
    saved_count_local = 0
    
    for z in range(Z):
        # CHECK: Is there any annotation in this slice?
        if np.max(manual_labels[z]) > 0:
            
            # Construct unique filename: BrainName_slice_005.tif
            fname = f"{base_name}_slice_{z:04d}.tif"
            
            img_save_path = os.path.join(IMG_OUTPUT_DIR, fname)
            lbl_save_path = os.path.join(LBL_OUTPUT_DIR, fname)
            
            # Save Image (The RGB version)
            tifffile.imwrite(img_save_path, raw_data_rgb[z])
            
            # Save Label
            tifffile.imwrite(lbl_save_path, manual_labels[z])
            
            saved_count_local += 1

    total_slices_saved += saved_count_local

print("------------------------------------------------")
print(f"🎉 Processing Complete.")
print(f"✅ Total clean slices saved: {total_slices_saved}")
print(f"📂 Images: {IMG_OUTPUT_DIR}")
print(f"📂 Labels: {LBL_OUTPUT_DIR}")