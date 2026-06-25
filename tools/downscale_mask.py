import tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import os

# --- Parameters ---
INPUT_MASK_PATH = r"D:\Thesis_images\Final_dataset\all\test\B1l.tif"
OUTPUT_DIR = r"D:\Thesis_images\Final_dataset\all\test"
TARGET_SHAPE = (1100, 1100) # (Y, X)
SUFFIX = "_downscale_1100_instance_mask"

try:
    # 1. Read the original instance mask (DO NOT binarize it)
    mask = tifffile.imread(INPUT_MASK_PATH)
    print(f"Original shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"Original unique instances: {len(np.unique(mask)) - 1} (excluding background)")

    # 2. Downscale using Nearest-Neighbor Interpolation
    if mask.ndim == 3:
        new_shape = (mask.shape[0], TARGET_SHAPE[0], TARGET_SHAPE[1])
        
        # KEY FIX: order=0 (nearest-neighbor) and anti_aliasing=False
        downscaled_mask = resize(
            mask, 
            new_shape, 
            order=0,                # Nearest-neighbor interpolation
            anti_aliasing=False,    # Disable blending
            preserve_range=True     # Do not normalize values between 0 and 1
        ).astype(mask.dtype)
        
        print(f"Downscaled shape: {downscaled_mask.shape}")
        print(f"Downscaled unique instances preserved: {len(np.unique(downscaled_mask)) - 1}")

        # 3. Save the new Instance Mask
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base_name = os.path.basename(INPUT_MASK_PATH)
        name, ext = os.path.splitext(base_name)
        new_filename = f"{name}{SUFFIX}{ext}"
        output_path = os.path.join(OUTPUT_DIR, new_filename)
        
        # --- NEW FIX: Convert uint32 to uint16 for ImageJ compatibility ---
        if downscaled_mask.dtype == np.uint32:
            downscaled_mask = downscaled_mask.astype(np.uint16)
        
        tifffile.imwrite(output_path, downscaled_mask, imagej=True)
        print(f"Instance mask saved successfully to: {output_path}")

        # --- Display the Output ---
        middle_slice_index = downscaled_mask.shape[0] // 2
        display_slice = downscaled_mask[middle_slice_index]
        
        # Using a highly distinct colormap 'nipy_spectral' to visualize different instances
        plt.figure(figsize=(8, 8))
        # Mask out 0s so background remains white/black, and instances get colored
        masked_display = np.ma.masked_where(display_slice == 0, display_slice)
        plt.imshow(masked_display, cmap='nipy_spectral', interpolation='nearest')
        plt.title(f"Downscaled Instance Mask (Slice {middle_slice_index})")
        plt.axis('off')
        plt.show()

    else:
        print("Mask is not 3D. Please check the input file.")

except Exception as e:
    print(f"An error occurred: {e}")