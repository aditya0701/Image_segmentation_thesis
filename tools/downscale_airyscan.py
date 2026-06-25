import numpy as np
import tifffile
from skimage.transform import resize
from tqdm import tqdm

# ==========================================
# 1. Setup File Paths & Parameters
# ==========================================
TARGET_SIZE  = 1100


INPUT_IMAGE  = r"D:\Thesis_images\All_images_tif\New_images\Surface_Masks\Surface_Masks\airyscan_processed\Brain7_1340_MTM_AiryscanSR_Fly3_CalyxR.tif"
# INPUT_MASK   = r"D:\Thesis_images\All_images_tif\annotations\Airyscan\test annotations\Brain6_1340_MTM_AiryscanSR_Fly3_CalyxL_sequential.tif"


OUTPUT_IMAGE = r"D:\Thesis_images\All_images_tif\New_images\Surface_Masks\Surface_Masks\airyscan_processed\Brain7_1340_MTM_AiryscanSR_Fly3_CalyxR_1100.tif"
# OUTPUT_MASK  = r"D:\Thesis_images\All_images_tif\Final_annotation\edited\airyscan\B6_CL_downscaled.tif"

print("Loading TIFFs into memory...")
image = tifffile.imread(INPUT_IMAGE)
# mask = tifffile.imread(INPUT_MASK)

# ==========================================
# 2. Verify Shapes & Initialize Output Arrays
# ==========================================
print(f"Original Image Shape: {image.shape}") # Expected: (70, 2, 1834, 1834)
# print(f"Original Mask Shape: {mask.shape}")   # Expected: (70, 1834, 1834)

# Extract dimensions
z_depth, channels, orig_y, orig_x = image.shape
target_shape_xy = (TARGET_SIZE, TARGET_SIZE)

# Pre-allocate memory for the downsized arrays to ensure fast processing
# We keep the exact same data types (likely uint16 or uint8) to avoid memory bloat
downsized_image = np.zeros((z_depth, channels, TARGET_SIZE, TARGET_SIZE), dtype=image.dtype)
# downsized_mask = np.zeros((z_depth, TARGET_SIZE, TARGET_SIZE), dtype=mask.dtype)

# ==========================================
# 3. Process Slice by Slice
# ==========================================
print(f"Resizing X and Y to {TARGET_SIZE}x{TARGET_SIZE}...")

# We use tqdm to give a progress bar since 70 slices might take a moment
for z in tqdm(range(z_depth), desc="Processing Z-stack"):
    
    # # --- Process the Mask ---
    # # order=0 (nearest-neighbor) and anti_aliasing=False are strictly required
    # current_mask_slice = mask[z]
    # downsized_mask[z] = resize(
    #     current_mask_slice, 
    #     target_shape_xy, 
    #     order=0, 
    #     preserve_range=True, 
    #     anti_aliasing=False
    # ).astype(mask.dtype)
    
    # --- Process the Image (Channel by Channel) ---
    # order=1 (bilinear) and anti_aliasing=True keep the microscopy textures smooth
    for c in range(channels):
        current_img_slice = image[z, c]
        downsized_image[z, c] = resize(
            current_img_slice, 
            target_shape_xy, 
            order=1, 
            preserve_range=True, 
            anti_aliasing=True
        ).astype(image.dtype)

# ==========================================
# 4. Save the Output
# ==========================================
print(f"Saving Downsized Image to:\n{OUTPUT_IMAGE}")
# imagej=True writes the metadata in a way that ImageJ/Fiji understands the Z/C axes
tifffile.imwrite(OUTPUT_IMAGE, downsized_image, imagej=True)

# print(f"Saving Downsized Mask to:\n{OUTPUT_MASK}")
# tifffile.imwrite(OUTPUT_MASK, downsized_mask, imagej=True)

print("Processing complete!")