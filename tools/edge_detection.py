import tifffile
import numpy as np
from skimage.feature import canny
from skimage.morphology import binary_closing, disk
import os

# --- Paths ---
IMAGE_PATH  = r"D:\Thesis_images\All_images_tif\New_images\Surface_Masks\Surface_Masks\airyscan_processed\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100.tif"
MASK_PATH   = r"D:\Thesis_images\Final_Results\Results\vit_l_lm_neo_augmented_PSF_v2\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100_masks.tif"
OUTPUT_PATH = r"D:\Thesis_images\Final_Results\Results\vit_l_lm_neo_augmented_PSF_v2\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100_inner_edges.tif"

def extract_channel(image: np.ndarray, channel: int) -> np.ndarray:
    """Auto-detect axis layout, return a single channel as (Z, Y, X)."""
    if image.ndim == 4:
        if image.shape[0] <= 4:
            data = np.transpose(image, (1, 2, 3, 0))
        elif image.shape[1] <= 4:
            data = np.transpose(image, (0, 2, 3, 1))
        else:
            data = image
    elif image.ndim == 3:
        data = image[..., np.newaxis]
    else:
        raise ValueError(f"Expected 3D or 4D, got shape {image.shape}.")
    
    if channel >= data.shape[-1]:
        raise ValueError(f"Image has {data.shape[-1]} channel(s), CHANNEL={channel} requested.")
        
    return data[..., channel]

try:
    # 1. Load BOTH the raw image and the mask
    raw_image = tifffile.imread(IMAGE_PATH)
    mask_raw = tifffile.imread(MASK_PATH)
    
    # 2. Standardize both to (Z, Y, X) using channel 0
    image_volume = extract_channel(raw_image, channel=0)
    mask_volume = extract_channel(mask_raw, channel=0)
    
    if image_volume.shape != mask_volume.shape:
        raise ValueError("Image and Mask shapes do not match!")

    # 3. Initialize the final output array using uint8 (perfect for 0/255 binary images)
    final_edges = np.zeros(mask_volume.shape, dtype=np.uint8)
    
    print("Processing inner gradients and applying morphological closing slice-by-slice...")
    
    # Define the structuring element for the closing operation. 
    # A disk of radius 2 is a good starting point for bridging small gaps.
    # Increase this number if your contours are still open.
    structuring_element = disk(2)
    
    # 4. Iterate through each Z-slice
    for z in range(mask_volume.shape[0]):
        slice_image = image_volume[z]
        slice_mask = mask_volume[z]
        
        # Isolate pixels that belong to ANY instance label
        valid_regions = slice_mask > 0
        
        if not np.any(valid_regions):
            continue
            
        # Extract the fluorescent gradients strictly inside your labels
        isolated_pixels = np.where(valid_regions, slice_image, 0)
        
        # 5. Apply Canny edge detection
        # (sigma controls sensitivity to small blobs)
        inner_edges = canny(isolated_pixels, sigma=1.0)
        
        # 6. Apply Morphological Closing to bridge fragmented edges
        # This thickens the white pixels to connect them, then thins them back down
        # closed_edges = binary_closing(inner_edges, footprint=structuring_element)
        
        # 7. Add the closed, continuous edges to our final stack
        # final_edges[z][closed_edges] = 255
            
    # 8. Save the result as an ImageJ-compatible stack
    tifffile.imwrite(OUTPUT_PATH, inner_edges, imagej=True)
    print(f"Success! Closed boundary image saved to: {OUTPUT_PATH}")

except Exception as e:
    print(f"An error occurred: {e}")