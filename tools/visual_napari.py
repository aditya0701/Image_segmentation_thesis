import tifffile
import numpy as np
import napari

# --- PATHS ---
IMAGE_PATH = r"D:\Thesis_images\All_images_tif\Brain2_PN_KC_Group3_25minfix_OvernightSecAB_488_561_Calyx1_Confocal.tif"
# Use the path where you saved your auto-generated results
# MASK_PATH = r'/work/scratch/rawat/ouput_masks/prediction_result.tif'
MASK_FIX_PATH = r"D:\Thesis_images\All_images_tif\annotations\Brain2_PN_KC_Group3_trackedS.tif"

print("Loading images...")
img = tifffile.imread(IMAGE_PATH)
# mask = tifffile.imread(MASK_PATH)
mask_fix = tifffile.imread(MASK_FIX_PATH)
print(f"Image shape (original): {img.shape}")
print(f"Mask shape: {mask_fix.shape}")

if img.ndim == 4 and img.shape[0] > img.shape[1]:
    print("Detected format (Z, C, H, W). Transposing to (C, Z, H, W)...")
    # Swap axis 0 and 1 to move Channel to the front
    img = np.transpose(img, (1, 0, 2, 3))
else:
    print("Detected format (C, Z, H, W). Keeping original orientation.")

print(f"Image shape (final): {img.shape}")

# --- 2. Channel & Color Logic ---
# Check the size of the first dimension (Channel axis)
num_channels = img.shape[0]

if num_channels == 2:
    current_colormaps = ['green', 'red']
    current_names = ['GFP', 'mCherry']
else:
    # Logic for 3 (or more) channels: Add Blue
    # We extend the lists to accommodate the extra channel
    current_colormaps = ['green', 'red', 'blue']
    current_names = ['GFP', 'mCherry', 'Blue']

# --- 3. Initialize Viewer ---
viewer = napari.Viewer()

# Add the Image
# channel_axis=0 splits the C dimension into separate layers
viewer.add_image(
    img, 
    channel_axis=0, 
    colormap=current_colormaps,
    name=current_names,
    scale=(1, 1, 1)  # Scale for (Z, H, W)
)

viewer.add_labels(
    mask_fix,
    name='Corrected Bouton Annotations',
    opacity=0.6,
    scale=(1, 1, 1)
)   
print("Napari launched. Use the slider at the bottom to scroll through Z-slices.")
napari.run()