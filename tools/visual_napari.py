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
import tifffile
import numpy as np
import napari

# --- PATHS ---
IMAGE_PATH = r"/home/students/rawat/czi_output/Brain2_PN_KC_Group3_25minfix_OvernightSecAB_488_561_Calyx1_Confocal.tif"
# Use the path where you saved your auto-generated results
# MASK_PATH = r'/work/scratch/rawat/ouput_masks/prediction_result.tif'
MASK_FIX_PATH = r"/work/scratch/rawat/ouput_masks/Brain2_PN_KC_Group3_trackedS.tif"

print("Loading images...")
img = tifffile.imread(IMAGE_PATH)
# mask = tifffile.imread(MASK_PATH)
mask_fix = tifffile.imread(MASK_FIX_PATH)
print(f"Image shape: {img.shape}")
# print(f"Mask shape: {mask.shape}")

# Check image dimensions and convert if necessary
# Determine the shape format: ZCHW or CZHW
if len(img.shape) == 4:
    # Check if it's ZCHW format (Z is typically largest for z-stack images)
    if img.shape[1] <= 3:  # C (channels) is typically 1-3
        # Image is in ZCHW format, convert to CZHW
        print(f"Detected ZCHW format. Converting to CZHW...")
        img = np.transpose(img, (1, 0, 2, 3))  # (Z, C, H, W) -> (C, Z, H, W)
    else:
        print(f"Already in CZHW format or different format. No conversion needed.")
else:
    print(f"Image has {len(img.shape)} dimensions. Expected 4D image (CZHW or ZCHW).")

print(f"Final image shape: {img.shape}")

# Transpose image from (Z, C, H, W) to (C, Z, H, W)
# This moves the channel dimension from axis 1 to axis 0
# img = np.transpose(img, (1, 0, 2, 3))
print(f"Image shape (transposed): {img.shape}")

# Determine colormap and channel names based on number of channels
num_channels = img.shape[0]
if num_channels == 3:
    colormaps = ['green', 'red', 'magenta']
    channel_names = ['GFP', 'mCherry', 'anti-synapsin']
elif num_channels == 2:
    colormaps = ['green', 'red']
    channel_names = ['GFP', 'mCherry']
else:
    # Default for single channel or other cases
    colormaps = ['viridis'] * num_channels
    channel_names = [f'Channel {i+1}' for i in range(num_channels)]

print(f"Number of channels: {num_channels}")
print(f"Colormaps: {colormaps}")
print(f"Channel names: {channel_names}")

# Initialize Viewer
viewer = napari.Viewer()

# 1. Add the Raw Image (Split into channels)
# channel_axis=0 splits the first dimension into separate layers
viewer.add_image(
    img, 
    channel_axis=0, 
    colormap=colormaps,
    name=channel_names,
    scale=(1, 1, 1) # Ensures pixels are treated as cubes
)

# 2. Add the Annotation Mask (as a Labels layer)
# We add this as a single layer. Napari automatically handles the Z-stack 
# overlay because the shape (92, 1431, 1431) matches the shape of your image channels.
# viewer.add_labels(
#     mask, 
#     name='Bouton Annotations',
#     opacity=0.6,
#     scale=(1, 1, 1)
# )

viewer.add_labels(
    mask_fix,
    name='Corrected Bouton Annotations',
    opacity=0.6,
    scale=(1, 1, 1)
)   
print("Napari launched. Use the slider at the bottom to scroll through Z-slices.")
napari.run()