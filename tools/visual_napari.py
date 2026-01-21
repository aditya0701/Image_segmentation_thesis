import tifffile
import numpy as np
import napari

# --- PATHS ---
IMAGE_PATH = r"D:\Thesis_images\All_images_tif\New_images\PN_KC_488_568_Brain1_CalyxL_LSMPlus_16bit.ome.tif"
# Use the path where you saved your auto-generated results
MASK_PATH = r"D:\Thesis_images\All_images_tif\annotations\proceesed_images_label\PN_KC_488_568_Brain1_CalyxL_LSMPlus_16bit_label_vit_l_lm_patch1024\PN_KC_488_568_Brain1_CalyxL_LSMPlus_16bit_label_vit_l_lm_patch1024_synced.tif"
MASK_FIX_PATH = r"D:\Thesis_images\All_images_tif\annotations\proceesed_images_label\PN_KC_488_568_Brain1_CalyxL_LSMPlus_16bit_label_vit_l_lm_patch1024\PN_KC_488_568_Brain1_CalyxL_LSMPlus_16bit_label_vit_l_lm_patch1024_synced_cleaned_auto.tif"
MASK_IMARIS_PATH = r"D:\Thesis_images\All_images_tif\annotations\proceesed_images_label\PN_KC_488_568_Brain1_CalyxL_LSMPlus_16bit.ims - PN_KC_488_568_Brain1_CalyxL_LSMPlus_16bit_Surface_Mask_Boutons.tif"


print("Loading images...")
img = tifffile.imread(IMAGE_PATH)
mask = tifffile.imread(MASK_PATH)
mask_fix = tifffile.imread(MASK_FIX_PATH)
mask_imaris = tifffile.imread(MASK_IMARIS_PATH)
print(f"Image shape (original): {img.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Mask fix shape: {mask_fix.shape}")
print(f"Mask imaris shape: {mask_imaris.shape}")

def generate_colormaps_and_names(img, is_label = False):
    num_channels = img.shape[0]
    if is_label:
        # Logic for label images
        if num_channels == 2:
            current_colormaps = ['green', 'red']
            current_names = ['GFP Labels', 'mCherry Labels']
        else:
            current_colormaps = ['green', 'red', 'blue']
            current_names = ['GFP Labels', 'mCherry Labels', 'Blue Labels']
        return current_colormaps, current_names
    else:
        # Logic for regular images
        if num_channels == 2:
            current_colormaps = ['green', 'red']
            current_names = ['GFP', 'mCherry']
        else:
            current_colormaps = ['green', 'red', 'blue']
            current_names = ['GFP', 'mCherry', 'Blue']
        return current_colormaps, current_names

def fix_dimensions(image):
    if image.ndim == 4 and image.shape[0] > image.shape[1]:
        print("Detected format (Z, C, H, W). Transposing to (C, Z, H, W)...")
        # Swap axis 0 and 1 to move Channel to the front
        image = np.transpose(image, (1, 0, 2, 3))
    else:
        print("Detected format (C, Z, H, W). Keeping original orientation.")

    print(f"Image shape (final): {image.shape}")
    return image

# --- 1. FIX DIMENSIONS IF NEEDED ---
img = fix_dimensions(img)
mask_imaris = fix_dimensions(mask_imaris)
low, high = np.percentile(img, (0.0001, 99.9999))

if high > low:
    # Normalize to 0-1 as float32
    img_float = (img.astype(np.float32) - low) / (high - low)
    img_float = np.clip(img_float, 0, 1)
else:
    # Flat image fallback
    img_float = np.zeros_like(img, dtype=np.float32)
# --- 2. GENERATE COLORMAPS AND NAMES ---
# colormap_label, name_label = generate_colormaps_and_names(mask, is_label=True)
colormap_image, name_image = generate_colormaps_and_names(img_float, is_label=False)

## only use first channel for mask_imaris 
mask_channel = mask_imaris[1, :, :, :]
# We say: "If the pixel value is greater than 0, make it Label 1. Otherwise Label 0."
mask_integer = (mask_channel).astype('uint16')
# --- 3. Initialize Viewer ---
viewer = napari.Viewer()      

# channel_axis=0 splits the C dimension into separate layers
viewer.add_image(
    img_float, 
    channel_axis=0, 
    colormap=colormap_image,
    name=name_image,
    scale=(1, 1, 1)  # Scale for (Z, H, W)
)
viewer.add_labels(
    mask_integer,
    name='Imaris Bouton Annotations',
    opacity=0.6,
    scale=(1, 1, 1)
)
## add labels for mask 
viewer.add_labels(
    mask,
    name='Predicted Bouton Annotations with blob',
    opacity=0.6,
    scale=(1, 1, 1)
)
viewer.add_labels(
    mask_fix,
    name='Predicted Bouton Annotations after blob',
    opacity=0.6,
    scale=(1, 1, 1)
)   
print("Napari launched. Use the slider at the bottom to scroll through Z-slices.")
napari.run()