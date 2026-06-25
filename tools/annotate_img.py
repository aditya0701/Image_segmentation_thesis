import tifffile
import numpy as np
import napari

# --- PATHS ---
MASK_IMARIS_PATH = r"D:\Thesis_images\All_images_tif\New_images\Processed_images_tif\PN_KC_488_568_Brain1_CalyxR_LSMPlus_16bit-1.tif"
IMAGE_PATH = r"D:\Thesis_images\All_images_tif\New_images\Final Dataset\PN_KC_488_568_Brain1_CalyxR_LSMPlus_16bit.ome.tif"
# Use the path where you saved your auto-generated results
MASK_PATH = r"D:\Thesis_images\All_images_tif\Final_annotation\PN_KC_488_568_Brain1_CalyxR_LSMPlus_16bit_label_no_finetuning.tif"
MASK_FIX_PATH = r"D:\Thesis_images\All_images_tif\Final_annotation\PN_KC_488_568_Brain1_CalyxR_LSMPlus_16bit_label_edited.tif"


print("Loading images...")
img = tifffile.imread(IMAGE_PATH)
mask = tifffile.imread(MASK_PATH)
mask_fix = tifffile.imread(MASK_FIX_PATH)
mask_imaris = tifffile.imread(MASK_IMARIS_PATH)
print(f"Image shape (original): {img.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Mask fix shape: {mask_fix.shape}")
print(f"Mask imaris shape: {mask_imaris.shape}")

def print_image_stats(image, name="Image"):
    print(f"{name} stats:")
    print(f"  - Shape: {image.shape}")
    print(f"  - Dtype: {image.dtype}")
    print(f"  - Min: {np.min(image)}")
    print(f"  - Max: {np.max(image)}")
    print(f"  - Mean: {np.mean(image)}")
    print(f"  - Std: {np.std(image)}")

print_image_stats(img, "Original Image")

def generate_colormaps_and_names(img, is_label = False):
    # For (Z, Y, X, C) format, channels are in the last dimension
    num_channels = img.shape[3] if img.ndim == 4 else 1
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

# --- 1. PREPROCESS AND NORMALIZE IMAGE ---
print("Processing image...")
raw_data = img
print(f"Original shape: {raw_data.shape}")

# Extract and reorder dimensions to (Z, Y, X, C)
if raw_data.ndim == 4:
    if raw_data.shape[0] <= 4:  # (C, Z, Y, X)
        print("Detected format (C, Z, Y, X). Transposing to (Z, Y, X, C)...")
        raw_data = np.transpose(raw_data, (1, 2, 3, 0))
    elif raw_data.shape[1] <= 4:  # (Z, C, Y, X)
        print("Detected format (Z, C, Y, X). Transposing to (Z, Y, X, C)...")
        raw_data = np.transpose(raw_data, (0, 2, 3, 1))
    else:
        print("Detected format (Z, Y, X, C). Keeping original orientation.")
elif raw_data.ndim == 3:
    print("Detected format (Z, Y, X). Adding channel dimension...")
    # Assume (Z, Y, X) if 3D, add dummy channel dim -> (Z, Y, X, 1)
    raw_data = raw_data[..., np.newaxis]

print(f"Processed shape (Z, Y, X, C): {raw_data.shape}")

print(f"The data type is {raw_data.dtype}, normalizing to float32.")
# Calculate percentiles on the entire volume
low, high = np.percentile(raw_data, (0.00001, 99.9999))
print(f"Percentile values: low={low}, high={high}")

if high > low:
    # Normalize to 0-1 as float32
    img_float = (raw_data.astype(np.float32) - low) / (high - low)
    img_float = np.clip(img_float, 0, 1)
else:
    # Flat image fallback
    img_float = np.zeros_like(raw_data, dtype=np.float32)

print(f"Image preprocessing complete. Shape: {img_float.shape}")

# --- 2. GENERATE COLORMAPS AND NAMES ---
# colormap_label, name_label = generate_colormaps_and_names(mask, is_label=True)
colormap_image, name_image = generate_colormaps_and_names(img_float, is_label=False)

# --- PREPROCESS MASK_IMARIS (same as image) ---
print("Processing mask_imaris...")
mask_data = mask_imaris
print(f"Original mask_imaris shape: {mask_data.shape}")

# Extract and reorder dimensions to (Z, Y, X, C) - same as image
if mask_data.ndim == 4:
    if mask_data.shape[0] <= 4:  # (C, Z, Y, X)
        print("Detected format (C, Z, Y, X). Transposing to (Z, Y, X, C)...")
        mask_data = np.transpose(mask_data, (1, 2, 3, 0))
    elif mask_data.shape[1] <= 4:  # (Z, C, Y, X)
        print("Detected format (Z, C, Y, X). Transposing to (Z, Y, X, C)...")
        mask_data = np.transpose(mask_data, (0, 2, 3, 1))
    else:
        print("Detected format (Z, Y, X, C). Keeping original orientation.")
elif mask_data.ndim == 3:
    print("Detected format (Z, Y, X). Adding channel dimension...")
    mask_data = mask_data[..., np.newaxis]

print(f"Processed mask_imaris shape (Z, Y, X, C): {mask_data.shape}")

# Extract the final channel as the mask
mask_channel = mask_data[:, :, :, -1]  # Get last channel
mask_integer = mask_channel.astype('uint16')
# --- 3. Initialize Viewer ---
viewer = napari.Viewer()      

# channel_axis=3 splits the C dimension (last axis) into separate layers for (Z, Y, X, C) format
viewer.add_image(
    img_float, 
    channel_axis=3, 
    colormap=colormap_image,
    name=name_image,
    scale=(1, 1, 1)  # Scale for (Z, Y, X)
)
viewer.add_labels(
    mask_integer,
    name='Imaris Bouton Annotations',
    opacity=0.6,
    scale=(1, 1, 1)
)
# # ## add labels for mask 
viewer.add_labels(
    mask,
    name='Predicted (before fix)',
    opacity=0.6,
    scale=(1, 1, 1)
)
viewer.add_labels(
    mask_fix,
    name='Edited Bouton Annotations',
    opacity=0.6,
    scale=(1, 1, 1)
)   
print("Napari launched. Use the slider at the bottom to scroll through Z-slices.")
napari.run()