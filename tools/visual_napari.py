import tifffile
import numpy as np
import napari

# --- PATHS ---
IMAGE_PATH = r"/home/students/rawat/training_data_new/data/train/images/PN_KC_488_568_Brain1_CalyxL_LSMPlus_16bit_slice_0016.tif"
# Use the path where you saved your auto-generated results
# MASK_PATH = r"D:\Thesis_images\All_images_tif\Final_annotation\PN_KC_488_568_Brain1_CalyxR_LSMPlus_16bit_label.tif"
MASK_FIX_PATH = r"/home/students/rawat/training_data_new/data/train/labels/PN_KC_488_568_Brain1_CalyxL_LSMPlus_16bit_slice_0016.tif"
# MASK_IMARIS_PATH = r"/home/students/rawat/Dataset/Surface_Masks/Surface_Masks/Brain1_1340_Cont_Fly1_CalyxL_Mask.tif"


print("Loading images...")
img = tifffile.imread(IMAGE_PATH)
# mask = tifffile.imread(MASK_PATH)
mask_fix = tifffile.imread(MASK_FIX_PATH)
# mask_imaris = tifffile.imread(MASK_IMARIS_PATH)
print(f"Image shape (original): {img.shape}")
# print(f"Mask shape: {mask.shape}")
print(f"Mask fix shape: {mask_fix.shape}")
# print(f"Mask imaris shape: {mask_imaris.shape}")

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

# --- Dimension Handling ---
# If the image is 3D and the last dimension is small (<=4), it's likely (Y, X, C).
# We add a Z-axis to make it (Z, Y, X, C) for napari.
if raw_data.ndim == 3 and raw_data.shape[-1] <= 4:
    print("Detected format (Y, X, C). Adding a Z dimension.")
    raw_data = raw_data[np.newaxis, ...]

# If the image is 3D and not (Y, X, C), it must be a (Z, Y, X) volume.
# We add a channel axis to make it (Z, Y, X, C).
elif raw_data.ndim == 3:
    print("Detected format (Z, Y, X). Adding a channel dimension.")
    raw_data = raw_data[..., np.newaxis]

# If the image is 4D, we assume it's one of the standard C/Z orders and transpose if needed.
elif raw_data.ndim == 4:
    if raw_data.shape[0] <= 4:  # (C, Z, Y, X)
        print("Detected format (C, Z, Y, X). Transposing to (Z, Y, X, C)...")
        raw_data = np.transpose(raw_data, (1, 2, 3, 0))
    elif raw_data.shape[1] <= 4:  # (Z, C, Y, X)
        print("Detected format (Z, C, Y, X). Transposing to (Z, Y, X, C)...")
        raw_data = np.transpose(raw_data, (0, 2, 3, 1))
    # Otherwise, we assume it's already (Z, Y, X, C)

print(f"Processed shape for napari (Z, Y, X, C): {raw_data.shape}")

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
colormap_image, name_image = generate_colormaps_and_names(img_float, is_label=False)

# only use first channel for mask_imaris
# Handle both 3D and 4D cases
# if mask_imaris.ndim == 4:
#     if mask_imaris.shape[0] > 3:
#         # Format is (Z, C, H, W), select channel 1
#         mask_channel = mask_imaris[:, 1, :, :]
#     else:
#         # Format is (C, Z, H, W), select channel 1
#         mask_channel = mask_imaris[1, :, :, :]
# else:
#     # If 3D (Z, H, W), use the entire array
#     mask_channel = mask_imaris
# # We say: "If the pixel value is greater than 0, make it Label 1. Otherwise Label 0."
# mask_integer = (mask_channel).astype('uint16')
# --- 3. Initialize Viewer ---
viewer = napari.Viewer()

# channel_axis=-1 is more robust as it always refers to the last axis
viewer.add_image(
    img_float,
    channel_axis=-1,
    colormap=colormap_image,
    name=name_image,
)
# viewer.add_labels(
#     mask_integer,
#     name='Imaris Bouton Annotations',
#     opacity=0.6,
# )
# # ## add labels for mask
# viewer.add_labels(
#     mask,
#     name='Predicted (before fix)',
#     opacity=0.6,
# )
viewer.add_labels(
    mask_fix,
    name='Edited Bouton Annotations',
    opacity=0.6,
)   
print("Napari launched. Use the slider at the bottom to scroll through Z-slices.")
napari.run()