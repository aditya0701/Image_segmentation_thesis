import tifffile
import numpy as np
import napari
from scipy.ndimage import zoom

# --- PATHS ---
ORIGINAL_IMAGE_PATH  = r"D:\Thesis_images\All_images_tif\New_images\Surface_Masks\Surface_Masks\airyscan_processed\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL.tif"
MASK_GT_PATH         = r"D:\Thesis_images\Final_dataset\all\test\masks\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100_masks_PS.tif"
MASK_PRED_PATH       = r"D:\Thesis_images\Updated_final_results\MicroSAM_Large\vit_l_lm_neo_augmented_All_v3\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100_masks.tif"

# --- LOAD ---
print("Loading original image...")
img = tifffile.imread(ORIGINAL_IMAGE_PATH)
print(f"Original image shape: {img.shape}")

print("Loading masks...")
mask_gt   = tifffile.imread(MASK_GT_PATH)
mask_pred = tifffile.imread(MASK_PRED_PATH)
print(f"Ground truth mask shape: {mask_gt.shape}")
print(f"Predicted mask shape:    {mask_pred.shape}")

# --- GET TARGET SPATIAL SHAPE FROM ORIGINAL IMAGE ---
# Normalise image to (Z, Y, X[, C]) first just for shape extraction
raw = img
if raw.ndim == 4:
    if raw.shape[0] <= 4:
        raw = np.transpose(raw, (1, 2, 3, 0))   # (C,Z,Y,X) → (Z,Y,X,C)
    elif raw.shape[1] <= 4:
        raw = np.transpose(raw, (0, 2, 3, 1))   # (Z,C,Y,X) → (Z,Y,X,C)
elif raw.ndim == 3:
    raw = raw[..., np.newaxis]                   # (Z,Y,X) → (Z,Y,X,1)

target_z, target_y, target_x = raw.shape[:3]
print(f"Target spatial shape (Z, Y, X): {target_z}, {target_y}, {target_x}")


def upscale_mask(mask, target_shape):
    """Upscale a label mask to target_shape using nearest-neighbour interpolation."""
    tz, ty, tx = target_shape
    mz, my, mx = mask.shape[:3]
    zoom_factors = (tz / mz, ty / my, tx / mx)
    print(f"  zoom factors: {zoom_factors[0]:.3f}, {zoom_factors[1]:.3f}, {zoom_factors[2]:.3f}")
    return zoom(mask.astype(np.int32), zoom_factors, order=0).astype(mask.dtype)


print("Upscaling ground truth mask...")
mask_gt_up   = upscale_mask(mask_gt,   (target_z, target_y, target_x))
print("Upscaling predicted mask...")
mask_pred_up = upscale_mask(mask_pred, (target_z, target_y, target_x))
print(f"Upscaled GT shape:   {mask_gt_up.shape}")
print(f"Upscaled Pred shape: {mask_pred_up.shape}")

# --- NORMALISE IMAGE FOR DISPLAY ---
low, high = np.percentile(raw, (0.00001, 99.9999))
if high > low:
    img_float = np.clip((raw.astype(np.float32) - low) / (high - low), 0, 1)
else:
    img_float = np.zeros_like(raw, dtype=np.float32)

print(f"Processed image shape (Z, Y, X, C): {img_float.shape}")

n_channels = img_float.shape[-1]
colormaps  = ['green', 'red', 'blue'][:n_channels]
names      = ['GFP', 'mCherry', 'Blue'][:n_channels]

# --- VIEWER ---
viewer = napari.Viewer()

viewer.add_image(
    img_float,
    channel_axis=3,
    colormap=colormaps,
    name=names,
    visible=True,
    scale=(1, 1, 1),
)

viewer.add_labels(
    mask_gt_up,
    name='Ground Truth masks',
    opacity=0.6,
    visible=True,
    scale=(1, 1, 1),
)

viewer.add_labels(
    mask_pred_up,
    name='Predicted masks',
    opacity=0.6,
    visible=True,
    scale=(1, 1, 1),
)

print("Napari launched.")
napari.run()
