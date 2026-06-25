import tifffile
import numpy as np
import napari

# =============================================================================
# CONFIGURATION — edit these paths and settings before running
# =============================================================================

IMAGE_PATH = (r"D:\Thesis_images\Final_dataset\all\validation\Brain2_1340_Cont_AiryscanSR_Fly1_CalyxR_1100.tif"#"D:\Thesis_images\Final_dataset\all\test\images\PN_KC_488_568_Brain2_CalyxR_LSMPlus_16bit.ome.tif"
    # r"D:\Thesis_images\All_images_tif\New_images\Surface_Masks"
    # r"\Surface_Masks\airyscan_processed"
    # r"\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100.tif"
)

MASK_GT_PATH = (r"D:\Thesis_images\Final_dataset\all\train\masks\Brain6_1340_MTM_AiryscanSR_Fly3_CalyxL_1100_mask.tif")#"D:\Thesis_images\Final_dataset\all\test\masks\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100_mask.tif"

MASK_PRED_PATH = (r"D:\Thesis_images\Updated_final_results\MicroSAM_Base\vit_b_lm_neo_augmented_Original_v1\PN_KC_488_568_Brain2_CalyxR_LSMPlus_16bit_masks.tif")#

# Optional: uncomment and set to also load a manually annotated mask
MASK_MANUAL_PATH = (r"D:\Thesis_images\Updated_final_results\CellposeSAM_xyz_2d\cellpose3d_calyx_dog_xyz_final_2d\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100_mask.tif")

# Physical voxel size in micrometres (Z, Y, X) — from your acquisition metadata.
# Airyscan Brain1: XY = 0.0709 µm, Z = 0.300 µm.
# Change these to match a different image if needed.
VOXEL_ZYX_UM = (0.300, 0.0709, 0.0709)

# Per-channel display settings — extend if your image has more than 2 channels
CHANNEL_COLORMAPS = ["green", "red", "blue"]
CHANNEL_NAMES     = ["GFP",   "mCherry", "Blue"]

# Percentile clipping for contrast normalisation (per channel).
# 0.5 / 99.5 works well for most fluorescence volumes.
# Lower the low-end or raise the high-end if the image looks over/under exposed.
PERCENTILE_LOW  = 0.00001
PERCENTILE_HIGH = 99.9999

# =============================================================================
# LOAD
# =============================================================================

print("Loading images...")
img        = tifffile.imread(IMAGE_PATH)
mask_gt    = tifffile.imread(MASK_GT_PATH)
mask_pred  = tifffile.imread(MASK_PRED_PATH)
mask_manual = tifffile.imread(MASK_MANUAL_PATH)  # uncomment if needed

print(f"  Raw image shape : {img.shape}  dtype={img.dtype}")
print(f"  GT mask shape   : {mask_gt.shape}  dtype={mask_gt.dtype}")
print(f"  Pred mask shape : {mask_pred.shape}  dtype={mask_pred.dtype}")
print(f"  Manual mask shape : {mask_manual.shape}  dtype={mask_manual.dtype}")

# =============================================================================
# PREPROCESS IMAGE → (Z, Y, X, C) float32, one normalisation pass per channel
# =============================================================================

raw = img.copy()

# Reorder to (Z, Y, X, C) regardless of how the file was saved
if raw.ndim == 4:
    if raw.shape[0] <= 4:          # (C, Z, Y, X)
        raw = np.transpose(raw, (1, 2, 3, 0))
    elif raw.shape[1] <= 4:        # (Z, C, Y, X)
        raw = np.transpose(raw, (0, 2, 3, 1))
elif raw.ndim == 3:                # (Z, Y, X) — single channel
    raw = raw[..., np.newaxis]

del img   # free original allocation before converting to float32

n_channels = raw.shape[-1]
img_float  = np.empty(raw.shape, dtype=np.float32)

for ch in range(n_channels):
    plane     = raw[..., ch].astype(np.float32)
    low       = np.float32(np.percentile(plane, PERCENTILE_LOW))
    high      = np.float32(np.percentile(plane, PERCENTILE_HIGH))
    if high > low:
        plane -= low
        plane /= (high - low)
        np.clip(plane, 0, 1, out=plane)
    img_float[..., ch] = plane

del raw

print(f"  Normalised shape (Z, Y, X, C): {img_float.shape}")

# =============================================================================
# VIEWER
# =============================================================================

colormaps = CHANNEL_COLORMAPS[:n_channels]
names     = CHANNEL_NAMES[:n_channels]

viewer = napari.Viewer(title="Bouton segmentation viewer")

viewer.add_image(
    img_float,
    channel_axis=3,
    colormap=colormaps,
    name=names,
    visible=True,
    scale=VOXEL_ZYX_UM,
    blending="additive",
)

viewer.add_labels(
    mask_gt,
    name="Ground-truth mask",
    opacity=0.5,
    visible=True,
    scale=VOXEL_ZYX_UM,
)

viewer.add_labels(
    mask_pred,
    name="Predicted mask microSAM L",
    opacity=0.5,
    visible=True,
    scale=VOXEL_ZYX_UM,
)

# viewer.add_labels(
#     mask_manual,
#     name="Manual mask",
#     opacity=0.5,
#     visible=True,
#     scale=VOXEL_ZYX_UM,
# )

print("Napari ready.  Close the window to exit.")
napari.run()