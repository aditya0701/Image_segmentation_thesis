import tifffile
import numpy as np

MASK_PATH = r"D:\Thesis_images\Final_dataset\DOG\train\masks\PN_KC_488_568_Brain4_CalyxL_LSMPlus_16bit_mask.tif"

# Voxel sizes in µm (z, y, x)
VOXEL_AIRYSCAN = (0.3, 0.0425, 0.0425)   # 0.0425 x 0.0425 x 0.3 µm³
VOXEL_CONFOCAL = (0.3, 0.0709, 0.0709)   # 0.0709 x 0.0709 x 0.3 µm³

VOXEL_SIZE_UM = VOXEL_AIRYSCAN  # <-- change to VOXEL_CONFOCAL if needed

mask = tifffile.imread(MASK_PATH)
print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")

labels, counts = np.unique(mask, return_counts=True)

voxel_vol = VOXEL_SIZE_UM[0] * VOXEL_SIZE_UM[1] * VOXEL_SIZE_UM[2]

print(f"\n{'Label':>8}  {'Voxels':>12}  {'Volume (µm³)':>14}")
print("-" * 40)
for label, count in zip(labels, counts):
    if label == 0:
        continue  # skip background
    print(f"{label:>8}  {count:>12}  {count * voxel_vol:>14.1f}")

print(f"\nTotal masks (excluding background): {len(labels) - (1 if 0 in labels else 0)}")
