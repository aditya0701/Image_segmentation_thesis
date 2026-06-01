"""
Downsize test images and their paired masks to TARGET_SIZE x TARGET_SIZE.
Only processes files where XY size is not already 1100 or 1101.
Saves overwriting the original files.

Images : bilinear (order=1) + anti-aliasing  — preserves texture
Masks  : nearest-neighbour (order=0)          — preserves integer instance IDs
"""

import numpy as np
import tifffile
from pathlib import Path
from skimage.transform import resize
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TEST_ROOT   = Path("/home/students/rawat/Final_dataset/Original/test")
IMG_FOLDER  = TEST_ROOT / "images"
MASK_FOLDER = TEST_ROOT / "masks"
TARGET_SIZE = 1100
SKIP_SIZES  = {1100, 1101}   # already correct — do not touch


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def find_mask(img_stem: str):
    """Find the matching mask — excludes already-downsized output files."""
    candidates = [
        p for p in MASK_FOLDER.glob(f"{img_stem}*.tif")
        if not p.stem.endswith(f"_{TARGET_SIZE}")
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print(f"  [WARN] Multiple mask candidates: {[c.name for c in candidates]} — using first")
        return candidates[0]
    return None


def resize_xy(arr: np.ndarray, target: int, order: int) -> np.ndarray:
    """Resize the last two (Y, X) dimensions of any ND array to (target, target).
    Works for both (Z, C, Y, X) images and (Z, Y, X) masks."""
    orig_shape = arr.shape
    leading    = int(np.prod(orig_shape[:-2]))
    flat_in    = arr.reshape(leading, *orig_shape[-2:])
    flat_out   = np.zeros((leading, target, target), dtype=arr.dtype)

    aa = (order > 0)
    for i in tqdm(range(leading), desc="  slices", leave=False):
        flat_out[i] = resize(
            flat_in[i],
            (target, target),
            order=order,
            preserve_range=True,
            anti_aliasing=aa,
        ).astype(arr.dtype)

    return flat_out.reshape(orig_shape[:-2] + (target, target))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
for img_path in sorted(IMG_FOLDER.glob("*.tif")):
    img = tifffile.imread(str(img_path))
    y, x = img.shape[-2], img.shape[-1]

    if y in SKIP_SIZES and x in SKIP_SIZES:
        print(f"SKIP    {img_path.name}  {img.shape}")
        continue

    print(f"\nRESIZE  {img_path.name}  {img.shape} → (..., {TARGET_SIZE}, {TARGET_SIZE})")
    img_out  = resize_xy(img, TARGET_SIZE, order=1)
    img_save = img_path.parent / f"{img_path.stem}_{TARGET_SIZE}.tif"
    tifffile.imwrite(str(img_save), img_out, imagej=True)
    print(f"  Saved image → {img_save.name}")

    mask_path = find_mask(img_path.stem)
    if mask_path is None:
        print(f"  [WARN] No mask found for '{img_path.stem}' — skipping mask")
        continue

    mask      = tifffile.imread(str(mask_path))
    print(f"  Mask   {mask_path.name}  {mask.shape} → (..., {TARGET_SIZE}, {TARGET_SIZE})")
    mask_out  = resize_xy(mask, TARGET_SIZE, order=0)
    mask_save = mask_path.parent / f"{mask_path.stem}_{TARGET_SIZE}.tif"
    tifffile.imwrite(str(mask_save), mask_out)
    print(f"  Saved mask  → {mask_save.name}")

print("\nDone.")
