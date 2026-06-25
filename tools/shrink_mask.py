"""
shrink_mask.py
--------------
Trims each label in a 3D instance segmentation mask to cover only its
high-intensity region, removing dim border voxels.

For every label:
  1. Crop to its bounding box (fast, low memory).
  2. Compute an intensity threshold = INTENSITY_FRACTION × (top INTENSITY_PERCENTILE
     of voxels inside that label).  Voxels below this are discarded.
  3. Fill holes slice-by-slice so the result stays solid.
  4. Remove 3D fragments smaller than MIN_FRAGMENT_VOXELS.
  5. Write the original label value back — no splitting, no new IDs.

=============================================================================
  CONFIGURATION
=============================================================================
"""

IMAGE_PATH  = r"D:\Thesis_images\All_images_tif\New_images\Surface_Masks\Surface_Masks\airyscan_processed\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100.tif"
MASK_PATH   = r"D:\Thesis_images\Final_Results\Results\vit_l_lm_neo_augmented_All_v2\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100_masks.tif"
OUTPUT_PATH = r"D:\Thesis_images\Final_Results\Results\vit_l_lm_neo_augmented_All_v2\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100_masks_shrunk.tif"

CHANNEL = 0  # intensity channel to use (0 = GFP)

# Threshold: keep voxels >= INTENSITY_FRACTION × (INTENSITY_PERCENTILE-th pct of label)
# Lower INTENSITY_FRACTION  → keep more voxels (less shrinking)
# Higher INTENSITY_FRACTION → keep fewer voxels (more aggressive trimming)
# Good starting range: 0.5 – 0.85
INTENSITY_PERCENTILE = 99      # robust peak reference (99th pct avoids outlier spikes)
INTENSITY_FRACTION   = 0.20    # keep voxels >= 70 % of that peak

MIN_FRAGMENT_VOXELS  = 20      # drop isolated 3D fragments smaller than this after trimming

# Closing: bridges small gaps and smooths pixelated edges after thresholding
# Applied slice-by-slice (2D disk) to avoid connecting things across Z
# CLOSING_RADIUS = 0  disables it
# Start with 2–4; higher values fill bigger gaps but may merge nearby labels
CLOSING_RADIUS = 4

"""
=============================================================================
  END OF CONFIGURATION
=============================================================================
"""

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.morphology import disk
from tqdm import tqdm


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_tif(path: str) -> np.ndarray:
    print(f"Loading : {path}")
    arr = tifffile.imread(path)
    print(f"  shape={arr.shape}  dtype={arr.dtype}")
    return arr


def save_tif(array: np.ndarray, path: str) -> None:
    print(f"Saving  : {path}")
    tifffile.imwrite(path, array, compression="zlib")
    print("  Done.")


def extract_channel(image: np.ndarray, channel: int) -> np.ndarray:
    """Auto-detect axis layout and return a single channel as (Z, Y, X)."""
    print(f"  Original shape: {image.shape}")
    if image.ndim == 4:
        if image.shape[0] <= 4:
            image = np.transpose(image, (1, 2, 3, 0))   # (C,Z,Y,X) → (Z,Y,X,C)
        elif image.shape[1] <= 4:
            image = np.transpose(image, (0, 2, 3, 1))   # (Z,C,Y,X) → (Z,Y,X,C)
    elif image.ndim == 3:
        image = image[..., np.newaxis]
    else:
        raise ValueError(f"Expected 3D or 4D image, got shape {image.shape}.")

    if channel >= image.shape[-1]:
        raise ValueError(f"Image has {image.shape[-1]} channel(s), CHANNEL={channel} requested.")

    vol = image[..., channel]
    print(f"  Extracted channel {channel} → {vol.shape}")
    return vol


# ---------------------------------------------------------------------------
# Per-label shrink
# ---------------------------------------------------------------------------

_STRUCT_26 = ndi.generate_binary_structure(3, 3)  # 26-connectivity for 3D labeling


def _shrink_label(crop_img: np.ndarray, crop_mask: np.ndarray,
                  intensity_percentile: float, intensity_fraction: float,
                  min_fragment_voxels: int, closing_radius: int) -> np.ndarray:
    """
    Returns a boolean array (same shape as crop_mask) with the trimmed region.
    """
    intensities = crop_img[crop_mask]
    if intensities.size == 0:
        return np.zeros_like(crop_mask)

    # Intensity threshold relative to this label's peak
    top_val   = np.percentile(intensities, intensity_percentile)
    threshold = intensity_fraction * top_val
    trimmed   = crop_mask & (crop_img >= threshold)

    # Closing: bridges small gaps between nearby voxels and smooths pixelated
    # edges. Done slice-by-slice so it never connects things across Z.
    # Clipped to crop_mask so it cannot expand outside the original label.
    if closing_radius > 0:
        struct = disk(closing_radius)
        closed = np.zeros_like(trimmed)
        for z in range(trimmed.shape[0]):
            if trimmed[z].any():
                closed[z] = ndi.binary_closing(trimmed[z], structure=struct)
        trimmed = closed & crop_mask

    # Fill any remaining enclosed holes slice-by-slice
    filled = np.zeros_like(trimmed)
    for z in range(trimmed.shape[0]):
        if trimmed[z].any():
            filled[z] = ndi.binary_fill_holes(trimmed[z])

    # Remove small 3D fragments (noise left after trimming)
    labeled, n = ndi.label(filled, structure=_STRUCT_26)
    if n == 0:
        return np.zeros_like(crop_mask)

    sizes = np.bincount(labeled.ravel())
    keep_ids = np.where(sizes >= min_fragment_voxels)[0]
    keep_ids = keep_ids[keep_ids != 0]
    return np.isin(labeled, keep_ids)


# ---------------------------------------------------------------------------
# Main refinement loop
# ---------------------------------------------------------------------------

def shrink_mask(img3d: np.ndarray, mask: np.ndarray,
                intensity_percentile: float, intensity_fraction: float,
                min_fragment_voxels: int, closing_radius: int) -> np.ndarray:

    if img3d.shape != mask.shape:
        raise ValueError(f"Image shape {img3d.shape} != mask shape {mask.shape}.")

    img3d   = img3d.astype(np.float32)
    refined = np.zeros_like(mask)

    labels = np.unique(mask)
    labels = labels[labels != 0]
    print(f"\nFound {len(labels)} label(s). Processing …\n")

    empty = 0
    for label in tqdm(labels, unit="label"):
        coords = np.where(mask == label)
        if coords[0].size == 0:
            continue

        zmin, zmax = coords[0].min(), coords[0].max() + 1
        ymin, ymax = coords[1].min(), coords[1].max() + 1
        xmin, xmax = coords[2].min(), coords[2].max() + 1

        crop_mask = mask[zmin:zmax, ymin:ymax, xmin:xmax] == label
        crop_img  = img3d[zmin:zmax, ymin:ymax, xmin:xmax]

        result = _shrink_label(
            crop_img, crop_mask,
            intensity_percentile, intensity_fraction,
            min_fragment_voxels, closing_radius,
        )

        if not result.any():
            empty += 1
            continue

        refined[zmin:zmax, ymin:ymax, xmin:xmax][result] = label

    if empty:
        print(f"\n[warn] {empty} label(s) were completely removed (too dim or too small).")

    return refined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    image_raw = load_tif(IMAGE_PATH)
    mask      = load_tif(MASK_PATH)

    print("\nAuto-detecting image channel layout …")
    img3d = extract_channel(image_raw, CHANNEL)

    if img3d.shape != mask.shape:
        raise ValueError(f"Image {img3d.shape} and mask {mask.shape} shapes do not match.")

    refined = shrink_mask(
        img3d                = img3d,
        mask                 = mask,
        intensity_percentile = INTENSITY_PERCENTILE,
        intensity_fraction   = INTENSITY_FRACTION,
        min_fragment_voxels  = MIN_FRAGMENT_VOXELS,
        closing_radius       = CLOSING_RADIUS,
    )

    orig = np.unique(mask);    orig = orig[orig != 0]
    ref  = np.unique(refined); ref  = ref[ref != 0]
    print(f"\nInput  labels : {len(orig)}")
    print(f"Output labels : {len(ref)}")
    print(f"Voxels before : {(mask > 0).sum():,}")
    print(f"Voxels after  : {(refined > 0).sum():,}")
    print(f"Reduction     : {100 * (1 - (refined > 0).sum() / max(1, (mask > 0).sum())):.1f}%")

    save_tif(refined.astype(mask.dtype), OUTPUT_PATH)


if __name__ == "__main__":
    main()
