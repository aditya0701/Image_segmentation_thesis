"""
refine_masks_v3_optimized.py
----------------------------
Refines 3D instance segmentation masks where each label is a large connected
region containing multiple bright spots. 

Original strict threshold logic applied: Retains the original "rough" pixelated
boundaries but utilizes Bounding Box extraction and float32 conversion to 
process extremely fast and prevent memory (RAM) crashes.

=============================================================================
  CONFIGURATION
=============================================================================
"""

IMAGE_PATH  = r"D:\Thesis_images\All_images_tif\New_images\Surface_Masks\Surface_Masks\airyscan_processed\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100.tif"
MASK_PATH   = r"D:\Thesis_images\Final_Results\Results\vit_l_lm_neo_augmented_All_v2\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100_masks.tif"
OUTPUT_PATH = r"D:\Thesis_images\Final_Results\Results\vit_l_lm_neo_augmented_All_v2\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100_masks_final_rough.tif"

CHANNEL         = 0        # which channel to use for intensity (0-indexed)
                           # shape is auto-detected — no need to set it manually

SEED_PERCENT    = 15.0     # top N% of voxels per label become seed regions
GROWTH_PERCENT  = 60.0     # watershed grows from seeds into top M% (M > N);
                           # raise to keep more of the label, lower to cut more
MIN_BLOB_VOXELS = 10       # discard blobs smaller than this (noise filter)
CLOSING_RADIUS  = 5       # morphological closing after watershed (0 = disabled)
CLOSING_MODE    = "2d"    # "2d" = slice-by-slice (good for anisotropic Z) | "3d" = volumetric ball

"""
=============================================================================
  END OF CONFIGURATION
=============================================================================
"""

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.morphology import ball, disk
from tqdm import tqdm


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_tif(path: str) -> np.ndarray:
    print(f"Loading : {path}")
    arr = tifffile.imread(path)
    print(f"  raw shape = {arr.shape}  dtype = {arr.dtype}")
    return arr


def save_tif(array: np.ndarray, path: str) -> None:
    print(f"Saving  : {path}")
    tifffile.imwrite(path, array, compression="zlib")
    print("  Done.")


# ---------------------------------------------------------------------------
# Channel extraction  —  auto-detects layout, always returns (Z, Y, X)
# ---------------------------------------------------------------------------

def extract_channel(image: np.ndarray, channel: int) -> np.ndarray:
    """
    Auto-detect the axis layout of a 3-D or 4-D image, normalise to
    (Z, Y, X, C), then return a single channel as (Z, Y, X).
    """
    print(f"  Original shape: {image.shape}")

    if image.ndim == 4:
        if image.shape[0] <= 4:
            print("  Detected (C, Z, Y, X) → transposing to (Z, Y, X, C)")
            data = np.transpose(image, (1, 2, 3, 0))
        elif image.shape[1] <= 4:
            print("  Detected (Z, C, Y, X) → transposing to (Z, Y, X, C)")
            data = np.transpose(image, (0, 2, 3, 1))
        else:
            print("  Detected (Z, Y, X, C) — keeping orientation")
            data = image
    elif image.ndim == 3:
        print("  Detected (Z, Y, X) — adding dummy channel dimension")
        data = image[..., np.newaxis]          # (Z, Y, X, 1)
    else:
        raise ValueError(
            f"Expected a 3-D or 4-D image array, got shape {image.shape}."
        )

    n_channels = data.shape[-1]
    if channel >= n_channels:
        raise ValueError(
            f"Image has {n_channels} channel(s) but CHANNEL={channel} was requested."
        )

    vol = data[..., channel]                   # (Z, Y, X)
    print(f"  Normalised shape (Z,Y,X,C): {data.shape}")
    print(f"  Extracted channel {channel}  →  volume shape {vol.shape}")
    return vol


# ---------------------------------------------------------------------------
# Core refinement (Original Logic + Bounding Box Optimization)
# ---------------------------------------------------------------------------

def _apply_closing(vol: np.ndarray, radius: int, mode: str) -> np.ndarray:
    if mode == "2d":
        struct = disk(radius)
        out = np.zeros_like(vol)
        for z in range(vol.shape[0]):
            out[z] = ndi.binary_closing(vol[z], structure=struct)
        return out
    return ndi.binary_closing(vol, structure=ball(radius))


def refine_mask(
    img3d: np.ndarray,
    mask: np.ndarray,
    seed_percent: float,
    growth_percent: float,
    min_blob_voxels: int,
    closing_radius: int = 0,
    closing_mode: str = "3d",
) -> np.ndarray:

    if img3d.shape != mask.shape:
        raise ValueError(
            f"Image volume shape {img3d.shape} != mask shape {mask.shape}.\n"
            "The auto-detected channel shape does not match the mask."
        )

    img3d = img3d.astype(np.float32)
    refined = np.zeros_like(mask)

    labels = np.unique(mask)
    labels = labels[labels != 0]
    print(f"\nFound {len(labels)} label(s). Processing …\n")

    for label in tqdm(labels, unit="label"):

        # ---- Bounding Box Extraction --------------------------------------
        coords_idx = np.where(mask == label)
        if len(coords_idx[0]) == 0:
            continue

        zmin, zmax = coords_idx[0].min(), coords_idx[0].max() + 1
        ymin, ymax = coords_idx[1].min(), coords_idx[1].max() + 1
        xmin, xmax = coords_idx[2].min(), coords_idx[2].max() + 1

        crop_mask = mask[zmin:zmax, ymin:ymax, xmin:xmax] == label
        crop_img  = img3d[zmin:zmax, ymin:ymax, xmin:xmax]

        # ---- per-label intensity sampling ---------------------------------
        intensities = crop_img[crop_mask]
        if intensities.size == 0:
            continue

        # ---- seeds: connected components of top SEED_PERCENT% ------------
        seed_thr  = np.percentile(intensities, 100.0 - seed_percent)
        seed_mask = crop_mask & (crop_img >= seed_thr)
        markers, n_seeds = ndi.label(seed_mask)
        if n_seeds == 0:
            continue

        # Dilate markers slightly so watershed doesn't start from single voxels
        markers = ndi.grey_dilation(markers, size=(3, 3, 3))

        # ---- growth mask: top GROWTH_PERCENT% region ----------------------
        growth_thr  = np.percentile(intensities, 100.0 - growth_percent)
        growth_mask = crop_mask & (crop_img >= growth_thr)

        # ---- watershed: flood from seeds, stop at growth_mask boundary ----
        img_region = np.where(crop_mask, crop_img, 0.0)
        blobs = watershed(
            -img_region,
            markers=markers,
            mask=growth_mask,
            compactness=0.001,
        )
        blobs = _remove_small(blobs, min_blob_voxels)
        final = blobs > 0
        if closing_radius > 0:
            # Union with original so erosion never shrinks the outer boundary
            final = (final | _apply_closing(final, closing_radius, closing_mode)) & crop_mask

        refined[zmin:zmax, ymin:ymax, xmin:xmax][final] = label

    return refined


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _remove_small(labeled: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return labeled
    out   = labeled.copy()
    sizes = np.bincount(labeled.ravel())
    for comp_id in np.where(sizes < min_size)[0]:
        if comp_id != 0:
            out[out == comp_id] = 0
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    image_raw = load_tif(IMAGE_PATH)
    mask      = load_tif(MASK_PATH)

    print(f"\nMask shape : {mask.shape}  (expected Z, Y, X)")
    print("Auto-detecting image channel layout …")

    img3d = extract_channel(image_raw, CHANNEL)

    refined = refine_mask(
        img3d           = img3d,
        mask            = mask,
        seed_percent    = SEED_PERCENT,
        growth_percent  = GROWTH_PERCENT,
        min_blob_voxels = MIN_BLOB_VOXELS,
        closing_radius  = CLOSING_RADIUS,
        closing_mode    = CLOSING_MODE,
    )

    orig_labels = np.unique(mask);    orig_labels = orig_labels[orig_labels != 0]
    ref_labels  = np.unique(refined); ref_labels  = ref_labels[ref_labels != 0]
    print(f"\nOriginal unique labels : {len(orig_labels)}")
    print(f"Refined  unique labels : {len(ref_labels)}  (same values, smaller footprint)")
    print(f"Total refined voxels   : {(refined > 0).sum()}")

    save_tif(refined, OUTPUT_PATH)


if __name__ == "__main__":
    main()