"""
refine_masks_v7.py
------------------
Refines instance segmentation by region-growing each seed inside its
parent label, stopping when intensity drops below a per-label threshold.
The grown blob inherits the parent label.

Algorithm (per input label N):
  1. Crop the image and seeds to the bounding box of label N.
  2. Compute a per-label intensity floor: the top INTENSITY_FLOOR_PERCENTILE%
     of voxel intensities inside label N.
  3. Build a growth mask = (inside label N) AND (intensity >= floor).
  4. Run seeded watershed on -smoothed_intensity, restricted to the growth mask.
     Result: each seed grows into the bright region around it. Adjacent seeds
     are split at intensity ridges. Dim voxels stay as 0 (background).
  5. Assign label N to every grown voxel.

  If label N contains NO seed voxels, fall back to peak detection (top
  TOP_PERCENT% intensity as candidate seeds).

=============================================================================
  CONFIGURATION
=============================================================================
"""

IMAGE_PATH  = r"D:\Thesis_images\All_images_tif\New_images\Surface_Masks\Surface_Masks\airyscan_processed\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100.tif"
MASK_PATH   = r"D:\Thesis_images\Final_Results\Results\vit_l_lm_neo_augmented_All_v2\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100_masks.tif"
SEEDS_PATH  = r"D:\Thesis_images\Final_dataset\all\test\seed\item_0003_ExtractSeedBasedIntensityWindowFilter\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100_ExtractSeedBasedIntensityWindowFilter_Out1_odd_only.tif"
OUTPUT_PATH = r"D:\Thesis_images\Final_Results\Results\vit_l_lm_neo_augmented_All_v2\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100_masks_final.tif"

CHANNEL          = 0
SMOOTH_SIGMA     = 1.0
SLICE_BY_SLICE   = True
MIN_BLOB_VOXELS  = 10

# ---- Region-growing controls -------------------------------------------
# Grow only into voxels brighter than this percentile of the parent label.
# Higher value = bigger blobs (more permissive). Lower = tighter blobs.
# The code computes: floor = percentile(intensities, 100 - pct)
#   50.0  -> floor = median  -> grow into top 50% -> LOOSER (recommended start)
#   30.0  -> floor = 70th percentile -> grow into top 30% -> TIGHTER
#   20.0  -> tighter still, more dim background excluded
#   100.0 -> no threshold, blobs fill entire parent mask (NOT recommended;
#            this is what v6 did and looked Voronoi-like)
INTENSITY_FLOOR_PERCENTILE = 50.0

# Watershed tuning
COMPACTNESS    = 0.001    # slight roundness bias, reduces irregular leakage
WATERSHED_LINE = False    # only useful in per_seed mode — in preserve mode it
                          # just punches 1-voxel holes in the output for no benefit

# ---- Fallback (label has no seed voxels) -------------------------------
TOP_PERCENT  = 20.0       # use top 20% intensity as candidate seed region

# ---- Output labeling ---------------------------------------------------
# "preserve" : grown blobs inherit the parent label (RECOMMENDED — matches
#              "labeled the same as the region they are found in").
# "per_seed" : each seed becomes a new unique instance.
OUTPUT_LABEL_MODE = "preserve"

"""
=============================================================================
  END OF CONFIGURATION
=============================================================================
"""

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.segmentation import watershed
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
    """Auto-detect axis layout, return a single channel as (Z, Y, X)."""
    print(f"  Original shape: {image.shape}")
    if image.ndim == 4:
        if image.shape[0] <= 4:
            print("  Detected (C, Z, Y, X) -> transposing")
            data = np.transpose(image, (1, 2, 3, 0))
        elif image.shape[1] <= 4:
            print("  Detected (Z, C, Y, X) -> transposing")
            data = np.transpose(image, (0, 2, 3, 1))
        else:
            print("  Detected (Z, Y, X, C)")
            data = image
    elif image.ndim == 3:
        print("  Detected (Z, Y, X)")
        data = image[..., np.newaxis]
    else:
        raise ValueError(f"Expected 3D or 4D, got shape {image.shape}.")

    if channel >= data.shape[-1]:
        raise ValueError(f"Image has {data.shape[-1]} channel(s), CHANNEL={channel} requested.")

    vol = data[..., channel]
    print(f"  Extracted channel {channel} -> {vol.shape}")
    return vol


def load_seeds_volume(path: str, reference_shape: tuple) -> np.ndarray:
    raw = load_tif(path)
    if raw.ndim == 4:
        if raw.shape[0] <= 4:
            vol = raw[0]
        elif raw.shape[1] <= 4:
            vol = raw[:, 0, :, :]
        else:
            vol = raw[..., 0]
    else:
        vol = raw

    if vol.shape != reference_shape:
        raise ValueError(
            f"Seeds volume shape {vol.shape} != mask shape {reference_shape}."
        )

    seeds_bool = vol > 0
    print(f"  Seed voxels: {seeds_bool.sum():,}")
    return seeds_bool


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def _dilate_markers(markers: np.ndarray) -> np.ndarray:
    """Dilate point markers by 1 voxel in each dimension (ndim-aware)."""
    size = (3,) * markers.ndim
    return ndi.grey_dilation(markers, size=size)


def _get_markers(crop_smooth, crop_mask, crop_seeds, has_seeds,
                 top_percent, label_id):
    """
    Returns integer-labeled markers (one label per connected seed component).
    Returns None if no markers found (logs a warning).
    """
    if has_seeds:
        seed_mask = crop_mask & crop_seeds
        markers, n = ndi.label(seed_mask)
        if n == 0:
            print(f"  [warn] label {label_id}: seed voxels present but no connected components found")
            return None
        return _dilate_markers(markers)

    # Fallback: connected components of the top TOP_PERCENT% intensity region
    intensities = crop_smooth[crop_mask]
    if intensities.size == 0:
        print(f"  [warn] label {label_id}: no intensity data, skipping")
        return None
    peak_threshold = np.percentile(intensities, 100.0 - top_percent)
    seed_mask = crop_mask & (crop_smooth >= peak_threshold)
    markers, n = ndi.label(seed_mask)
    if n == 0:
        print(f"  [warn] label {label_id}: fallback found no seed regions, skipping")
        return None
    return _dilate_markers(markers)


# ---------------------------------------------------------------------------
# Region growing (intensity-thresholded watershed)
# ---------------------------------------------------------------------------

def _grow_blobs(crop_smooth, crop_mask, markers,
                intensity_floor_pct, compactness, watershed_line):
    """
    Region-grow each seed inside crop_mask, stopping where intensity falls
    below the per-label floor. Returns labeled int array, 0 = background.
    """
    # Per-label intensity floor: keep voxels brighter than the
    # (100 - INTENSITY_FLOOR_PERCENTILE)-th percentile.
    intensities = crop_smooth[crop_mask]
    if intensities.size == 0:
        return np.zeros(crop_mask.shape, dtype=np.int32)

    floor = np.percentile(intensities, 100.0 - intensity_floor_pct)

    # Growth allowed where: inside parent mask AND intensity >= floor.
    # Always include the seed voxels themselves so they don't get excluded
    # by a high threshold.
    bright_enough = crop_smooth >= floor
    growth_mask = crop_mask & (bright_enough | (markers > 0))

    # Watershed flood on negated intensity. Floods stop at growth_mask edges
    # (the "edge" the user wants — where intensity drops off) and at
    # intensity ridges between adjacent seeds (dips between blobs).
    ws_image = -crop_smooth.astype(np.float32)
    labeled = watershed(
        ws_image, markers=markers, mask=growth_mask,
        compactness=compactness, watershed_line=watershed_line,
    )
    return labeled


def _remove_small_components(labeled, min_size):
    """Drop labeled components below min_size voxels."""
    if min_size <= 0:
        return labeled
    sizes = np.bincount(labeled.ravel())
    too_small = np.where(sizes < min_size)[0]
    too_small = too_small[too_small != 0]
    if too_small.size:
        keep = ~np.isin(labeled, too_small)
        return (labeled * keep).astype(labeled.dtype)
    return labeled


def _process_crop(crop_smooth, crop_mask, crop_seeds, has_seeds,
                  intensity_floor_pct, top_percent,
                  min_blob_voxels, compactness, watershed_line, label_id):
    """Return labeled int crop. Background = 0, each blob has a unique local id."""
    markers = _get_markers(
        crop_smooth, crop_mask, crop_seeds, has_seeds,
        top_percent, label_id,
    )
    if markers is None:
        # No seeds available — leave region as background (label assigned later)
        return np.zeros(crop_mask.shape, dtype=np.int32)

    labeled = _grow_blobs(
        crop_smooth, crop_mask, markers,
        intensity_floor_pct, compactness, watershed_line,
    )
    labeled = _remove_small_components(labeled, min_blob_voxels)
    return labeled.astype(np.int32)


# ---------------------------------------------------------------------------
# Main refinement
# ---------------------------------------------------------------------------

def refine_mask(
    img3d, mask, seeds_vol, *,
    smooth_sigma, slice_by_slice,
    intensity_floor_pct, top_percent, min_blob_voxels,
    compactness, watershed_line, output_label_mode,
):
    if img3d.shape != mask.shape:
        raise ValueError("Image and mask shape mismatch.")
    if seeds_vol.shape != mask.shape:
        raise ValueError("Seeds and mask shape mismatch.")

    print(f"Smoothing (sigma={smooth_sigma}) ...")
    img_smooth = ndi.gaussian_filter(img3d.astype(np.float32), sigma=smooth_sigma)

    refined = np.zeros_like(mask, dtype=np.int32)
    next_label = 1  # only used for "per_seed" mode

    labels = np.unique(mask)
    labels = labels[labels != 0]

    seeded = 0
    fallback = 0

    mode_str = "2D slice-by-slice" if slice_by_slice else "full 3D"
    print(f"Refining {len(labels)} label(s)  |  floor={intensity_floor_pct}%  |  {mode_str}\n")

    for label in tqdm(labels, unit="label"):
        coords = np.where(mask == label)
        if coords[0].size == 0:
            continue

        zmin, zmax = coords[0].min(), coords[0].max() + 1
        ymin, ymax = coords[1].min(), coords[1].max() + 1
        xmin, xmax = coords[2].min(), coords[2].max() + 1

        crop_mask   = mask[zmin:zmax, ymin:ymax, xmin:xmax] == label
        crop_smooth = img_smooth[zmin:zmax, ymin:ymax, xmin:xmax]
        crop_seeds  = seeds_vol[zmin:zmax, ymin:ymax, xmin:xmax]

        if not crop_smooth[crop_mask].size:
            continue

        has_seeds = bool((crop_seeds & crop_mask).any())
        if has_seeds:
            seeded += 1
        else:
            fallback += 1

        if slice_by_slice:
            crop_labeled = np.zeros(crop_mask.shape, dtype=np.int32)
            offset = 0
            for z in range(crop_mask.shape[0]):
                if not crop_mask[z].any():
                    continue
                slab_labeled = _process_crop(
                    crop_smooth[z], crop_mask[z], crop_seeds[z], has_seeds,
                    intensity_floor_pct, top_percent,
                    min_blob_voxels, compactness, watershed_line, label_id=label,
                )
                slab_labeled = np.where(
                    slab_labeled > 0, slab_labeled + offset, 0
                ).astype(np.int32)
                crop_labeled[z] = slab_labeled
                offset = int(crop_labeled.max())
        else:
            crop_labeled = _process_crop(
                crop_smooth, crop_mask, crop_seeds, has_seeds,
                intensity_floor_pct, top_percent,
                min_blob_voxels, compactness, watershed_line, label_id=label,
            )

        # Write back into the full volume
        if output_label_mode == "preserve":
            # All grown voxels in this label region get the parent label
            refined[zmin:zmax, ymin:ymax, xmin:xmax][crop_labeled > 0] = label
        elif output_label_mode == "per_seed":
            # Each connected blob gets a fresh unique label
            unique_local = np.unique(crop_labeled)
            unique_local = unique_local[unique_local != 0]
            for local_id in unique_local:
                region = crop_labeled == local_id
                refined[zmin:zmax, ymin:ymax, xmin:xmax][region] = next_label
                next_label += 1
        else:
            raise ValueError(f"Unknown OUTPUT_LABEL_MODE: {output_label_mode}")

    print(f"\nLabels with seed data : {seeded}")
    print(f"Intensity fallbacks   : {fallback}")
    return refined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    image_raw = load_tif(IMAGE_PATH)
    mask      = load_tif(MASK_PATH)

    print(f"\nMask shape: {mask.shape}")
    print("Detecting image channel layout ...")
    img3d = extract_channel(image_raw, CHANNEL)

    print("\nLoading seed volume ...")
    seeds_vol = load_seeds_volume(SEEDS_PATH, reference_shape=img3d.shape)

    refined = refine_mask(
        img3d, mask, seeds_vol,
        smooth_sigma         = SMOOTH_SIGMA,
        slice_by_slice       = SLICE_BY_SLICE,
        intensity_floor_pct  = INTENSITY_FLOOR_PERCENTILE,
        top_percent          = TOP_PERCENT,
        min_blob_voxels      = MIN_BLOB_VOXELS,
        compactness          = COMPACTNESS,
        watershed_line       = WATERSHED_LINE,
        output_label_mode    = OUTPUT_LABEL_MODE,
    )

    orig = np.unique(mask);    orig = orig[orig != 0]
    new  = np.unique(refined); new  = new[new != 0]
    print(f"\nInput  unique labels : {len(orig)}")
    print(f"Output unique labels : {len(new)}")
    print(f"Refined voxel count  : {(refined > 0).sum():,}")
    print(f"Coverage of input    : {(refined > 0).sum() / max(1, (mask > 0).sum()) * 100:.1f}%")

    save_tif(refined, OUTPUT_PATH)


if __name__ == "__main__":
    main()