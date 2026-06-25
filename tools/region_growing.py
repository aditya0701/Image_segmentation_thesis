"""
region_growing.py
-----------------
Refines 3D instance segmentation masks using BFS region growing from seed
points. Seeds are loaded from a separate seed image. Labels with no seeds
fall back to top-N% intensity thresholding (same as threshold_rough.py).

Two processing modes:
  SLICE_BY_SLICE = False  →  full 3D region growing per label
  SLICE_BY_SLICE = True   →  2D region growing per Z-slice per label,
                              results merged back under the same label ID

Stopping criterion: gradient magnitude (computed via Gaussian derivative).
A neighbour voxel is accepted only if its gradient < threshold, meaning
the algorithm stops at sharp intensity transitions (edges).

=============================================================================
  CONFIGURATION
=============================================================================
"""

IMAGE_PATH  = r"D:\Thesis_images\All_images_tif\New_images\Surface_Masks\Surface_Masks\airyscan_processed\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100.tif"
MASK_PATH   = r"D:\Thesis_images\Final_Results\Results\vit_l_lm_neo_augmented_PSF_v2\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100_masks.tif"
SEED_PATH   = r"D:\Thesis_images\Final_dataset\all\test\seed\item_0003_ExtractSeedBasedIntensityWindowFilter\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100_ExtractSeedBasedIntensityWindowFilter_Out1_odd_only.tif"
OUTPUT_PATH = r"D:\Thesis_images\Final_Results\Results\vit_l_lm_neo_augmented_PSF_v2\Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100_masks_region_growing.tif"

CHANNEL              = 0      # which channel to use for intensity (0-indexed)

# --- Fallback thresholding (labels with no seeds) --------------------------
TOP_PERCENT          = 17.0   # keep brightest N% of voxels per label
MIN_BLOB_VOXELS      = 10     # discard blobs smaller than this

# --- Region growing --------------------------------------------------------
GRAD_SIGMA           = 1.5    # Gaussian sigma before computing gradient magnitude
                               # larger → smoother gradients, less noise sensitive
                               # smaller → tighter to actual edge, noisier

GRAD_THRESHOLD       = None   # None = auto per label (recommended to start)
                               # float = fixed threshold for all labels
                               # lower value → stops sooner (tighter blobs)
                               # higher value → grows more (fills more of the label)

GRAD_AUTO_PERCENTILE = 75.0   # used only when GRAD_THRESHOLD = None
                               # percentile of gradient values inside the label mask
                               # 50 → stops at median gradient (conservative)
                               # 75 → stops at 75th percentile (moderate, default)
                               # 90 → stops only at very sharp edges (permissive)

# --- Processing mode -------------------------------------------------------
SLICE_BY_SLICE       = False  # False = 3D region growing per label
                               # True  = 2D region growing per Z-slice per label

"""
=============================================================================
  END OF CONFIGURATION
=============================================================================
"""

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_gradient_magnitude
from collections import deque
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
# Channel extraction — same auto-detect logic as threshold_rough.py
# ---------------------------------------------------------------------------

def extract_channel(image: np.ndarray, channel: int) -> np.ndarray:
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
        data = image[..., np.newaxis]
    else:
        raise ValueError(f"Expected a 3-D or 4-D image array, got shape {image.shape}.")

    n_channels = data.shape[-1]
    if channel >= n_channels:
        raise ValueError(
            f"Image has {n_channels} channel(s) but CHANNEL={channel} was requested."
        )

    vol = data[..., channel]
    print(f"  Normalised shape (Z,Y,X,C): {data.shape}")
    print(f"  Extracted channel {channel}  →  volume shape {vol.shape}")
    return vol


def normalise_seed_image(seed_img: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Squeeze any trailing singleton dims and verify the seed image matches
    the target (Z, Y, X) volume shape.
    """
    s = seed_img
    while s.ndim > 3 and s.shape[-1] == 1:
        s = s[..., 0]
    if s.shape != target_shape:
        raise ValueError(
            f"Seed image shape {s.shape} does not match volume shape {target_shape}.\n"
            "Ensure the seed image covers the same spatial extent as the mask."
        )
    return s


# ---------------------------------------------------------------------------
# Helpers
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


def _threshold_3d(crop_img, crop_mask, top_percent, min_blob_voxels) -> np.ndarray:
    intensities = crop_img[crop_mask]
    if intensities.size == 0:
        return np.zeros(crop_mask.shape, dtype=bool)
    thr    = np.percentile(intensities, 100.0 - top_percent)
    bright = crop_mask & (crop_img >= thr)
    blobs, _ = ndi.label(bright)
    return _remove_small(blobs, min_blob_voxels) > 0


def _threshold_2d(img_sl, mask_sl, top_percent, min_blob_voxels) -> np.ndarray:
    intensities = img_sl[mask_sl]
    if intensities.size == 0:
        return np.zeros(mask_sl.shape, dtype=bool)
    thr    = np.percentile(intensities, 100.0 - top_percent)
    bright = mask_sl & (img_sl >= thr)
    blobs, _ = ndi.label(bright)
    return _remove_small(blobs, min_blob_voxels) > 0


def _auto_threshold(grad: np.ndarray, mask: np.ndarray, percentile: float) -> float:
    vals = grad[mask]
    return float(np.percentile(vals, percentile)) if vals.size > 0 else np.inf


# ---------------------------------------------------------------------------
# BFS Region Growing — 3D
# ---------------------------------------------------------------------------

def region_grow_3d(
    crop_mask: np.ndarray,
    grad: np.ndarray,
    seed_coords: list,
    grad_threshold: float,
) -> np.ndarray:
    """
    BFS flood from seed_coords (local crop coordinates, list of (z,y,x)).
    Expands into neighbours that are:
      - inside crop_mask (within the label's original footprint)
      - below grad_threshold (not an edge)
    Returns boolean grown region.
    """
    shape   = crop_mask.shape
    result  = np.zeros(shape, dtype=bool)
    visited = np.zeros(shape, dtype=bool)
    queue   = deque()

    for z, y, x in seed_coords:
        z, y, x = int(z), int(y), int(x)
        if (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]
                and crop_mask[z, y, x] and not visited[z, y, x]):
            visited[z, y, x] = True
            result[z, y, x]  = True
            queue.append((z, y, x))

    neighbors = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]

    while queue:
        z, y, x = queue.popleft()
        for dz, dy, dx in neighbors:
            nz, ny, nx = z + dz, y + dy, x + dx
            if (0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]
                    and not visited[nz, ny, nx]
                    and crop_mask[nz, ny, nx]
                    and grad[nz, ny, nx] < grad_threshold):
                visited[nz, ny, nx] = True
                result[nz, ny, nx]  = True
                queue.append((nz, ny, nx))

    return result


# ---------------------------------------------------------------------------
# BFS Region Growing — 2D
# ---------------------------------------------------------------------------

def region_grow_2d(
    mask_sl: np.ndarray,
    grad_sl: np.ndarray,
    seed_coords: list,
    grad_threshold: float,
) -> np.ndarray:
    """
    BFS flood from seed_coords (local 2D crop coordinates, list of (y,x)).
    Same stopping logic as 3D version but for a single Z-slice.
    Returns boolean grown region.
    """
    shape   = mask_sl.shape
    result  = np.zeros(shape, dtype=bool)
    visited = np.zeros(shape, dtype=bool)
    queue   = deque()

    for y, x in seed_coords:
        y, x = int(y), int(x)
        if (0 <= y < shape[0] and 0 <= x < shape[1]
                and mask_sl[y, x] and not visited[y, x]):
            visited[y, x] = True
            result[y, x]  = True
            queue.append((y, x))

    neighbors = [(-1,0),(1,0),(0,-1),(0,1)]

    while queue:
        y, x = queue.popleft()
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if (0 <= ny < shape[0] and 0 <= nx < shape[1]
                    and not visited[ny, nx]
                    and mask_sl[ny, nx]
                    and grad_sl[ny, nx] < grad_threshold):
                visited[ny, nx] = True
                result[ny, nx]  = True
                queue.append((ny, nx))

    return result


# ---------------------------------------------------------------------------
# Per-label refinement
# ---------------------------------------------------------------------------

def refine_mask(
    img3d: np.ndarray,
    mask: np.ndarray,
    seed_img: np.ndarray,
    top_percent: float,
    grad_sigma: float,
    grad_threshold_cfg,
    grad_auto_percentile: float,
    min_blob_voxels: int,
    slice_by_slice: bool,
) -> np.ndarray:

    if img3d.shape != mask.shape:
        raise ValueError(f"Image shape {img3d.shape} != mask shape {mask.shape}.")
    if seed_img.shape != mask.shape:
        raise ValueError(f"Seed image shape {seed_img.shape} != mask shape {mask.shape}.")

    img3d   = img3d.astype(np.float32)
    refined = np.zeros_like(mask)

    labels = np.unique(mask)
    labels = labels[labels != 0]
    print(f"\nFound {len(labels)} label(s).  Mode: {'2D slice-by-slice' if slice_by_slice else '3D'}\n")

    no_seed_count = 0

    for label in tqdm(labels, unit="label"):

        # ---- Bounding box -------------------------------------------------
        coords_idx = np.where(mask == label)
        if len(coords_idx[0]) == 0:
            continue

        zmin, zmax = coords_idx[0].min(), coords_idx[0].max() + 1
        ymin, ymax = coords_idx[1].min(), coords_idx[1].max() + 1
        xmin, xmax = coords_idx[2].min(), coords_idx[2].max() + 1

        crop_mask = (mask    [zmin:zmax, ymin:ymax, xmin:xmax] == label)
        crop_img  =  img3d  [zmin:zmax, ymin:ymax, xmin:xmax]
        crop_seed =  seed_img[zmin:zmax, ymin:ymax, xmin:xmax]

        # ---- Seeds inside this label --------------------------------------
        seed_voxels = crop_mask & (crop_seed != 0)
        has_seeds   = bool(seed_voxels.any())
        if not has_seeds:
            no_seed_count += 1

        refined_view = refined[zmin:zmax, ymin:ymax, xmin:xmax]

        # ==================================================================
        # 3D mode
        # ==================================================================
        if not slice_by_slice:
            if has_seeds:
                grad = gaussian_gradient_magnitude(crop_img, sigma=grad_sigma)
                thr  = (grad_threshold_cfg
                        if grad_threshold_cfg is not None
                        else _auto_threshold(grad, crop_mask, grad_auto_percentile))
                seed_coords = list(zip(*np.where(seed_voxels)))
                grown = region_grow_3d(crop_mask, grad, seed_coords, thr)
            else:
                grown = _threshold_3d(crop_img, crop_mask, top_percent, min_blob_voxels)

            refined_view[grown] = label

        # ==================================================================
        # 2D slice-by-slice mode
        # ==================================================================
        else:
            for zi in range(crop_mask.shape[0]):
                mask_sl = crop_mask[zi]
                if not mask_sl.any():
                    continue

                img_sl  = crop_img [zi]
                seed_sl = crop_seed[zi]

                seed_px_2d   = mask_sl & (seed_sl != 0)
                has_seeds_2d = bool(seed_px_2d.any())

                if has_seeds_2d:
                    grad_sl = gaussian_gradient_magnitude(img_sl, sigma=grad_sigma)
                    thr_2d  = (grad_threshold_cfg
                               if grad_threshold_cfg is not None
                               else _auto_threshold(grad_sl, mask_sl, grad_auto_percentile))
                    seed_coords_2d = list(zip(*np.where(seed_px_2d)))
                    grown_2d = region_grow_2d(mask_sl, grad_sl, seed_coords_2d, thr_2d)
                else:
                    grown_2d = _threshold_2d(img_sl, mask_sl, top_percent, min_blob_voxels)

                refined_view[zi][grown_2d] = label

    print(f"\n  Labels that used threshold fallback (no seeds): {no_seed_count} / {len(labels)}")
    return refined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    image_raw = load_tif(IMAGE_PATH)
    mask      = load_tif(MASK_PATH)
    seed_raw  = load_tif(SEED_PATH)

    print(f"\nMask shape : {mask.shape}  (expected Z, Y, X)")
    print("Auto-detecting image channel layout …")
    img3d = extract_channel(image_raw, CHANNEL)

    print("Normalising seed image …")
    seed_img = normalise_seed_image(seed_raw, img3d.shape)
    n_seeds  = int((seed_img != 0).sum())
    print(f"  Seed voxels found: {n_seeds}")

    refined = refine_mask(
        img3d               = img3d,
        mask                = mask,
        seed_img            = seed_img,
        top_percent         = TOP_PERCENT,
        grad_sigma          = GRAD_SIGMA,
        grad_threshold_cfg  = GRAD_THRESHOLD,
        grad_auto_percentile= GRAD_AUTO_PERCENTILE,
        min_blob_voxels     = MIN_BLOB_VOXELS,
        slice_by_slice      = SLICE_BY_SLICE,
    )

    orig_labels = np.unique(mask);    orig_labels = orig_labels[orig_labels != 0]
    ref_labels  = np.unique(refined); ref_labels  = ref_labels[ref_labels != 0]
    print(f"\nOriginal unique labels : {len(orig_labels)}")
    print(f"Refined  unique labels : {len(ref_labels)}")
    print(f"Total refined voxels   : {(refined > 0).sum()}")

    save_tif(refined, OUTPUT_PATH)


if __name__ == "__main__":
    main()
