"""
Cellpose 3D Training Script (Fixed)
=====================================
Fixes applied vs previous version:
  1. 'cpsam' not found  → use model_type= not pretrained_model=
                          ensure_model_downloaded() handles first-run download
  2. 'channels' kwarg   → removed from train_seg entirely;
                          channel selection happens in load_data instead
  3. 'channel_axis'     → removed (not in Cellpose 3.x train_seg)
"""

import os
import numpy as np
import tifffile
import torch
from pathlib import Path
from scipy.ndimage import zoom
from cellpose import models, train

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Paths
    "IMAGE_FOLDER":    r"/home/students/rawat/czi_output/processed_images",
    "LABEL_FOLDER":    r"/home/students/rawat/Dataset/Normal_labels",
    "SAVE_FOLDER":     r"/work/scratch/rawat/models/cellpose3d",

    # Training
    "EPOCHS":          100,
    "LEARNING_RATE":   1e-5,
    "BATCH_SIZE":      8,
    "MODEL_NAME":      "cellpose3d_calyx",

    # ── Pretrained model ──────────────────────────────────────────────────────
    # Newer Cellpose maps "cyto3" → "cpsam" (Cellpose-SAM).
    # First run downloads weights automatically via ensure_model_downloaded().
    # If download fails or you are offline, switch to "cyto2".
    "MODEL_TYPE":      "cyto3",
    "DOWNLOAD_MODEL":  True,

    # Data
    "IMAGE_GLOB":      "*.ome.tif",
    "LABEL_SUFFIX":    "_label.tif",   # replaces ".ome.tif" in image filename

    # ── Channel selection ─────────────────────────────────────────────────────
    # Done during preprocessing — NOT passed to train_seg.
    #
    # CHANNEL_IDX : index of your main (cytoplasm) channel, or None = mean all
    # NUCLEUS_IDX : index of nucleus channel, or None = single-channel mode
    #
    # Examples:
    #   Single-channel confocal:  CHANNEL_IDX=None, NUCLEUS_IDX=None
    #   Two-channel (cyto+nuc):   CHANNEL_IDX=0,    NUCLEUS_IDX=1
    "CHANNEL_IDX":     None,
    "NUCLEUS_IDX":     None,

    # Voxel sizes in µm — SET THESE TO YOUR ACTUAL VALUES
    "XY_VOXEL_UM":     0.1,
    "Z_VOXEL_UM":      1.0,

    # Axis strategy
    # "xy"         → XY slices only (recommended for 5-10x anisotropy)
    # "xy+xz+yz"   → all three axes
    "AXES":            "xy",
    "XZ_YZ_STRIDE":    50,             # sample every Nth XZ/YZ slice

    # Validation split at the volume level
    "VAL_FRACTION":    0.1,

    "USE_GPU":         True,
}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
def ensure_model_downloaded(model_type: str, gpu: bool) -> str:
    """
    Instantiating CellposeModel triggers automatic weight download on first run.
    Returns the model_type that was successfully loaded (may fall back to cyto2).
    """
    print(f"Checking model weights: '{model_type}' ...")
    try:
        _ = models.CellposeModel(gpu=gpu, model_type=model_type)
        print(f"  '{model_type}' ready.")
        return model_type
    except Exception as e:
        print(f"  Could not load '{model_type}': {e}")
        print("  Falling back to 'cyto2'.")
        _ = models.CellposeModel(gpu=gpu, model_type="cyto2")
        return "cyto2"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fix_image_dims(img: np.ndarray) -> np.ndarray:
    """Return (Z, Y, X, C) regardless of input axis order."""
    if img.ndim == 2:
        return img[np.newaxis, ..., np.newaxis]         # (1,Y,X,1)
    if img.ndim == 3:
        return img[..., np.newaxis]                     # (Z,Y,X,1)
    if img.ndim == 4:
        min_ax = int(np.argmin(img.shape))
        if min_ax == 0:                                 # (C,Z,Y,X) → (Z,Y,X,C)
            img = np.transpose(img, (1, 2, 3, 0))
    return img


def fix_label_dims(lbl: np.ndarray) -> np.ndarray:
    if lbl.ndim == 4:
        lbl = lbl.squeeze()
    if lbl.ndim == 2:
        lbl = lbl[np.newaxis]
    return lbl.astype(np.int32)


def select_channels(img: np.ndarray,
                    channel_idx,
                    nucleus_idx) -> np.ndarray:
    """
    img : (Z, Y, X, C)
    Returns (Z, Y, X, 1) or (Z, Y, X, 2).
    Channel selection is done HERE so train_seg never needs a 'channels' arg.
    """
    if img.ndim == 3:
        return img[..., np.newaxis]

    if channel_idx is None:
        cyto = img.mean(axis=-1, keepdims=True)         # mean → grayscale
    else:
        cyto = img[..., [channel_idx]]

    if nucleus_idx is not None and nucleus_idx < img.shape[-1]:
        nuc = img[..., [nucleus_idx]]
        return np.concatenate([cyto, nuc], axis=-1)     # (Z,Y,X,2)

    return cyto                                         # (Z,Y,X,1)


def normalize_slice(sl: np.ndarray) -> np.ndarray:
    """Per-channel 1-99 percentile norm → float32 [0,1]."""
    sl = sl.astype(np.float32)
    for c in range(sl.shape[-1]):
        lo, hi = np.percentile(sl[..., c], (1, 99))
        sl[..., c] = (sl[..., c] - lo) / (hi - lo) if hi > lo else 0.0
    return np.clip(sl, 0.0, 1.0)


def extract_slices(volume: np.ndarray,
                   labels: np.ndarray,
                   axes: str,
                   xy_voxel: float,
                   z_voxel: float,
                   stride: int):
    """
    volume : (Z, Y, X, C) float32
    labels : (Z, Y, X)    int32
    Returns lists of 2-D (H, W, C) images and (H, W) label arrays.
    """
    imgs, lbls = [], []
    aniso = z_voxel / xy_voxel

    def to_cellpose_format(sl_img: np.ndarray) -> np.ndarray:
        """
        Cellpose train_seg expects images as (C, Ly, Lx) — channels first.
        Input sl_img is (Ly, Lx, C) or (Ly, Lx).
        Returns (C, Ly, Lx) for multi-channel or (Ly, Lx) for single channel.
        """
        if sl_img.ndim == 2:
            return sl_img                               # already (Ly, Lx)
        # (Ly, Lx, C) → (C, Ly, Lx)
        sl_img = np.transpose(sl_img, (2, 0, 1))
        if sl_img.shape[0] == 1:
            return sl_img.squeeze(0)                    # (1,Ly,Lx) → (Ly,Lx)
        return sl_img                                   # (2, Ly, Lx)

    if "xy" in axes:
        for z in range(volume.shape[0]):
            if labels[z].max() == 0:
                continue
            imgs.append(to_cellpose_format(volume[z]))
            lbls.append(labels[z])

    if "xz" in axes:
        for y in range(0, volume.shape[1], stride):
            si = volume[:, y, :, :]             # (Z, X, C)
            sl = labels[:, y, :]                # (Z, X)
            if sl.max() == 0:
                continue
            si = zoom(si, (aniso, 1.0, 1.0), order=1)
            sl = zoom(sl, (aniso, 1.0),       order=0)
            imgs.append(to_cellpose_format(si.astype(np.float32)))
            lbls.append(sl.astype(np.int32))

    if "yz" in axes:
        for x in range(0, volume.shape[2], stride):
            si = volume[:, :, x, :]             # (Z, Y, C)
            sl = labels[:, :, x]                # (Z, Y)
            if sl.max() == 0:
                continue
            si = zoom(si, (aniso, 1.0, 1.0), order=1)
            sl = zoom(sl, (aniso, 1.0),       order=0)
            imgs.append(to_cellpose_format(si.astype(np.float32)))
            lbls.append(sl.astype(np.int32))

    return imgs, lbls


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_data(image_folder: str,
              label_folder: str,
              cfg: dict,
              split: str = "train"):

    image_paths = sorted(Path(image_folder).glob(cfg["IMAGE_GLOB"]))
    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {image_folder!r} "
            f"matching {cfg['IMAGE_GLOB']!r}"
        )

    # Volume-level split — keeps slices from the same volume together
    n_val = max(1, int(len(image_paths) * cfg["VAL_FRACTION"]))
    image_paths = (image_paths[:-n_val] if split == "train"
                   else image_paths[-n_val:])

    all_imgs, all_lbls = [], []
    skipped = 0

    for img_p in image_paths:
        lbl_name = img_p.name.replace(".ome.tif", cfg["LABEL_SUFFIX"])
        lbl_p = Path(label_folder) / lbl_name

        if not lbl_p.exists():
            print(f"  [SKIP] no label for {img_p.name}")
            skipped += 1
            continue

        img = tifffile.imread(str(img_p))
        lbl = tifffile.imread(str(lbl_p))

        img = fix_image_dims(img)
        lbl = fix_label_dims(lbl)

        if img.shape[0] != lbl.shape[0]:
            print(f"  [SKIP] Z mismatch img={img.shape} lbl={lbl.shape} "
                  f"in {img_p.name}")
            skipped += 1
            continue

        # Channel selection → must happen before normalisation
        img = select_channels(img, cfg["CHANNEL_IDX"], cfg["NUCLEUS_IDX"])

        # Per-slice per-channel normalisation
        for z in range(img.shape[0]):
            img[z] = normalize_slice(img[z])

        slc_i, slc_l = extract_slices(
            volume=img,
            labels=lbl,
            axes=cfg["AXES"],
            xy_voxel=cfg["XY_VOXEL_UM"],
            z_voxel=cfg["Z_VOXEL_UM"],
            stride=cfg["XZ_YZ_STRIDE"],
        )
        all_imgs.extend(slc_i)
        all_lbls.extend(slc_l)

    print(f"[{split.upper():5s}] {len(all_imgs):6d} slices  "
          f"from {len(image_paths)} volumes  (skipped: {skipped})")

    if not all_imgs:
        raise ValueError(
            f"No slices loaded for split='{split}'. "
            "Check IMAGE_FOLDER, LABEL_FOLDER, and LABEL_SUFFIX."
        )

    return all_imgs, all_lbls


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPER
# ─────────────────────────────────────────────────────────────────────────────
def predict_3d(volume_path: str,
               model_path: str,
               cfg: dict,
               gpu: bool = True) -> np.ndarray:
    """
    do_3D=True runs the multi-plane 2.5D inference strategy.
    This is the ONLY place do_3D belongs — it is not a training parameter.
    """
    mdl = models.CellposeModel(gpu=gpu, pretrained_model=model_path)
    vol = tifffile.imread(volume_path)

    # Channel list for eval — [cyto_ch, nuc_ch], 0 = grayscale
    ch = [1, 2] if cfg["NUCLEUS_IDX"] is not None else [0, 0]

    masks, _, _ = mdl.eval(
        vol,
        do_3D=True,
        anisotropy=cfg["Z_VOXEL_UM"] / cfg["XY_VOXEL_UM"],
        channels=ch,
        diameter=30,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        normalize=True,
    )
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    os.makedirs(CONFIG["SAVE_FOLDER"], exist_ok=True)

    use_gpu = CONFIG["USE_GPU"] and torch.cuda.is_available()
    print(f"GPU available : {torch.cuda.is_available()}")
    print(f"Using GPU     : {use_gpu}")
    print(f"Axis strategy : {CONFIG['AXES']}")
    print(f"Anisotropy    : {CONFIG['Z_VOXEL_UM'] / CONFIG['XY_VOXEL_UM']:.1f}x\n")

    # 1. Ensure model weights exist (downloads on first run)
    if CONFIG["DOWNLOAD_MODEL"]:
        CONFIG["MODEL_TYPE"] = ensure_model_downloaded(
            CONFIG["MODEL_TYPE"], use_gpu
        )

    # 2. Load data
    train_imgs, train_lbls = load_data(
        CONFIG["IMAGE_FOLDER"], CONFIG["LABEL_FOLDER"], CONFIG, split="train"
    )
    val_imgs, val_lbls = load_data(
        CONFIG["IMAGE_FOLDER"], CONFIG["LABEL_FOLDER"], CONFIG, split="val"
    )

    # 3. Build model
    model = models.CellposeModel(
        gpu=use_gpu,
        model_type=CONFIG["MODEL_TYPE"],    # ← model_type, not pretrained_model
    )

    # 4. Train
    # ── Parameters intentionally NOT passed to train_seg ─────────────────────
    #   channels    → handled in load_data via select_channels()
    #   channel_axis→ removed in Cellpose 3.x
    #   do_3D       → inference only, not a training parameter
    # ─────────────────────────────────────────────────────────────────────────
    print("Starting training ...")
    model_path, train_losses, test_losses = train.train_seg(
        model.net,

        train_data=train_imgs,
        train_labels=train_lbls,

        test_data=val_imgs,
        test_labels=val_lbls,

        normalize=False,                # already normalised in load_data

        learning_rate=CONFIG["LEARNING_RATE"],
        n_epochs=CONFIG["EPOCHS"],
        batch_size=CONFIG["BATCH_SIZE"],

        save_path=CONFIG["SAVE_FOLDER"],
        model_name=CONFIG["MODEL_NAME"],

        min_train_masks=1,
    )

    print(f"\nTraining complete.")
    print(f"  Model saved : {model_path}")
    print(f"  Train loss  : {train_losses[-1]:.4f}")
    if test_losses.size > 0:
        print(f"  Val loss    : {test_losses[-1]:.4f}")
    print(f"\nTo predict on a new volume:")
    print(f"  masks = predict_3d('your_volume.ome.tif', r'{model_path}', CONFIG)")