"""
Cellpose 3D Training Script
============================
- io.logger_setup() prints per-epoch loss/LR to .err file automatically
- Full train_losses and test_losses arrays printed after training
  so you can see what happened to validation across all epochs
"""

import os
import argparse
import numpy as np
import tifffile
import torch
from pathlib import Path
from scipy.ndimage import zoom, label as cc_label
from cellpose import models, train, io

# Enables Cellpose's internal logger.
# Per-epoch output goes to .err file automatically — no custom loop needed.
io.logger_setup()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "DATA_ROOT":       r"/home/students/rawat/Final_dataset/all/",
    "SAVE_FOLDER":     r"/work/scratch/rawat/models/cellpose3d_final",

    "EPOCHS":          300,
    "EARLY_STOPPING":  40,
    "EVAL_EVERY":      10,
    "LEARNING_RATE":   1e-5,
    "BATCH_SIZE":      1,
    "MODEL_NAME":      "cellpose3d_calyx_all_final",

    "IMAGE_GLOB":      "*.tif",
    "IMAGE_POSTFIXES": ["_dog", "_deconv", "_100"],
    "LABEL_POSTFIX":   "_mask.tif",

    "CHANNEL_IDX":     0,
    "NUCLEUS_IDX":     1,

    "XY_VOXEL_UM":     0.0709,
    "Z_VOXEL_UM":      0.3,

    "AXES":            "xy",
    "XZ_YZ_STRIDE":    100,

    "USE_GPU":         True,
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fix_image_dims(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2: return img[np.newaxis, ..., np.newaxis]
    if img.ndim == 3: return img[..., np.newaxis]
    if img.ndim == 4:
        if int(np.argmin(img.shape)) == 0:
            img = np.transpose(img, (1, 2, 3, 0))
    return img

def ensure_zyxc(img: np.ndarray) -> np.ndarray:
    if img.ndim == 4 and img.shape[1] <= 4 and img.shape[1] < img.shape[2]:
        img = np.transpose(img, (0, 2, 3, 1))
        print(f"  [fix] Transposed from (Z,C,Y,X) → (Z,Y,X,C): {img.shape}")
    return img

def fix_label_dims(lbl: np.ndarray) -> np.ndarray:
    if lbl.ndim == 4: lbl = lbl.squeeze()
    if lbl.ndim == 2: lbl = lbl[np.newaxis]
    return lbl.astype(np.int32)

def select_channels(img: np.ndarray, channel_idx, nucleus_idx) -> np.ndarray:
    if img.ndim == 3: return img[..., np.newaxis]
    cyto = img.mean(axis=-1, keepdims=True) if channel_idx is None \
           else img[..., [channel_idx]]
    if nucleus_idx is not None and nucleus_idx < img.shape[-1]:
        return np.concatenate([cyto, img[..., [nucleus_idx]]], axis=-1)
    return cyto

def normalize_slice(sl: np.ndarray) -> np.ndarray:
    sl = sl.astype(np.float32)
    for c in range(sl.shape[-1]):
        lo, hi = np.percentile(sl[..., c], (0.00001, 99.99999))
        sl[..., c] = (sl[..., c] - lo) / (hi - lo) if hi > lo else 0.0
    return np.clip(sl, 0.0, 1.0)

def relabel_slice(sl):
    """Binarise and re-label a 2D slice so every disconnected region gets a
    unique integer ID.  Fixes two annotation artefacts in one step:
      - same label assigned to multiple disconnected boutons in one slice
      - binary-only masks where all boutons share label=1
    Cross-Z label consistency is irrelevant because Cellpose trains per-slice."""
    binary = (sl > 0).astype(np.uint8)
    relabeled, _ = cc_label(binary, structure=np.ones((3, 3)))
    return relabeled.astype(np.int32)


def extract_slices(volume, labels, axes, xy_voxel, z_voxel, stride):
    imgs, lbls = [], []
    aniso = z_voxel / xy_voxel

    def to_cellpose_format(sl_img):
        if sl_img.ndim == 2: return sl_img
        sl_img = np.transpose(sl_img, (2, 0, 1))
        if sl_img.shape[0] == 1: return sl_img.squeeze(0)
        return sl_img

    if "xy" in axes:
        for z in range(volume.shape[0]):
            if labels[z].max() == 0: continue
            imgs.append(to_cellpose_format(volume[z]))
            lbls.append(relabel_slice(labels[z]))

    if "xz" in axes:
        for y in range(0, volume.shape[1], stride):
            si, sl = volume[:, y, :, :], labels[:, y, :]
            if sl.max() == 0: continue
            si = zoom(si, (aniso, 1.0, 1.0), order=1)
            sl = zoom(sl, (aniso, 1.0), order=0)
            imgs.append(to_cellpose_format(si.astype(np.float32)))
            lbls.append(relabel_slice(sl.astype(np.int32)))

    if "yz" in axes:
        for x in range(0, volume.shape[2], stride):
            si, sl = volume[:, :, x, :], labels[:, :, x]
            if sl.max() == 0: continue
            si = zoom(si, (aniso, 1.0, 1.0), order=1)
            sl = zoom(sl, (aniso, 1.0), order=0)
            imgs.append(to_cellpose_format(si.astype(np.float32)))
            lbls.append(relabel_slice(sl.astype(np.int32)))

    return imgs, lbls


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_data(data_root: str, cfg: dict, split: str = "train"):
    image_folder = Path(data_root) / split / "images"
    label_folder = Path(data_root) / split / "masks"
    image_paths  = sorted(image_folder.glob(cfg["IMAGE_GLOB"]))
    
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_folder!r}")

    print(f"Found {len(image_paths)} {split} images in {image_folder}")

    all_imgs, all_lbls, skipped = [], [], 0

    for img_p in image_paths:
        base = img_p.name
        for px in cfg["IMAGE_POSTFIXES"]:
            if base.endswith(px + ".tif"):
                base = base.replace(px + ".tif", "")
                break
        if not any(img_p.name.endswith(p + ".tif") for p in cfg["IMAGE_POSTFIXES"]):
            base = img_p.name.replace(".tif", "")

        lbl_p = label_folder / (base + cfg["LABEL_POSTFIX"])
        if not lbl_p.exists():
            alt = label_folder / (img_p.stem + "_mask.tif")
            if alt.exists(): lbl_p = alt
            else:
                print(f"  [SKIP] no label for {img_p.name}")
                skipped += 1
                continue

        img = tifffile.imread(str(img_p))
        lbl = tifffile.imread(str(lbl_p))
        img = ensure_zyxc(fix_image_dims(img))
        lbl = fix_label_dims(lbl)

        if img.shape[0] != lbl.shape[0]:
            print(f"  [SKIP] Z mismatch in {img_p.name}")
            skipped += 1
            continue

        img = select_channels(img, cfg["CHANNEL_IDX"], cfg["NUCLEUS_IDX"])
        for z in range(img.shape[0]):
            img[z] = normalize_slice(img[z])

        si, sl = extract_slices(img, lbl, cfg["AXES"],
                                cfg["XY_VOXEL_UM"], cfg["Z_VOXEL_UM"],
                                cfg["XZ_YZ_STRIDE"])
        all_imgs.extend(si)
        all_lbls.extend(sl)

    print(f"[{split.upper():5s}] {len(all_imgs):6d} slices  "
          f"from {len(image_paths)-skipped} volumes  (skipped: {skipped})")

    if not all_imgs:
        raise ValueError(f"No slices loaded for split='{split}'.")
    return all_imgs, all_lbls


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPER
# ─────────────────────────────────────────────────────────────────────────────
def predict_3d(volume_path: str, model_path: str,
               cfg: dict, gpu: bool = True) -> np.ndarray:
    mdl = models.CellposeModel(gpu=gpu, pretrained_model=model_path)
    vol = tifffile.imread(volume_path)
    vol = ensure_zyxc(fix_image_dims(vol))
    vol = select_channels(vol, cfg["CHANNEL_IDX"], cfg["NUCLEUS_IDX"])
    masks, _, _ = mdl.eval(
        vol, do_3D=True,
        anisotropy=cfg["Z_VOXEL_UM"] / cfg["XY_VOXEL_UM"],
        channels=[1, 2], diameter=30, batch_size=1, tile=True,
    )
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOTS = {
    "all":      "/home/students/rawat/Final_dataset/all/",
    "dog":      "/home/students/rawat/Final_dataset/DOG/",
    "original": "/home/students/rawat/Final_dataset/Original/",
    "psf":      "/home/students/rawat/Final_dataset/PSF/",
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=list(DATASET_ROOTS.keys()),
                        help="Which image variant to train on")
    args = parser.parse_args()

    CONFIG["DATA_ROOT"]  = DATASET_ROOTS[args.dataset]
    CONFIG["MODEL_NAME"] = f"cellpose3d_calyx_{args.dataset}_final"

    os.makedirs(CONFIG["SAVE_FOLDER"], exist_ok=True)

    use_gpu = CONFIG["USE_GPU"] and torch.cuda.is_available()

    sep = "=" * 54
    print(sep)
    print(f"  EXPERIMENT       : {CONFIG['MODEL_NAME']}")
    print(f"  Dataset          : {args.dataset}")
    print(f"  Data root        : {CONFIG['DATA_ROOT']}")
    print(f"  Save folder      : {CONFIG['SAVE_FOLDER']}")
    print(f"  Pretrained       : cpsam (CellposeSAM)")
    print(f"  Axes             : {CONFIG['AXES']}")
    print(f"  Max epochs       : {CONFIG['EPOCHS']}")
    print(f"  Early stopping   : {CONFIG['EARLY_STOPPING']} epochs patience")
    print(f"  Eval every       : {CONFIG['EVAL_EVERY']} epochs")
    print(f"  LR               : {CONFIG['LEARNING_RATE']}")
    print(f"  Batch size       : {CONFIG['BATCH_SIZE']}")
    print(f"  Anisotropy       : {CONFIG['Z_VOXEL_UM'] / CONFIG['XY_VOXEL_UM']:.2f}x  "
          f"(Z={CONFIG['Z_VOXEL_UM']} XY={CONFIG['XY_VOXEL_UM']} µm)")
    print(f"  GPU avail        : {torch.cuda.is_available()}")
    print(f"  Using GPU        : {use_gpu}")
    print(sep + "\n")

    train_imgs, train_lbls = load_data(CONFIG["DATA_ROOT"], CONFIG, "train")
    val_imgs,   val_lbls   = load_data(CONFIG["DATA_ROOT"], CONFIG, "validation")

    # ── Training loop with early stopping ────────────────────────────────────
    # train_seg has no patience parameter, so we run in EVAL_EVERY-epoch chunks,
    # reload weights between chunks, and stop when val loss stops improving.
    model       = models.CellposeModel(gpu=use_gpu)
    best_val    = np.inf
    patience    = 0
    total_epoch = 0
    model_path  = None
    all_train   = []
    all_test    = []

    print("Starting training ...")
    while total_epoch < CONFIG["EPOCHS"]:
        chunk = min(CONFIG["EVAL_EVERY"], CONFIG["EPOCHS"] - total_epoch)

        model_path, tr, te = train.train_seg(
            model.net,
            train_data=train_imgs,
            train_labels=train_lbls,
            test_data=val_imgs,
            test_labels=val_lbls,
            normalize=False,
            learning_rate=CONFIG["LEARNING_RATE"],
            n_epochs=chunk,
            batch_size=CONFIG["BATCH_SIZE"],
            save_path=CONFIG["SAVE_FOLDER"],
            model_name=CONFIG["MODEL_NAME"],
            min_train_masks=1,
        )

        all_train.extend(tr.tolist())
        all_test.extend(te.tolist())
        total_epoch += chunk

        # val is only computed at certain epochs inside each chunk; skip zeros
        nonzero_val = te[te > 0]
        current_val = float(nonzero_val[-1]) if len(nonzero_val) > 0 else None

        if current_val is not None:
            improved = current_val < best_val - 1e-6
            if improved:
                best_val = current_val
                patience = 0
                status = "↓ best"
            else:
                patience += chunk
                status = f"no improvement ({patience}/{CONFIG['EARLY_STOPPING']})"
            print(f"  [epoch {total_epoch:>3d}]  train={float(tr[-1]):.4f}  "
                  f"val={current_val:.4f}  {status}")
            if patience >= CONFIG["EARLY_STOPPING"]:
                print(f"\nEarly stopping at epoch {total_epoch}. "
                      f"Best val loss: {best_val:.6f}")
                break
        else:
            print(f"  [epoch {total_epoch:>3d}]  train={float(tr[-1]):.4f}  "
                  f"val=n/a (no eval this chunk)")

        # reload saved weights for next chunk
        model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_path)

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\nTraining complete.  Total epochs: {total_epoch}")
    print(f"  Model saved : {model_path}")

    print(f"\n  train_losses — {len(all_train)} epochs:")
    for i, v in enumerate(all_train):
        print(f"    epoch {i:>3d} : {v:.6f}")

    nonzero_test = [(i, v) for i, v in enumerate(all_test) if v > 0]
    print(f"\n  test_losses (non-zero evaluations only) — {len(nonzero_test)} values:")
    if not nonzero_test:
        print("    EMPTY — Cellpose did not evaluate validation.")
    else:
        for i, v in nonzero_test:
            print(f"    epoch {i:>3d} : {v:.6f}")
        best_i, best_v = min(nonzero_test, key=lambda x: x[1])
        print(f"\n  Best train loss : {min(all_train):.6f}"
              f"  at epoch {all_train.index(min(all_train))}")
        print(f"  Best val loss   : {best_v:.6f}  at epoch {best_i}")