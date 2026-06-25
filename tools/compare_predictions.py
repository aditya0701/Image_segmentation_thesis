"""
Sequential visual comparison: GT vs predicted masks for one model family.
Close the napari window to advance to the next model/image.

Change MODEL_FAMILY below to switch between:
  "CellposeSAM", "MicroSAM_Base", "MicroSAM_Large", "nnUNet", "SwinUNETR", CellposeSAM_2d
"""

import numpy as np
import tifffile
import napari
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_FAMILY  = "MicroSAM_Large"   # change this to switch family
RESULTS_ROOT  = Path(r"D:\Thesis_images\Updated_final_results")
DATASET_ROOT  = Path(r"D:\Thesis_images\Final_dataset")

GT_SUFFIXES   = ("_masks_PS.tif", "_Labels_PS.tif")
PRED_SUFFIXES = ("_masks.tif", "_mask.tif")
SKIP_SUFFIXES = ("_edges", "_inner_edges", "_shrunk", "_final", "_rough")
# ──────────────────────────────────────────────────────────────────────────────


def infer_dataset_type(folder_name: str) -> str | None:
    name = folder_name.lower()
    for dtype in ("all", "original", "dog", "psf"):
        if dtype in name:
            return dtype
    return None


def build_gt_index(dataset_type: str) -> dict[str, Path]:
    masks_dir = DATASET_ROOT / dataset_type / "test" / "masks"
    index = {}
    for f in masks_dir.glob("*.tif"):
        for suffix in GT_SUFFIXES:
            if f.name.endswith(suffix):
                index[f.name[: -len(suffix)]] = f
                break
    return index


def find_matching_gt(pred_path: Path, gt_index: dict[str, Path]) -> Path | None:
    name = pred_path.name
    pred_base = name
    for suffix in PRED_SUFFIXES:
        if name.endswith(suffix):
            pred_base = name[: -len(suffix)]
            break
    if pred_base in gt_index:
        return gt_index[pred_base]
    matches = [b for b in gt_index if pred_base.startswith(b)]
    if matches:
        return gt_index[max(matches, key=len)]
    return None


def find_matching_image(pred_base: str, dataset_type: str) -> Path | None:
    images_dir = DATASET_ROOT / dataset_type / "test" / "images"
    # Exact stem match first
    for f in images_dir.glob("*.tif"):
        if f.stem == pred_base:
            return f
    # Shortest stem that starts with pred_base (e.g. Brain2 original → .ome.tif)
    candidates = [f for f in images_dir.glob("*.tif") if f.stem.startswith(pred_base)]
    if candidates:
        return min(candidates, key=lambda f: len(f.stem))
    return None


def get_pred_base(pred_path: Path) -> str:
    name = pred_path.name
    for suffix in PRED_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return pred_path.stem


def load_image(path: Path) -> np.ndarray:
    raw = tifffile.imread(str(path))
    if raw.ndim == 4:
        if raw.shape[0] <= 4:
            raw = np.transpose(raw, (1, 2, 3, 0))   # (C,Z,Y,X) → (Z,Y,X,C)
        elif raw.shape[1] <= 4:
            raw = np.transpose(raw, (0, 2, 3, 1))   # (Z,C,Y,X) → (Z,Y,X,C)
    elif raw.ndim == 3:
        raw = raw[..., np.newaxis]                   # (Z,Y,X) → (Z,Y,X,1)
    lo, hi = np.percentile(raw, (0.00001, 99.9999))
    if hi > lo:
        return np.clip((raw.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return np.zeros_like(raw, dtype=np.float32)


def load_mask(path: Path) -> np.ndarray:
    arr = np.squeeze(tifffile.imread(str(path))).astype(np.int32)
    if arr.ndim == 4:
        arr = arr[0]
    return arr


def show_comparison(model_name: str, image_name: str,
                    img: np.ndarray, gt: np.ndarray, pred: np.ndarray):
    viewer = napari.Viewer(title=f"{model_name}  |  {image_name}")

    n_ch = img.shape[-1]
    colormaps = ["green", "red", "blue"][:n_ch]
    ch_names  = ["GFP", "mCherry", "Blue"][:n_ch]
    viewer.add_image(img, channel_axis=-1, colormap=colormaps, name=ch_names)

    viewer.add_labels(gt,   name="GT",        opacity=0.5)
    viewer.add_labels(pred, name="Predicted",  opacity=0.5)

    print(f"  -> Close the window to continue...")
    napari.run()


# ── COLLECT ALL (model, image) PAIRS ─────────────────────────────────────────
family_dir = RESULTS_ROOT / MODEL_FAMILY
if not family_dir.exists():
    raise SystemExit(f"Model family folder not found: {family_dir}")

entries = []   # list of (model_name, pred_path, gt_path, image_path)

for model_dir in sorted(family_dir.iterdir()):
    if not model_dir.is_dir():
        continue
    dataset_type = infer_dataset_type(model_dir.name)
    if dataset_type is None:
        continue
    gt_index = build_gt_index(dataset_type)

    for pred_path in sorted(model_dir.glob("*.tif")):
        if any(s in pred_path.name for s in SKIP_SUFFIXES):
            continue
        gt_path = find_matching_gt(pred_path, gt_index)
        if gt_path is None:
            print(f"[WARN] No GT for {pred_path.name}")
            continue
        pred_base = get_pred_base(pred_path)
        img_path = find_matching_image(pred_base, dataset_type)
        if img_path is None:
            print(f"[WARN] No image for {pred_path.name}")
            continue
        entries.append((model_dir.name, pred_path, gt_path, img_path))

if not entries:
    raise SystemExit("No entries found — check paths.")

print(f"\nFound {len(entries)} comparisons for {MODEL_FAMILY}:")
for i, (name, pred, gt, img) in enumerate(entries):
    print(f"  [{i+1:2d}] {name} | {pred.name}")

print("\nStarting viewer — close each window to advance.\n")

# ── SEQUENTIAL LOOP ───────────────────────────────────────────────────────────
for i, (model_name, pred_path, gt_path, img_path) in enumerate(entries):
    print(f"\n[{i+1}/{len(entries)}] {model_name}  |  {pred_path.name}")
    print(f"       GT    : {gt_path.name}")
    print(f"       Image : {img_path.name}")

    img  = load_image(img_path)
    gt   = load_mask(gt_path)
    pred = load_mask(pred_path)

    show_comparison(model_name, pred_path.stem, img, gt, pred)

print("\nAll comparisons done.")
