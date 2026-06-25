"""
nnU-Net v2 Dataset Preparation Script
======================================
Converts TIFF fluorescence microscopy data into nnU-Net's required format:
  - Images  → NIfTI (.nii.gz), single channel, float32
  - Labels  → NIfTI (.nii.gz), binary uint8  (instance IDs → 0/1)
  - dataset.json   with channel names and label map
  - splits_final.json  preserving the existing train/validation split

Run this ONCE before nnU-Net preprocessing.
"""

import json
import re
import argparse
import numpy as np
import tifffile
import SimpleITK as sitk
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--dataset", type=str, required=True,
                     choices=["all", "dog", "original", "psf"])
_args = _parser.parse_args()

DATASET_MAP = {
    "all":      ("/home/students/rawat/Final_dataset/all",      1, "Calyx_All"),
    "dog":      ("/home/students/rawat/Final_dataset/DOG",      3, "Calyx_DOG"),
    "original": ("/home/students/rawat/Final_dataset/Original", 2, "Calyx_Original"),
    "psf":      ("/home/students/rawat/Final_dataset/PSF",      4, "Calyx_PSF"),
}
_data_root, _dataset_id, _dataset_name = DATASET_MAP[_args.dataset]

CONFIG = {
    "DATA_ROOT":      _data_root,
    "NNUNET_RAW":     "/work/scratch/rawat/nnunet/nnUNet_raw",
    "DATASET_ID":     _dataset_id,
    "DATASET_NAME":   _dataset_name,

    # Physical voxel size in µm (nnU-Net uses these for anisotropy planning)
    "XY_SPACING":     0.0709,
    "Z_SPACING":      0.300,
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def make_case_id(img_path: Path) -> str:
    """Derive a unique, nnU-Net-safe case identifier from the image filename.
    Preserves the full stem (including _dog, _deconv suffixes) so that
    multiple image types from the same volume get distinct case IDs."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", img_path.stem)


def build_pairs(data_root: str, split: str) -> list:
    img_dir = Path(data_root) / split / "images"
    lbl_dir = Path(data_root) / split / "masks"
    pairs = []
    for img_path in sorted(img_dir.glob("*.tif")):
        stem = img_path.stem
        for suffix in ("_dog", "_deconv"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        lbl_path = lbl_dir / f"{stem}_mask.tif"
        if not lbl_path.exists():
            lbl_path = lbl_dir / f"{img_path.stem}_mask.tif"
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
        else:
            print(f"  [WARN] No mask for {img_path.name} — skipping")
    return pairs


def to_zcyx(arr: np.ndarray) -> np.ndarray:
    """Normalise any 4D TIFF to (Z, C, Y, X) regardless of how it was saved.
    Detects (C, Z, Y, X) when the first dim is small (<=4) and smaller than Z."""
    if arr.ndim == 3:
        return arr[:, np.newaxis]                  # (Z, Y, X) → (Z, 1, Y, X)
    if arr.shape[0] <= 4 and arr.shape[0] < arr.shape[1]:
        arr = arr.transpose(1, 0, 2, 3)            # (C, Z, Y, X) → (Z, C, Y, X)
    return arr                                     # already (Z, C, Y, X)


def tiff_to_sitk_channel(path: Path, channel_idx: int,
                          xy_um: float, z_um: float) -> sitk.Image:
    """Load one channel from TIFF → (Z, Y, X) SimpleITK image."""
    arr = to_zcyx(tifffile.imread(str(path)))      # → (Z, C, Y, X)
    arr = arr[:, channel_idx]                      # → (Z, Y, X)
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing([xy_um, xy_um, z_um])
    return img


def tiff_to_sitk_label(path: Path, xy_um: float, z_um: float) -> sitk.Image:
    """Load TIFF instance label (Z, Y, X), binarise → SimpleITK image uint8."""
    arr = tifffile.imread(str(path))        # (Z, Y, X) uint32 instance IDs
    arr = (arr > 0).astype(np.uint8)        # binarise: any instance → 1
    lbl = sitk.GetImageFromArray(arr)
    lbl.SetSpacing([xy_um, xy_um, z_um])
    return lbl


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    xy = CONFIG["XY_SPACING"]
    z  = CONFIG["Z_SPACING"]

    dataset_dir = (
        Path(CONFIG["NNUNET_RAW"])
        / f"Dataset{CONFIG['DATASET_ID']:03d}_{CONFIG['DATASET_NAME']}"
    )
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    train_pairs = build_pairs(CONFIG["DATA_ROOT"], "train")
    val_pairs   = build_pairs(CONFIG["DATA_ROOT"], "validation")

    print(f"Train: {len(train_pairs)} volumes  |  Val: {len(val_pairs)} volumes")
    print(f"Anisotropy: Z/XY = {z/xy:.2f}x  (nnU-Net will plan around this)")

    train_ids, val_ids = [], []

    for split_pairs, id_list, tag in [
        (train_pairs, train_ids, "TRAIN"),
        (val_pairs,   val_ids,   "VAL  "),
    ]:
        for img_path, lbl_path in split_pairs:
            case_id = make_case_id(img_path)
            id_list.append(case_id)

            # nnU-Net naming: {case_id}_{modality:04d}.nii.gz — one file per channel
            sitk.WriteImage(tiff_to_sitk_channel(img_path, 0, xy, z),
                            str(images_tr / f"{case_id}_0000.nii.gz"))
            sitk.WriteImage(tiff_to_sitk_channel(img_path, 1, xy, z),
                            str(images_tr / f"{case_id}_0001.nii.gz"))

            sitk.WriteImage(tiff_to_sitk_label(lbl_path, xy, z),
                            str(labels_tr / f"{case_id}.nii.gz"))

            print(f"  [{tag}] {case_id}")

    # ── dataset.json ──────────────────────────────────────────────────────────
    dataset_json = {
        "channel_names": {"0": "AlexaFluor568_psf", "1": "AlexaFluor488_psf"},
        "labels":        {"background": 0, "bouton": 1},
        "numTraining":   len(train_pairs) + len(val_pairs),
        "file_ending":   ".nii.gz",
    }
    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"\ndataset.json  →  {dataset_dir / 'dataset.json'}")

    # ── splits_final.json  (preserves existing train / val split) ─────────────
    # nnU-Net does 5-fold cross-validation by default.
    # We override all 5 folds with the same pre-defined split so our ablation
    # uses exactly the same train/val cases as MicroSAM and CellposeSAM.
    splits = [{"train": train_ids, "val": val_ids}] * 5
    splits_path = dataset_dir / "splits_final.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"splits_final.json  →  {splits_path}")

    # ── Instructions ──────────────────────────────────────────────────────────
    d  = CONFIG["DATASET_ID"]
    nm = CONFIG["DATASET_NAME"]
    raw_root  = CONFIG["NNUNET_RAW"]
    prep_root = raw_root.replace("nnUNet_raw", "nnUNet_preprocessed")
    res_root  = raw_root.replace("nnUNet_raw", "nnUNet_results")

    print(f"""
Next steps
──────────
1. Install nnU-Net (once):
       pip install nnunetv2

2. Set environment variables (add to your .bashrc or Condor job):
       export nnUNet_raw={raw_root}
       export nnUNet_preprocessed={prep_root}
       export nnUNet_results={res_root}

3. Verify + preprocess (run once, ~5-10 min):
       nnUNetv2_plan_and_preprocess -d {d} --verify_dataset_integrity

4. Copy the splits file into the preprocessed directory:
       cp {splits_path} \\
          {prep_root}/Dataset{d:03d}_{nm}/splits_final.json

5. Train fold 0 (this is the job you submit to Condor):
       nnUNetv2_train {d} 3d_fullres 0

   Or submit via:
       condor_submit nnunet_train.sub
""")
