"""
Batch Cellpose3D inference — xyz-trained models, do_3D=True.
Models trained with all three planes (XY/XZ/YZ).
"""
import os
import numpy as np
import tifffile
import torch
from pathlib import Path
from cellpose import models
from scipy.ndimage import label as ndimage_label, generate_binary_structure
from skimage.morphology import remove_small_objects

RUNS = [
    {
        "model_path":   "/work/scratch/rawat/models/cellpose3d_xyz_final/models/cellpose3d_calyx_all_xyz_final",
        "input_folder": "/home/students/rawat/Final_dataset/all/test/images",
    },
    {
        "model_path":   "/work/scratch/rawat/models/cellpose3d_xyz_final/models/cellpose3d_calyx_dog_xyz_final",
        "input_folder": "/home/students/rawat/Final_dataset/DOG/test/images",
    },
    {
        "model_path":   "/work/scratch/rawat/models/cellpose3d_xyz_final/models/cellpose3d_calyx_psf_xyz_final",
        "input_folder": "/home/students/rawat/Final_dataset/PSF/test/images",
    },
]

OUTPUT_BASE            = "/home/students/rawat/Final_results/CellposeSAM_xyz"
VALID_XY_SIZES         = {1100, 1101}
BLOB_REMOVAL_THRESHOLD = 1450
XY_VOXEL_UM            = 0.0709
Z_VOXEL_UM             = 0.3
DIAMETER               = 30
CHANNEL_IDX            = 0
NUCLEUS_IDX            = 1
USE_GPU                = torch.cuda.is_available()


def fix_image_dims(img):
    if img.ndim == 2: return img[np.newaxis, ..., np.newaxis]
    if img.ndim == 3: return img[..., np.newaxis]
    if img.ndim == 4 and int(np.argmin(img.shape)) == 0:
        img = np.transpose(img, (1, 2, 3, 0))
    return img


def ensure_zyxc(img):
    if img.ndim == 4 and img.shape[1] <= 4 and img.shape[1] < img.shape[2]:
        img = np.transpose(img, (0, 2, 3, 1))
        print(f"  [fix] (Z,C,Y,X) → (Z,Y,X,C): {img.shape}")
    return img


def normalize_slice(sl):
    sl = sl.astype(np.float32)
    for c in range(sl.shape[-1]):
        lo, hi = np.percentile(sl[..., c], (0.00001, 99.99999))
        sl[..., c] = (sl[..., c] - lo) / (hi - lo) if hi > lo else 0.0
    return np.clip(sl, 0.0, 1.0)


def preprocess_volume(path):
    vol = tifffile.imread(str(path))
    vol = ensure_zyxc(fix_image_dims(vol))
    cyto = vol[..., [CHANNEL_IDX]]
    if NUCLEUS_IDX is not None and NUCLEUS_IDX < vol.shape[-1]:
        vol = np.concatenate([cyto, vol[..., [NUCLEUS_IDX]]], axis=-1)
    else:
        vol = cyto
    for z in range(vol.shape[0]):
        vol[z] = normalize_slice(vol[z])
    return vol.astype(np.float32)


if __name__ == "__main__":
    print(f"GPU: {USE_GPU}\n")
    structure = generate_binary_structure(3, 3)

    for run in RUNS:
        model_path = run["model_path"]
        folder     = run["input_folder"]
        model_name = Path(model_path).name

        print(f"\n{'='*60}")
        print(f"Model : {model_name}")
        print(f"Data  : {folder}")

        valid = [p for p in sorted(Path(folder).glob("*.tif"))
                 if tifffile.imread(str(p)).shape[-1] in VALID_XY_SIZES
                 and tifffile.imread(str(p)).shape[-2] in VALID_XY_SIZES]
        print(f"Images: {len(valid)} valid")

        print("Loading model...")
        model = models.CellposeModel(gpu=USE_GPU, pretrained_model=model_path)
        print("Model ready.")

        for img_path in valid:
            print(f"\n  {img_path.name}")
            vol = preprocess_volume(img_path)
            print(f"  Shape : {vol.shape}  (Z,Y,X,C)")

            masks, _, _ = model.eval(
                vol, do_3D=True, z_axis=0, channel_axis=3,
                anisotropy=Z_VOXEL_UM / XY_VOXEL_UM,
                diameter=DIAMETER, normalize=False, batch_size=1,
            )
            clean = masks.copy()
            for lbl in np.unique(masks)[1:]:
                if (masks == lbl).sum() < BLOB_REMOVAL_THRESHOLD:
                    clean[clean == lbl] = 0
            clean, n_final = ndimage_label(clean > 0, structure=structure)
            print(f"  Raw   : {len(np.unique(masks)) - 1}  →  After blob removal: {n_final}")

            out_path = Path(OUTPUT_BASE) / model_name / f"{img_path.stem}_mask.tif"
            os.makedirs(out_path.parent, exist_ok=True)
            dtype = np.uint32 if n_final > 65535 else np.uint16
            tifffile.imwrite(str(out_path), clean.astype(dtype))
            print(f"  Saved → {out_path}")

        del model

    print(f"\n{'='*60}")
    print("All Cellpose3D xyz inference done.")
