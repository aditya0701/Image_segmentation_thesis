"""
Batch nnUNet inference — all 4 datasets, matched model folders.
Run with bw_env:
  /work/scratch/rawat/installations/conda/envs/bw_env/bin/python -u batch_infer_nnunet.py
"""
import os, shutil, tempfile
import numpy as np
import tifffile
import nibabel as nib
import torch
from pathlib import Path
from scipy.ndimage import label as ndimage_label, generate_binary_structure
from skimage.morphology import remove_small_objects

os.environ["nnUNet_raw"]          = "/work/scratch/rawat/nnunet/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/work/scratch/rawat/nnunet/nnUNet_preprocessed"
os.environ["nnUNet_results"]      = "/work/scratch/rawat/models/nnunet"

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

RUNS = [
    {
        "model_folder": "/work/scratch/rawat/models/nnunet/Dataset001_Calyx_All/nnUNetTrainer__nnUNetPlans__3d_fullres",
        "input_folder": "/home/students/rawat/Final_dataset/all/test/images",
    },
    {
        "model_folder": "/work/scratch/rawat/models/nnunet/Dataset003_Calyx_DOG/nnUNetTrainer__nnUNetPlans__3d_fullres",
        "input_folder": "/home/students/rawat/Final_dataset/DOG/test/images",
    },
    {
        "model_folder": "/work/scratch/rawat/models/nnunet/Dataset002_Calyx_Original/nnUNetTrainer__nnUNetPlans__3d_fullres",
        "input_folder": "/home/students/rawat/Final_dataset/Original/test/images",
    },
    {
        "model_folder": "/work/scratch/rawat/models/nnunet/Dataset004_Calyx_PSF/nnUNetTrainer__nnUNetPlans__3d_fullres",
        "input_folder": "/home/students/rawat/Final_dataset/PSF/test/images",
    },
]

OUTPUT_BASE            = "/home/students/rawat/Results"
VALID_XY_SIZES         = {1100, 1101}
BLOB_REMOVAL_THRESHOLD = 1450
FOLD                   = 0
CHECKPOINT             = "checkpoint_best.pth"
Z_VOXEL_UM             = 0.3
XY_VOXEL_UM            = 0.0709
DEVICE                 = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_zcyx(arr):
    if arr.ndim == 3:
        return arr[:, np.newaxis]
    if arr.shape[0] <= 4 and arr.shape[0] < arr.shape[1]:
        arr = arr.transpose(1, 0, 2, 3)
        print(f"  [fix] (C,Z,Y,X) → (Z,C,Y,X): {arr.shape}")
    return arr


def tiff_to_nifti(tiff_path, out_dir):
    arr = to_zcyx(tifffile.imread(str(tiff_path)))
    affine = np.diag([Z_VOXEL_UM, XY_VOXEL_UM, XY_VOXEL_UM, 1.0])
    stem = tiff_path.stem
    paths = []
    for c in range(arr.shape[1]):
        nii = nib.Nifti1Image(arr[:, c].astype(np.float32), affine)
        p = out_dir / f"{stem}_{c:04d}.nii.gz"
        nib.save(nii, str(p))
        paths.append(str(p))
    return paths


if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")
    structure = generate_binary_structure(3, 3)

    for run in RUNS:
        model_folder = run["model_folder"]
        folder       = run["input_folder"]
        model_name   = Path(model_folder).parent.name

        print(f"\n{'='*60}")
        print(f"Model : {model_name}")
        print(f"Data  : {folder}")

        valid = [p for p in sorted(Path(folder).glob("*.tif"))
                 if tifffile.imread(str(p)).shape[-1] in VALID_XY_SIZES
                 and tifffile.imread(str(p)).shape[-2] in VALID_XY_SIZES]
        print(f"Images: {len(valid)} valid")

        print("Loading predictor...")
        predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
            perform_everything_on_device=True, device=DEVICE, verbose=False,
        )
        predictor.initialize_from_trained_model_folder(
            model_folder, use_folds=(FOLD,), checkpoint_name=CHECKPOINT,
        )
        print("Predictor ready.")

        for img_path in valid:
            print(f"\n  {img_path.name}")
            tmp_in  = Path(tempfile.mkdtemp(prefix="nnunet_in_"))
            tmp_out = Path(tempfile.mkdtemp(prefix="nnunet_out_"))
            try:
                channel_files = tiff_to_nifti(img_path, tmp_in)
                print(f"  NIfTI : {[Path(f).name for f in channel_files]}")
                predictor.predict_from_files(
                    [channel_files], str(tmp_out),
                    save_probabilities=False, overwrite=True,
                    num_processes_preprocessing=1, num_processes_segmentation_export=1,
                )
                pred_nii = next(tmp_out.glob("*.nii.gz"))
                binary = nib.load(str(pred_nii)).get_fdata().astype(bool)
                labeled, n_obj = ndimage_label(binary, structure=structure)
                print(f"  3D CC : {n_obj} objects")
                clean = remove_small_objects(labeled.astype(bool), min_size=BLOB_REMOVAL_THRESHOLD, connectivity=1)
                clean, n_final = ndimage_label(clean, structure=structure)
                print(f"  Final : {n_final} objects  (blob threshold={BLOB_REMOVAL_THRESHOLD})")
                out_path = Path(OUTPUT_BASE) / model_name / f"{img_path.stem}_mask.tif"
                os.makedirs(out_path.parent, exist_ok=True)
                dtype = np.uint32 if n_final > 65535 else np.uint16
                tifffile.imwrite(str(out_path), clean.astype(dtype))
                print(f"  Saved → {out_path}")
            finally:
                shutil.rmtree(tmp_in,  ignore_errors=True)
                shutil.rmtree(tmp_out, ignore_errors=True)

        del predictor
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("All nnUNet inference done.")
