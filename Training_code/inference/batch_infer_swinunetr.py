"""
Batch SwinUNETR inference — all 4 datasets, matched checkpoints.
"""
import os, re
import numpy as np
import tifffile
import torch
from pathlib import Path
from scipy.ndimage import label as ndimage_label, generate_binary_structure
from skimage.morphology import remove_small_objects
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR

RUNS = [
    {
        "checkpoint": "/work/scratch/rawat/models/monai_swinunetr_final/swinunetr_calyx_all_final/swinunetr_calyx_all_final_best.pth",
        "input_folder": "/home/students/rawat/Final_dataset/all/test/images",
    },
    {
        "checkpoint": "/work/scratch/rawat/models/monai_swinunetr_final/swinunetr_calyx_dog_final/swinunetr_calyx_dog_final_best.pth",
        "input_folder": "/home/students/rawat/Final_dataset/DOG/test/images",
    },
    {
        "checkpoint": "/work/scratch/rawat/models/monai_swinunetr_final/swinunetr_calyx_original_final/swinunetr_calyx_original_final_best.pth",
        "input_folder": "/home/students/rawat/Final_dataset/Original/test/images",
    },
    {
        "checkpoint": "/work/scratch/rawat/models/monai_swinunetr_final/swinunetr_calyx_psf_final/swinunetr_calyx_psf_final_best.pth",
        "input_folder": "/home/students/rawat/Final_dataset/PSF/test/images",
    },
]

OUTPUT_BASE            = "/home/students/rawat/Results"
VALID_XY_SIZES         = {1100, 1101}
BLOB_REMOVAL_THRESHOLD = 1450
PATCH_SIZE             = (32, 192, 192)
SW_BATCH_SIZE          = 2
IN_CHANNELS            = 2
OUT_CHANNELS           = 2
FEATURE_SIZE           = 48
DEVICE                 = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_zcyx(arr):
    if arr.ndim == 3:
        return arr[:, np.newaxis]
    if arr.shape[0] <= 4 and arr.shape[0] < arr.shape[1]:
        arr = arr.transpose(1, 0, 2, 3)
    return arr


def percentile_normalize(img):
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[0]):
        ch = img[c]
        nonzero = ch[ch > 0]
        if nonzero.size == 0:
            out[c] = ch
            continue
        lo = float(np.percentile(nonzero, 0.00001))
        hi = float(np.percentile(nonzero, 99.99999))
        out[c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0) if hi > lo else ch
    return out


def preprocess(path):
    arr = tifffile.imread(str(path))
    arr = to_zcyx(arr)
    arr = arr.transpose(1, 0, 2, 3).astype(np.float32)
    arr = percentile_normalize(arr)
    return torch.from_numpy(arr).unsqueeze(0)


def load_model(checkpoint_path):
    model = SwinUNETR(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        feature_size=FEATURE_SIZE,
        use_checkpoint=False,
    ).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def run_inference(model, img_path, model_name):
    tensor = preprocess(img_path).to(DEVICE)
    print(f"  Input : {tuple(tensor.shape)}")
    with torch.no_grad():
        output = sliding_window_inference(
            tensor, roi_size=PATCH_SIZE, sw_batch_size=SW_BATCH_SIZE,
            predictor=model, overlap=0.5,
        )
    binary = output[0].argmax(dim=0).cpu().numpy() == 1
    structure = generate_binary_structure(3, 3)
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


if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")
    for run in RUNS:
        ckpt = run["checkpoint"]
        folder = run["input_folder"]
        m = re.search(r'/monai_swinunetr/([^/]+)/', ckpt)
        model_name = m.group(1) if m else Path(ckpt).parent.name
        print(f"\n{'='*60}")
        print(f"Model : {model_name}")
        print(f"Data  : {folder}")

        valid = [p for p in sorted(Path(folder).glob("*.tif"))
                 if tifffile.imread(str(p)).shape[-1] in VALID_XY_SIZES
                 and tifffile.imread(str(p)).shape[-2] in VALID_XY_SIZES]
        print(f"Images: {len(valid)} valid")

        model = load_model(ckpt)
        for img_path in valid:
            print(f"\n  {img_path.name}")
            run_inference(model, img_path, model_name)

        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("All SwinUNETR inference done.")
