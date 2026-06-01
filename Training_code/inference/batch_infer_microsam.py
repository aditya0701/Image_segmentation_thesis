"""
Batch MicroSAM inference — all 4 datasets, matched checkpoints (vit_l_lm_neo_augmented).
"""
import os
import torch
import tifffile
import numpy as np
from pathlib import Path
from tqdm import tqdm
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
from scipy.ndimage import label as ndimage_label, generate_binary_structure
from skimage.morphology import remove_small_objects


class ChannelSafeIntensityAug:
    def __call__(self, x):
        for c in range(2):
            if np.random.rand() > 0.5:
                x[c] = x[c] + np.random.randn(*x[c].shape) * 0.05
            if np.random.rand() > 0.5:
                x[c] = np.clip(x[c] * np.random.uniform(0.8, 1.2), 0, 1)
        return x


RUNS = [
    {
        "checkpoint": "/work/scratch/rawat/models/microsam_training_output_final_v2/checkpoints/vit_l_lm_neo_augmented_All_v3/best.pt",
        "input_folder": "/home/students/rawat/Final_dataset/all/test/images",
    },
    {
        "checkpoint": "/work/scratch/rawat/models/microsam_training_output_final_v2/checkpoints/vit_l_lm_neo_augmented_DOG_v3/best.pt",
        "input_folder": "/home/students/rawat/Final_dataset/DOG/test/images",
    },
    {
        "checkpoint": "/work/scratch/rawat/models/microsam_training_output_final_v2/checkpoints/vit_l_lm_neo_augmented_Original_v3/best.pt",
        "input_folder": "/home/students/rawat/Final_dataset/Original/test/images",
    },
    {
        "checkpoint": "/work/scratch/rawat/models/microsam_training_output_final_v2/checkpoints/vit_l_lm_neo_augmented_PSF_v3/best.pt",
        "input_folder": "/home/students/rawat/Final_dataset/PSF/test/images",
    },
]

OUTPUT_BASE            = "/home/students/rawat/Results"
VALID_XY_SIZES         = {1100, 1101}
BLOB_REMOVAL_THRESHOLD = 862
TILE_SHAPE             = (896, 896)
HALO                   = (64, 64)
MODEL_TYPE             = "vit_l_lm"
DEVICE                 = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess(raw_data):
    if raw_data.ndim == 4:
        if raw_data.shape[0] <= 4:
            raw_data = np.transpose(raw_data, (1, 2, 3, 0))
        elif raw_data.shape[1] <= 4:
            raw_data = np.transpose(raw_data, (0, 2, 3, 1))
    elif raw_data.ndim == 3:
        raw_data = raw_data[..., np.newaxis]
    low, high = np.percentile(raw_data, (0.00001, 99.9999))
    if high > low:
        raw_data = np.clip((raw_data.astype(np.float32) - low) / (high - low), 0, 1) * 255
    else:
        raw_data = np.zeros_like(raw_data, dtype=np.float32)
    Z, Y, X, C = raw_data.shape
    rgb = np.zeros((Z, Y, X, 3), dtype=raw_data.dtype)
    rgb[..., :min(C, 2)] = raw_data[..., :min(C, 2)]
    return rgb


if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")
    structure = generate_binary_structure(3, 3)

    for run in RUNS:
        ckpt   = run["checkpoint"]
        folder = run["input_folder"]
        model_name = Path(ckpt).parent.name

        print(f"\n{'='*60}")
        print(f"Model : {model_name}")
        print(f"Data  : {folder}")

        valid = [p for p in sorted(Path(folder).glob("*.tif"))
                 if tifffile.imread(str(p)).shape[-1] in VALID_XY_SIZES
                 and tifffile.imread(str(p)).shape[-2] in VALID_XY_SIZES]
        print(f"Images: {len(valid)} valid")

        print("Loading model...")
        predictor, segmenter = get_predictor_and_segmenter(
            model_type=MODEL_TYPE, checkpoint=ckpt, device=DEVICE, is_tiled=True,
        )
        print("Model ready.")

        for img_path in valid:
            print(f"\n  {img_path.name}")
            raw = tifffile.imread(str(img_path))
            rgb = preprocess(raw)
            print(f"  Shape : {rgb.shape}  (Z,Y,X,3)")

            stack = []
            for z in tqdm(range(rgb.shape[0]), desc="  Slices", leave=False):
                pred = automatic_instance_segmentation(
                    predictor=predictor, segmenter=segmenter,
                    input_path=rgb[z], ndim=2,
                    tile_shape=TILE_SHAPE, halo=HALO,
                )
                stack.append(pred)

            pred3d = np.array(stack, dtype=np.uint32)
            labeled, n_obj = ndimage_label(pred3d > 0, structure=structure)
            print(f"  3D CC : {n_obj} objects")

            clean = remove_small_objects(labeled.astype(bool), min_size=BLOB_REMOVAL_THRESHOLD, connectivity=1)
            # z-span filter: remove objects spanning ≤3 z-slices
            _, clean = ndimage_label(clean, structure=structure)
            clean, _ = ndimage_label(clean > 0, structure=structure)
            for lbl in np.unique(clean)[1:]:
                if np.any(clean == lbl, axis=(1, 2)).sum() <= 3:
                    clean[clean == lbl] = 0
            clean, n_final = ndimage_label(clean > 0, structure=structure)
            print(f"  Final : {n_final} objects  (blob={BLOB_REMOVAL_THRESHOLD}, z-span>3)")

            out_path = Path(OUTPUT_BASE) / model_name / f"{img_path.stem}_masks.tif"
            os.makedirs(out_path.parent, exist_ok=True)
            dtype = np.uint32 if n_final > 65535 else np.uint16
            tifffile.imwrite(str(out_path), clean.astype(dtype))
            print(f"  Saved → {out_path}")

        del predictor, segmenter
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("All MicroSAM inference done.")
