"""
MicroSAM Large inference — DOG dataset, v4 checkpoint.
Mirrors inference.ipynb exactly. Saves outside MicroSAM_Large folder.
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


CHECKPOINT_PATH        = "/work/scratch/rawat/models/microsam_training_output_final_v2/checkpoints/vit_l_lm_neo_augmented_DOG_v4/best.pt"
INPUT_FOLDER           = "/home/students/rawat/Final_dataset/DOG/test/images"
OUTPUT_DIR             = "/home/students/rawat/Final_results/MicroSAM_Large_DOG_v4"
MODEL_TYPE             = "vit_l_lm"
VALID_XY_SIZES         = {1100, 1101}
BLOB_REMOVAL_THRESHOLD = 862
DEVICE                 = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device     : {DEVICE}")
print(f"Checkpoint : {CHECKPOINT_PATH}")
print(f"Output     : {OUTPUT_DIR}\n")

valid_images = [
    p for p in sorted(Path(INPUT_FOLDER).glob("*.tif"))
    if tifffile.imread(str(p)).shape[-1] in VALID_XY_SIZES
    and tifffile.imread(str(p)).shape[-2] in VALID_XY_SIZES
]
print(f"Found {len(valid_images)} valid image(s)\n")

print("Loading model...")
predictor, segmenter = get_predictor_and_segmenter(
    model_type=MODEL_TYPE, checkpoint=CHECKPOINT_PATH,
    device=DEVICE, is_tiled=True,
)
print("Model ready.\n")

structure_3d = generate_binary_structure(3, 3)

for img_path in valid_images:
    print(f"{'─'*60}")
    print(f"Processing: {img_path.name}")

    raw_data = tifffile.imread(str(img_path))
    if raw_data.ndim == 4:
        if raw_data.shape[0] <= 4:
            raw_data = np.transpose(raw_data, (1, 2, 3, 0))
        elif raw_data.shape[1] <= 4:
            raw_data = np.transpose(raw_data, (0, 2, 3, 1))
    elif raw_data.ndim == 3:
        raw_data = raw_data[..., np.newaxis]

    low, high = np.percentile(raw_data, (0.00001, 99.9999))
    raw_data = np.clip((raw_data.astype(np.float32) - low) / (high - low), 0, 1) * 255 if high > low else np.zeros_like(raw_data, dtype=np.float32)

    Z, Y, X, C = raw_data.shape
    raw_rgb = np.zeros((Z, Y, X, 3), dtype=raw_data.dtype)
    raw_rgb[..., :min(C, 2)] = raw_data[..., :min(C, 2)]
    print(f"  Shape: {raw_rgb.shape}")

    stack = []
    for z in tqdm(range(Z), desc="  Slices", leave=False):
        pred = automatic_instance_segmentation(
            predictor=predictor, segmenter=segmenter,
            input_path=raw_rgb[z], ndim=2,
            tile_shape=(896, 896), halo=(64, 64),
        )
        stack.append(pred)

    pred3d = np.array(stack, dtype=np.uint32)
    labeled, n_obj = ndimage_label(pred3d > 0, structure=structure_3d)
    print(f"  3D CC  : {n_obj} objects")

    clean = remove_small_objects(labeled.astype(bool), min_size=BLOB_REMOVAL_THRESHOLD, connectivity=1)
    clean, _ = ndimage_label(clean, structure=structure_3d)
    for lbl in np.unique(clean)[1:]:
        if np.any(clean == lbl, axis=(1, 2)).sum() <= 3:
            clean[clean == lbl] = 0
    clean, n_final = ndimage_label(clean > 0, structure=structure_3d)
    print(f"  Final  : {n_final} objects  (blob>{BLOB_REMOVAL_THRESHOLD}, z-span>3)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dtype = np.uint32 if n_final > 65535 else np.uint16
    out = Path(OUTPUT_DIR) / f"{img_path.stem}_masks.tif"
    tifffile.imwrite(str(out), clean.astype(dtype))
    print(f"  Saved  → {out}")

print(f"\n{'='*60}")
print("Done.")
