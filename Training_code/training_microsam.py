import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch_em
from sklearn.model_selection import GroupShuffleSplit
from torch_em.transform.augmentation import KorniaAugmentationPipeline
import kornia.augmentation as K
from torchvision.transforms import Compose
from micro_sam.training import default_sam_loader, train_sam

# ── Batch size and dataset from run.sh ────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--batch_size", type=int, default=1)
_parser.add_argument("--dataset", type=str, default="all",
                     choices=["all", "dog", "original", "psf"],
                     help="Which preprocessing dataset to train on")
_args = _parser.parse_args()
BATCH_SIZE = _args.batch_size

DATASET_CONFIG = {
    "all":      ("/home/students/rawat/training_data/training_data_All",      "vit_l_lm_neo_augmented_All_v5"),
    "dog":      ("/home/students/rawat/training_data/training_data_DOG",      "vit_l_lm_neo_augmented_DOG_v4"),
    "original": ("/home/students/rawat/training_data/training_data_Original", "vit_l_lm_neo_augmented_Original_v4"),
    "psf":      ("/home/students/rawat/training_data/training_data_PSF",      "vit_l_lm_neo_augmented_PSF_v3"),
}

# ── Configuration ─────────────────────────────────────────────────────────────
PATCH_SHAPE  = (1024, 1024)
N_EPOCHS     = 200
_DATA_PATH, experiment_name = DATASET_CONFIG[_args.dataset]
SAVE_ROOT    = r"/work/scratch/rawat/models/microsam_training_output_final_v2/"
MODEL_TYPE   = "vit_l_lm"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
USE_AUGMENTATIONS = True

print(f"Device: {DEVICE} | Batch size: {BATCH_SIZE}")
print(f"VIT base model with augmentations and pre-split data.")
# ── Augmentations ──────────────────────────────────────────────────────────────
# Geometric transforms applied jointly to image and label.
# Intensity transforms applied to channels 0 and 1 only;
# channel 2 is unused (black) and left untouched.

geometric_augmentations = KorniaAugmentationPipeline(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomRotation(degrees=90, p=0.5, resample='nearest'),
    K.RandomElasticTransform(p=0.3, resample='nearest'),
)

class ChannelSafeIntensityAug:
    def __call__(self, x):
        # This transform operates on numpy arrays
        for c in range(2):
            if np.random.rand() > 0.5:
                # Add gaussian noise
                x[c] = x[c] + np.random.randn(*x[c].shape) * 0.05
            if np.random.rand() > 0.5:
                # Adjust brightness
                factor = np.random.uniform(0.8, 1.2)
                x[c] = np.clip(x[c] * factor, 0, 1)
        return x

# ── Data Loading ───────────────────────────────────────────────────────────────
_img_dir = os.path.join(_DATA_PATH, "train", "images")
_lbl_dir = os.path.join(_DATA_PATH, "train", "masks")
_val_img_dir = os.path.join(_DATA_PATH, "validation", "images")
_val_lbl_dir = os.path.join(_DATA_PATH, "validation", "masks")

all_raw = sorted(glob.glob(os.path.join(_img_dir, "*.tif")))
all_lbl = sorted(glob.glob(os.path.join(_lbl_dir, "*.tif")))

if not all_raw:
    print(f"Error: No training images found at {_img_dir}")
    sys.exit(1)

if os.path.isdir(_val_img_dir):
    # Pre-split validation folder exists — use it directly
    train_raw = all_raw
    train_lbl = all_lbl
    val_raw   = sorted(glob.glob(os.path.join(_val_img_dir, "*.tif")))
    val_lbl   = sorted(glob.glob(os.path.join(_val_lbl_dir, "*.tif")))
else:
    # No validation folder — split by brain volume to avoid slice leakage
    # Extract brain prefix from filename (everything before "_slice_")
    brain_ids = [os.path.basename(p).split("_slice_")[0] for p in all_raw]
    unique_brains = sorted(set(brain_ids))
    n_val_brains  = max(1, round(len(unique_brains) * 0.15))
    val_brains    = set(unique_brains[-n_val_brains:])  # last N brains → val
    train_raw = [p for p, b in zip(all_raw, brain_ids) if b not in val_brains]
    train_lbl = [p for p, b in zip(all_lbl, brain_ids) if b not in val_brains]
    val_raw   = [p for p, b in zip(all_raw, brain_ids) if b in val_brains]
    val_lbl   = [p for p, b in zip(all_lbl, brain_ids) if b in val_brains]
    print(f"Auto-split: val brains = {sorted(val_brains)}")

print(f"Found {len(train_raw)} training slices.")
print(f"Found {len(val_raw)} validation slices.")

# ── Data Loaders ───────────────────────────────────────────────────────────────

sampler = torch_em.data.sampler.MinForegroundSampler(min_fraction=0.001)

if USE_AUGMENTATIONS:
    print("Using custom augmentations.")
    # The 'raw_transform' is now a composition of intensity augmentations AND the final scaling.
    # This is necessary because providing a custom transform overrides the loader's default logic.
    def clip_to_one(x):
        return np.clip(x, 0, 1)

    def scale_to_255(x):
        return x * 255

    intensity_and_scale_transform = Compose([
        ChannelSafeIntensityAug(),
        clip_to_one,      # Clip back to [0, 1] after augmentations
        scale_to_255                # Scale from [0, 1] to [0, 255]
    ])

    train_loader = default_sam_loader(
        raw_paths=train_raw,
        raw_key=None,
        label_paths=train_lbl,
        label_key=None,
        patch_shape=PATCH_SHAPE,
        batch_size=BATCH_SIZE,
        with_segmentation_decoder=True,
        shuffle=True,
        sampler=sampler,
        raw_transform=intensity_and_scale_transform,
        transform=geometric_augmentations,
    )

    val_loader = default_sam_loader(
        raw_paths=val_raw,
        raw_key=None,
        label_paths=val_lbl,
        label_key=None,
        patch_shape=PATCH_SHAPE,
        batch_size=BATCH_SIZE,
        with_segmentation_decoder=True,
        shuffle=False,
        sampler=sampler,
        raw_transform=scale_to_255  # Also scale validation data
    )
else:
    print("Using default transforms.")
    train_loader = default_sam_loader(
        raw_paths=train_raw,
        raw_key=None,
        label_paths=train_lbl,
        label_key=None,
        patch_shape=PATCH_SHAPE,
        batch_size=BATCH_SIZE,
        with_segmentation_decoder=True,
        shuffle=True,
        sampler=sampler
    )

    val_loader = default_sam_loader(
        raw_paths=val_raw,
        raw_key=None,
        label_paths=val_lbl,
        label_key=None,
        patch_shape=PATCH_SHAPE,
        batch_size=BATCH_SIZE,
        with_segmentation_decoder=True,
        shuffle=False,
        sampler=sampler
    )

# ── Training ───────────────────────────────────────────────────────────────────

print(f"Starting training for experiment: {experiment_name}")

train_sam(
    name=experiment_name,
    model_type=MODEL_TYPE,
    early_stopping=40,
    train_loader=train_loader,
    val_loader=val_loader,
    n_epochs=N_EPOCHS,
    save_root=SAVE_ROOT,
    device=DEVICE,
)