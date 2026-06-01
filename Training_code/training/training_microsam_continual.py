import os
import sys
import glob
import numpy as np
import torch
import torch_em
from sklearn.model_selection import GroupShuffleSplit
from torch_em.transform.augmentation import KorniaAugmentationPipeline
import kornia.augmentation as K
from torchvision.transforms import Compose
from micro_sam.training import default_sam_loader, train_sam

# ── Configuration ─────────────────────────────────────────────────────────────

BATCH_SIZE   = 1
PATCH_SHAPE  = (1024, 1024)
N_EPOCHS     = 200
DATA_PATH    = r"/home/students/rawat/training_data/training_data_Original/"
SAVE_ROOT    = r"/work/scratch/rawat/models/microsam_training_output_continual/"
MODEL_TYPE   = "vit_b_lm"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
USE_AUGMENTATIONS = True
experiment_name = "vit_b_lm_augmented_Original"
# Define paths for the pre-split data
TRAIN_IMG_DIR = os.path.join(DATA_PATH, "train", "images")
TRAIN_LBL_DIR = os.path.join(DATA_PATH, "train", "masks")
VAL_IMG_DIR   = os.path.join(DATA_PATH, "validation", "images")
VAL_LBL_DIR   = os.path.join(DATA_PATH, "validation", "masks")

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

# ── Data Loading (No Split Needed) ─────────────────────────────────────────────
# The data is already split into train and validation folders.

train_raw = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.tif")))
train_lbl = sorted(glob.glob(os.path.join(TRAIN_LBL_DIR, "*.tif")))
val_raw   = sorted(glob.glob(os.path.join(VAL_IMG_DIR, "*.tif")))
val_lbl   = sorted(glob.glob(os.path.join(VAL_LBL_DIR, "*.tif")))

# Sanity check to ensure we found the data
if not train_raw or not val_raw:
    print("Error: Could not find training or validation data. Check the following paths:")
    print(f"  - Training Images: {TRAIN_IMG_DIR}")
    print(f"  - Validation Images: {VAL_IMG_DIR}")
    sys.exit(1)

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
    checkpoint_path= r"/work/scratch/rawat/models/microsam_training_output/checkpoints/vit_b_lm_3channel_augmented/best.pt"
)