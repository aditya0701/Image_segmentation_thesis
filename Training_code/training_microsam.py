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

BATCH_SIZE   = int(sys.argv[1]) if len(sys.argv) > 1 else 1
PATCH_SHAPE  = (1024, 1024)
N_EPOCHS     = 50
DATA_PATH    = r"/home/students/rawat/training_data_new"
SAVE_ROOT    = "/work/scratch/rawat/models/microsam_training_output"
MODEL_TYPE   = "vit_b_lm"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

ALL_IMG_DIR  = os.path.join(DATA_PATH, "data", "train", "images")
ALL_LBL_DIR  = os.path.join(DATA_PATH, "data", "train", "labels")

print(f"Device: {DEVICE} | Batch size: {BATCH_SIZE}")
print(f"VIT Base model with augmentaionts and new data in RGB format")
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

# ── Data Split ─────────────────────────────────────────────────────────────────
# Group-aware split: all slices from the same brain volume stay in one partition.

all_raw_paths = sorted(glob.glob(os.path.join(ALL_IMG_DIR, "*.tif")))
all_lbl_paths = sorted(glob.glob(os.path.join(ALL_LBL_DIR, "*.tif")))

groups = np.array([os.path.basename(p).split("_slice_")[0] for p in all_raw_paths])

print(f"Volumes found: {np.unique(groups)}")

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(all_raw_paths, groups=groups))

train_raw = [all_raw_paths[i] for i in train_idx]
train_lbl = [all_lbl_paths[i] for i in train_idx]
val_raw   = [all_raw_paths[i] for i in val_idx]
val_lbl   = [all_lbl_paths[i] for i in val_idx]

print(f"Train: {len(train_raw)} slices {np.unique(groups[train_idx])}")
print(f"Val:   {len(val_raw)} slices {np.unique(groups[val_idx])}")

# ── Data Loaders ───────────────────────────────────────────────────────────────

sampler = torch_em.data.sampler.MinForegroundSampler(min_fraction=0.001)

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

# ── Training ───────────────────────────────────────────────────────────────────

train_sam(
    name="vit_b_lm_3channel_augmented",
    model_type=MODEL_TYPE,
    early_stopping=40,
    train_loader=train_loader,
    val_loader=val_loader,
    n_epochs=N_EPOCHS,
    save_root=SAVE_ROOT,
    device=DEVICE,
)