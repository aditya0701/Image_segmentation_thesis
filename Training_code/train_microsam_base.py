import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch_em
from torch_em.transform.augmentation import KorniaAugmentationPipeline
import kornia.augmentation as K
from torchvision.transforms import Compose
from micro_sam.training import default_sam_loader, train_sam

# ── Args from run_microsam_base.sh ────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--batch_size", type=int, default=1)
_parser.add_argument("--dataset", type=str, default="all",
                     choices=["all", "dog", "original", "psf"])
_parser.add_argument("--augmented", action="store_true",
                     help="Enable augmentations during training")
_args = _parser.parse_args()
BATCH_SIZE       = _args.batch_size
USE_AUGMENTATIONS = _args.augmented

DATASET_CONFIG = {
    "all":      "/home/students/rawat/training_data/training_data_All",
    "dog":      "/home/students/rawat/training_data/training_data_DOG",
    "original": "/home/students/rawat/training_data/training_data_Original",
    "psf":      "/home/students/rawat/training_data/training_data_PSF",
}

_aug_tag = "augmented" if USE_AUGMENTATIONS else "non_augmented"
_DATASET_LABEL  = {"all": "All", "dog": "DOG", "original": "Original", "psf": "PSF"}
_DATA_PATH      = DATASET_CONFIG[_args.dataset]
experiment_name = f"vit_b_lm_neo_{_aug_tag}_{_DATASET_LABEL[_args.dataset]}_v2"
SAVE_ROOT       = "/work/scratch/rawat/models/microsam_training_output_final_v2/"
MODEL_TYPE      = "vit_b_lm"
PATCH_SHAPE     = (1024, 1024)
N_EPOCHS        = 200
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE} | Batch size: {BATCH_SIZE}")
print(f"Dataset: {_args.dataset} | Augmented: {USE_AUGMENTATIONS}")
print(f"Experiment: {experiment_name}")

# ── Data Loading ───────────────────────────────────────────────────────────────
_img_dir     = os.path.join(_DATA_PATH, "train", "images")
_lbl_dir     = os.path.join(_DATA_PATH, "train", "masks")
_val_img_dir = os.path.join(_DATA_PATH, "validation", "images")
_val_lbl_dir = os.path.join(_DATA_PATH, "validation", "masks")

all_raw = sorted(glob.glob(os.path.join(_img_dir, "*.tif")))
all_lbl = sorted(glob.glob(os.path.join(_lbl_dir, "*.tif")))

if not all_raw:
    print(f"Error: No training images found at {_img_dir}")
    sys.exit(1)

if os.path.isdir(_val_img_dir):
    train_raw = all_raw
    train_lbl = all_lbl
    val_raw   = sorted(glob.glob(os.path.join(_val_img_dir, "*.tif")))
    val_lbl   = sorted(glob.glob(os.path.join(_val_lbl_dir, "*.tif")))
else:
    brain_ids     = [os.path.basename(p).split("_slice_")[0] for p in all_raw]
    unique_brains = sorted(set(brain_ids))
    n_val_brains  = max(1, round(len(unique_brains) * 0.15))
    val_brains    = set(unique_brains[-n_val_brains:])
    train_raw = [p for p, b in zip(all_raw, brain_ids) if b not in val_brains]
    train_lbl = [p for p, b in zip(all_lbl, brain_ids) if b not in val_brains]
    val_raw   = [p for p, b in zip(all_raw, brain_ids) if b in val_brains]
    val_lbl   = [p for p, b in zip(all_lbl, brain_ids) if b in val_brains]
    print(f"Auto-split: val brains = {sorted(val_brains)}")

print(f"Found {len(train_raw)} training slices.")
print(f"Found {len(val_raw)} validation slices.")

# ── Augmentations ──────────────────────────────────────────────────────────────
geometric_augmentations = KorniaAugmentationPipeline(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomRotation(degrees=90, p=0.5, resample='nearest'),
    K.RandomElasticTransform(p=0.3, resample='nearest'),
)

class ChannelSafeIntensityAug:
    def __call__(self, x):
        for c in range(2):
            if np.random.rand() > 0.5:
                x[c] = x[c] + np.random.randn(*x[c].shape) * 0.05
            if np.random.rand() > 0.5:
                x[c] = np.clip(x[c] * np.random.uniform(0.8, 1.2), 0, 1)
        return x

def scale_to_255(x):
    return x * 255

def clip_to_one(x):
    return np.clip(x, 0, 1)

# ── Data Loaders ───────────────────────────────────────────────────────────────
sampler = torch_em.data.sampler.MinForegroundSampler(min_fraction=0.001)

if USE_AUGMENTATIONS:
    print("Using custom augmentations.")
    intensity_and_scale_transform = Compose([ChannelSafeIntensityAug(), clip_to_one, scale_to_255])
    train_loader = default_sam_loader(
        raw_paths=train_raw, raw_key=None,
        label_paths=train_lbl, label_key=None,
        patch_shape=PATCH_SHAPE, batch_size=BATCH_SIZE,
        with_segmentation_decoder=True, shuffle=True, sampler=sampler,
        raw_transform=intensity_and_scale_transform,
        transform=geometric_augmentations,
    )
    val_loader = default_sam_loader(
        raw_paths=val_raw, raw_key=None,
        label_paths=val_lbl, label_key=None,
        patch_shape=PATCH_SHAPE, batch_size=BATCH_SIZE,
        with_segmentation_decoder=True, shuffle=False, sampler=sampler,
        raw_transform=scale_to_255,
    )
else:
    print("Using default transforms.")
    train_loader = default_sam_loader(
        raw_paths=train_raw, raw_key=None,
        label_paths=train_lbl, label_key=None,
        patch_shape=PATCH_SHAPE, batch_size=BATCH_SIZE,
        with_segmentation_decoder=True, shuffle=True, sampler=sampler,
    )
    val_loader = default_sam_loader(
        raw_paths=val_raw, raw_key=None,
        label_paths=val_lbl, label_key=None,
        patch_shape=PATCH_SHAPE, batch_size=BATCH_SIZE,
        with_segmentation_decoder=True, shuffle=False, sampler=sampler,
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
