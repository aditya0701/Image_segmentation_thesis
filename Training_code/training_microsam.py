import os
import sys
import glob
import numpy as np
import torch
import torch_em
from sklearn.model_selection import GroupShuffleSplit
from micro_sam.training import default_sam_loader, train_sam

# --- 1. CONFIGURATION ---
if len(sys.argv) > 1:
    BATCH_SIZE = int(sys.argv[1])
else:
    BATCH_SIZE = 1

print(f"Running with BATCH_SIZE: {BATCH_SIZE}")

PATCH_SHAPE = (1024,1024)
N_EPOCHS = 50
DATA_PATH = r"/home/students/rawat/training_data"
SAVE_ROOT = f"/work/scratch/rawat/models/microsam_training_output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPE = "vit_b_lm"

ALL_IMG_DIR = os.path.join(DATA_PATH, "data", "train", "images")
ALL_LBL_DIR = os.path.join(DATA_PATH, "data", "train", "labels")

# --- 2. INTELLIGENT GROUP SPLIT ---
print("Organizing files by 3D Stack ID...")

# Get all paths sorted
all_raw_paths = sorted(glob.glob(os.path.join(ALL_IMG_DIR, "*.tif")))
all_lbl_paths = sorted(glob.glob(os.path.join(ALL_LBL_DIR, "*.tif")))

groups = []
for p in all_raw_paths:
    filename = os.path.basename(p)
    # This splits "Brain1_slice_001.tif" into ["Brain1", "001.tif"]
    # We take the first part as the Group ID
    group_id = filename.split("_slice_")[0]
    groups.append(group_id)

groups = np.array(groups)
unique_brains = np.unique(groups)
# print all the groups found
print(f"Groups found: {unique_brains}")
print(f"Found {len(unique_brains)} unique 3D volumes: {unique_brains}")

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
sampler = torch_em.data.sampler.MinForegroundSampler(min_fraction=0.001)
# Perform the split
# keys are indices of the file list
train_idx, val_idx = next(splitter.split(all_raw_paths, groups=groups))

# Create the final lists
train_raw = [all_raw_paths[i] for i in train_idx]
train_lbl = [all_lbl_paths[i] for i in train_idx]
val_raw = [all_raw_paths[i] for i in val_idx]
val_lbl = [all_lbl_paths[i] for i in val_idx]

print(f"Split Complete (By Subject):")
print(f"Training:   {len(train_raw)} slices (from brains: {np.unique(groups[train_idx])})")
print(f"Validation: {len(val_raw)} slices (from brains: {np.unique(groups[val_idx])})")

# --- 3. CREATE LOADERS ---
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

# --- 4. START TRAINING ---
train_sam(
    name="vit_b_lm_patch1024_loss_ce",
    model_type=MODEL_TYPE,
    early_stopping = 40,
    train_loader=train_loader,
    val_loader=val_loader,
    n_epochs=N_EPOCHS,
    save_root=SAVE_ROOT,
    device=DEVICE,
)