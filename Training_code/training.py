import sys
import os
import torch
import numpy as np
import tifffile
from skimage import exposure
from sklearn.model_selection import train_test_split
from micro_sam.training import default_sam_loader, train_sam
from segment_anything import sam_model_registry

# --- 1. CONFIGURATION ---
# Get Batch Size from Cluster (default to 1 if not provided)
if len(sys.argv) > 1:
    BATCH_SIZE = int(sys.argv[1])
else:
    BATCH_SIZE = 1

BATCH_SIZE = 1
print(f"Running with BATCH_SIZE: {BATCH_SIZE}")

# PATHS (Ensure these exist on the cluster!)
DATA_PATH = r"/home/students/rawat/training_data/data/train/images/"
LABEL_PATH = r"/home/students/rawat/training_data/data/train/labels/"
CHECKPOINT_PATH = f"/work/scratch/rawat/image-segmentation-thesis/weight/sam_vit_h_4b8939.pth"
SAVE_ROOT = f"/work/scratch/rawat/microsam_training_output"

MODEL_TYPE = "vit_b_lm"
PATCH_SHAPE = (1024, 1024)
N_EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Output will be saved to: {SAVE_ROOT}")

## Data has already been sliced and saved to disk using images_2_slices.py

# Train/Val Split
x_train, x_val, y_train, y_val = train_test_split(x_slices, y_slices, test_size=0.1, random_state=42)

# --- 3. SAVE SLICES TO DISK (Required for DataLoader) ---
# We save to a temporary fast folder on the cluster node
def save_slices_to_disk(images, labels, root_dir, subset_name):
    img_dir = os.path.join(root_dir, "data", subset_name, "images")
    lbl_dir = os.path.join(root_dir, "data", subset_name, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    
    img_paths, lbl_paths = [], []
    for i, (img, lbl) in enumerate(zip(images, labels)):
        fname = f"slice_{i:03d}.tif"
        img_p = os.path.join(img_dir, fname)
        lbl_p = os.path.join(lbl_dir, fname)
        tifffile.imwrite(img_p, img)
        tifffile.imwrite(lbl_p, lbl)
        img_paths.append(img_p)
        lbl_paths.append(lbl_p)
    return img_paths, lbl_paths

print("Saving temporary training files on :" , SAVE_ROOT)
train_raw, train_lbl = save_slices_to_disk(x_train, y_train, SAVE_ROOT, "train")
val_raw, val_lbl = save_slices_to_disk(x_val, y_val, SAVE_ROOT, "val")

# --- 4. TRAIN ---
raw_key, label_key = "*.tif", "*.tif"
train_loader = default_sam_loader(raw_paths=train_raw, raw_key=None, label_paths=train_lbl, label_key=None, patch_shape=PATCH_SHAPE, batch_size=BATCH_SIZE, with_segmentation_decoder=True, shuffle=True)
val_loader = default_sam_loader(raw_paths=val_raw, raw_key=None, label_paths=val_lbl, label_key=None, patch_shape=PATCH_SHAPE, batch_size=BATCH_SIZE, with_segmentation_decoder=True, shuffle=False)

print("Starting Training...")
train_sam(
    name="best_bouton_model",
    model_type=MODEL_TYPE,
    train_loader=train_loader,
    val_loader=val_loader,
    n_epochs=N_EPOCHS,
    save_root=SAVE_ROOT,
    device=DEVICE,
)
print("Training Complete.")