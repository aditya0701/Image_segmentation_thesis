"""
SwinUNETR 3D Training Script
Drosophila microglomeruli synaptic bouton segmentation
"""

import os
import argparse
import random
import numpy as np
import tifffile
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.data import Dataset, DataLoader, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureTyped,
    MapTransform,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    SpatialPadd,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Paths — DATA_ROOT and MODEL_NAME are set by --dataset argparse at runtime
    "DATA_ROOT":       "",
    "MODEL_NAME":      "",
    "SAVE_FOLDER":     "",

    # Training
    "EPOCHS":          300,
    "LEARNING_RATE":   1e-4,
    "BATCH_SIZE":      1,
    "NUM_WORKERS":     4,
    "VAL_INTERVAL":    5,    # validate every N epochs
    "CKPT_INTERVAL":   50,   # save checkpoint every N epochs

    # Patch size: Z × Y × X (near-isotropic in physical space)
    #   Z:  32 × 0.300  µm =  9.6 µm
    #   XY: 192 × 0.0709 µm = 13.6 µm
    "PATCH_SIZE":      (32, 192, 192),
    "NUM_SAMPLES":     4,    # random crops per volume per step
    "SW_BATCH_SIZE":   2,    # sliding-window sub-batch at inference

    # Model
    "IN_CHANNELS":     2,    # ch0=AlexaFluor568 (boutons), ch1=AlexaFluor488 (Kenyon cells)
    "OUT_CHANNELS":    2,    # background + foreground
    "FEATURE_SIZE":    48,

    # Foreground voxel fraction across the training set.
    # Leave as None to compute automatically at startup (adds ~30 s).
    # After computing once, hard-code the printed value here.
    "FG_FRACTION":     None,

    "SEED":            42,
    "USE_GPU":         True,
}


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    monai.utils.set_determinism(seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM LOADERS
# These handle the actual on-disk format:
#   images : (Z, C, Y, X) float32, varying intensity range
#   labels : (Z, Y, X)    uint32, instance IDs  (0 = background)
# ─────────────────────────────────────────────────────────────────────────────
def to_zcyx(arr: np.ndarray) -> np.ndarray:
    """Normalise any 4D TIFF to (Z, C, Y, X) regardless of how it was saved.
    Detects (C, Z, Y, X) when the first dim is small (<=4) and smaller than Z."""
    if arr.ndim == 3:
        return arr[:, np.newaxis]                  # (Z, Y, X) → (Z, 1, Y, X)
    if arr.shape[0] <= 4 and arr.shape[0] < arr.shape[1]:
        arr = arr.transpose(1, 0, 2, 3)            # (C, Z, Y, X) → (Z, C, Y, X)
    return arr                                     # already (Z, C, Y, X)


class LoadAllChannelsd(MapTransform):
    """Load TIFF image → (C, Z, Y, X) float32, handles (Z,C,Y,X) and (C,Z,Y,X)."""

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            arr = to_zcyx(tifffile.imread(d[key]))      # → (Z, C, Y, X)
            arr = arr.transpose(1, 0, 2, 3)             # → (C, Z, Y, X) for MONAI
            d[key] = arr.astype(np.float32)
        return d


class LoadBinaryLabeld(MapTransform):
    """Load TIFF instance label (Z, Y, X), binarise → (1, Z, Y, X) float32."""

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            arr = tifffile.imread(d[key])           # (Z, Y, X) uint32
            d[key] = (arr > 0).astype(np.float32)[np.newaxis]  # → (1, Z, Y, X)
        return d


class PercentileNormalized(MapTransform):
    """
    Percentile-based intensity normalisation to [0, 1].
    Clips to (plo, phi) percentiles computed over nonzero voxels only,
    then scales the result to [0, 1].  Matches the Cellpose pipeline.
    """

    def __init__(self, keys, plo: float = 0.00001, phi: float = 99.99999,
                 allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.plo = plo
        self.phi = phi

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]                            # (C, Z, Y, X) float32
            out = np.zeros_like(img)
            for c in range(img.shape[0]):           # normalise each channel independently
                ch      = img[c]
                nonzero = ch[ch > 0]
                if nonzero.size == 0:
                    out[c] = ch
                    continue
                lo = float(np.percentile(nonzero, self.plo))
                hi = float(np.percentile(nonzero, self.phi))
                out[c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0) if hi > lo else ch
            d[key] = out
        return d


# ─────────────────────────────────────────────────────────────────────────────
# DATASET VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def verify_dataset(file_list: list) -> None:
    """
    Print shape and intensity range for every channel of every file.
    Exits if any channel is entirely zero.
    """
    print(f"\n{'─'*70}")
    print(f"  Dataset verification  (2 channels)")
    print(f"{'─'*70}")

    errors  = 0
    z_sizes = []
    xy_sizes = []

    for item in file_list:
        img = to_zcyx(tifffile.imread(item["image"]))   # → (Z, C, Y, X)
        lbl = tifffile.imread(item["label"])

        z_sizes.append(img.shape[0])
        xy_sizes.append(img.shape[-2:])
        n_ch = img.shape[1] if img.ndim == 4 else 1

        ch_info = []
        for c in range(n_ch):
            ch  = img[:, c] if img.ndim == 4 else img
            cmax = float(ch.max())
            ch_info.append(f"ch{c}=[{float(ch.min()):.3f},{cmax:.3f}]")
            if cmax == 0.0:
                ch_info[-1] += " !!EMPTY!!"
                errors += 1

        fg_pct = int((lbl > 0).sum()) / lbl.size * 100
        print(
            f"  {Path(item['image']).name[-40:]:40s}"
            f"  {img.shape}  {'  '.join(ch_info)}  fg={fg_pct:.2f}%"
        )

    print(f"{'─'*70}")
    print(f"  Z slices : min={min(z_sizes)}, max={max(z_sizes)}")
    unique_xy = set(xy_sizes)
    if len(unique_xy) > 1:
        print(f"  [WARN] Mixed XY sizes: {unique_xy}  (fine — patches handle this)")
    else:
        print(f"  XY size  : {unique_xy.pop()} (consistent)")
    print(f"{'─'*70}\n")

    if errors > 0:
        raise RuntimeError(f"{errors} channel(s) are entirely zero across files.")


# ─────────────────────────────────────────────────────────────────────────────
# FILE LIST
# ─────────────────────────────────────────────────────────────────────────────
def build_file_list(data_root: str, split: str) -> list:
    """Return list of {"image": path, "label": path} dicts for MONAI Dataset."""
    img_dir = Path(data_root) / split / "images"
    lbl_dir = Path(data_root) / split / "masks"

    files = []
    for img_path in sorted(img_dir.glob("*.tif")):
        stem = img_path.stem
        for suffix in ("_dog", "_deconv", "_100"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break

        lbl_path = lbl_dir / f"{stem}_mask.tif"
        if not lbl_path.exists():
            # Fallback: try without stripping the suffix
            lbl_path = lbl_dir / f"{img_path.stem}_mask.tif"

        if lbl_path.exists():
            files.append({"image": str(img_path), "label": str(lbl_path)})
        else:
            print(f"  [WARN] No mask found for {img_path.name} — skipping")

    return files


# ─────────────────────────────────────────────────────────────────────────────
# FOREGROUND FRACTION  (used for class weights in DiceCELoss)
# ─────────────────────────────────────────────────────────────────────────────
def compute_fg_fraction(file_list: list) -> float:
    total, fg = 0, 0
    for item in file_list:
        mask = tifffile.imread(item["label"])
        total += mask.size
        fg    += int((mask > 0).sum())
    fraction = fg / total
    print(f"  FG fraction: {fraction:.6f}  (fg={fg:,}, total={total:,})")
    print(f"  → hard-code CONFIG['FG_FRACTION'] = {fraction:.6f} to skip this step next run")
    return float(fraction)


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
def get_transforms(split: str, cfg: dict) -> Compose:
    patch = cfg["PATCH_SIZE"]

    # CropForeground runs on the raw image (before normalisation)
    # so that background zeros are reliably detected.
    # PercentileNormalized clips each channel independently to the
    # 0.00001–99.99999 %ile of nonzero voxels then scales to [0, 1].
    shared = [
        LoadAllChannelsd(keys=["image"]),
        LoadBinaryLabeld(keys=["label"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        PercentileNormalized(keys=["image"]),
    ]

    if split == "train":
        return Compose(shared + [
            # Pad Z to at least PATCH_SIZE[0]; Y/X are left untouched (-1 = no pad)
            SpatialPadd(keys=["image", "label"], spatial_size=(patch[0], -1, -1)),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch,
                pos=3,           # 75 % foreground crops
                neg=1,
                num_samples=cfg["NUM_SAMPLES"],
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(1, 2)),
            RandGaussianNoised(keys=["image"], prob=0.5, std=0.05),
            RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.5),
            EnsureTyped(keys=["image", "label"]),
        ])
    else:
        # Full-volume validation — sliding-window inference handles patching
        return Compose(shared + [
            EnsureTyped(keys=["image", "label"]),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOTS = {
    "all":      "/home/students/rawat/Final_dataset/all/",
    "dog":      "/home/students/rawat/Final_dataset/DOG/",
    "original": "/home/students/rawat/Final_dataset/Original/",
    "psf":      "/home/students/rawat/Final_dataset/PSF/",
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=list(DATASET_ROOTS.keys()),
                        help="Which image variant to train on")
    args = parser.parse_args()

    CONFIG["DATA_ROOT"]   = DATASET_ROOTS[args.dataset]
    CONFIG["MODEL_NAME"]  = f"swinunetr_calyx_{args.dataset}_final"
    CONFIG["SAVE_FOLDER"] = f"/work/scratch/rawat/models/monai_swinunetr_final/{CONFIG['MODEL_NAME']}"

    set_seed(CONFIG["SEED"])

    device = torch.device("cuda" if CONFIG["USE_GPU"] and torch.cuda.is_available() else "cpu")

    sep = "=" * 54
    print(sep)
    print(f"  EXPERIMENT  : {CONFIG['MODEL_NAME']}")
    print(f"  Dataset     : {args.dataset}")
    print(f"  Data root   : {CONFIG['DATA_ROOT']}")
    print(f"  Save folder : {CONFIG['SAVE_FOLDER']}")
    print(f"  Pretrained  : model_swinvit.pt (MONAI SSL)")
    print(f"  Patch size  : {CONFIG['PATCH_SIZE']}")
    print(f"  Epochs      : {CONFIG['EPOCHS']}")
    print(f"  Val every   : {CONFIG['VAL_INTERVAL']}")
    print(f"  LR          : {CONFIG['LEARNING_RATE']}")
    print(f"  Batch size  : {CONFIG['BATCH_SIZE']}")
    print(f"  GPU avail   : {torch.cuda.is_available()}")
    print(f"  Using GPU   : {device}")
    print(sep + "\n")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    os.makedirs(CONFIG["SAVE_FOLDER"], exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(CONFIG["SAVE_FOLDER"], "tb_logs"))

    # ── File lists ────────────────────────────────────────────────────────────
    train_files = build_file_list(CONFIG["DATA_ROOT"], "train")
    val_files   = build_file_list(CONFIG["DATA_ROOT"], "validation")
    print(f"Train: {len(train_files)} volumes  |  Val: {len(val_files)} volumes")

    # ── Shape + channel verification (catches empty channels before training)
    verify_dataset(train_files + val_files)

    # ── Foreground fraction → CE class weights ────────────────────────────────
    if CONFIG["FG_FRACTION"] is None:
        print("Computing foreground fraction …")
        CONFIG["FG_FRACTION"] = compute_fg_fraction(train_files)

    fg = CONFIG["FG_FRACTION"]
    # Inverse-frequency weighting: rare foreground gets a higher weight
    ce_weight = torch.tensor([fg, 1.0 - fg], dtype=torch.float32).to(device)
    print(f"CE class weights — bg: {ce_weight[0]:.4f},  fg: {ce_weight[1]:.4f}")

    # ── Datasets & DataLoaders ────────────────────────────────────────────────
    train_ds = Dataset(data=train_files, transform=get_transforms("train", CONFIG))
    val_ds   = Dataset(data=val_files,   transform=get_transforms("val",   CONFIG))

    # list_data_collate is required because RandCropByPosNegLabeld returns
    # a list of NUM_SAMPLES dicts per volume rather than a single dict.
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        num_workers=CONFIG["NUM_WORKERS"],
        collate_fn=list_data_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(CONFIG["NUM_WORKERS"] > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=CONFIG["NUM_WORKERS"],
        collate_fn=list_data_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(CONFIG["NUM_WORKERS"] > 0),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SwinUNETR(
        in_channels=CONFIG["IN_CHANNELS"],
        out_channels=CONFIG["OUT_CHANNELS"],
        feature_size=CONFIG["FEATURE_SIZE"],
        use_checkpoint=True,    # gradient checkpointing — saves ~30 % VRAM
    ).to(device)

    # Uncomment to load MONAI SSL pretrained SwinViT weights (strongly recommended):
    weights = torch.load("/work/scratch/rawat/models/model_swinvit.pt", map_location="cpu")
    model.load_from(weights=weights)
    print("Loaded pretrained SwinViT weights")

    # ── Loss / optimiser / scheduler ──────────────────────────────────────────
    loss_fn = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        lambda_dice=2.0,
        lambda_ce=0.5,
        weight=ce_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["EPOCHS"]
    )
    scaler = torch.amp.GradScaler("cuda")

    # ── Post-processing for DiceMetric ────────────────────────────────────────
    post_pred  = AsDiscrete(argmax=True,  to_onehot=CONFIG["OUT_CHANNELS"])
    post_label = AsDiscrete(to_onehot=CONFIG["OUT_CHANNELS"])
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_metric       = -1.0
    best_metric_epoch = -1

    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        epoch_loss = 0.0
        n_steps    = max(len(train_loader), 1)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['EPOCHS']}"):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss    = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        epoch_loss /= n_steps
        scheduler.step()

        writer.add_scalar("train/loss", epoch_loss, epoch + 1)
        writer.add_scalar("train/lr",   scheduler.get_last_lr()[0], epoch + 1)
        print(
            f"  Epoch {epoch + 1:3d}  |  loss: {epoch_loss:.4f}"
            f"  |  lr: {scheduler.get_last_lr()[0]:.2e}"
        )

        # ── Validation ────────────────────────────────────────────────────────
        if (epoch + 1) % CONFIG["VAL_INTERVAL"] == 0:
            model.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)

                    val_outputs = sliding_window_inference(
                        val_inputs,
                        CONFIG["PATCH_SIZE"],
                        CONFIG["SW_BATCH_SIZE"],
                        model,
                        overlap=0.5,
                    )

                    val_outputs = [post_pred(i)  for i in decollate_batch(val_outputs)]
                    val_labels  = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()

            writer.add_scalar("val/dice", metric, epoch + 1)
            print(f"  Val Dice: {metric:.4f}")

            if metric > best_metric:
                best_metric       = metric
                best_metric_epoch = epoch + 1
                torch.save(
                    model.state_dict(),
                    os.path.join(CONFIG["SAVE_FOLDER"], f"{CONFIG['MODEL_NAME']}_best.pth"),
                )
                print(f"  Saved best model  (Dice = {best_metric:.4f})")

        # ── Periodic full checkpoint (resumes training) ───────────────────────
        if (epoch + 1) % CONFIG["CKPT_INTERVAL"] == 0:
            torch.save(
                {
                    "epoch":                epoch + 1,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_metric":          best_metric,
                    "config":               CONFIG,
                },
                os.path.join(
                    CONFIG["SAVE_FOLDER"],
                    f"{CONFIG['MODEL_NAME']}_ckpt_ep{epoch + 1:04d}.pth",
                ),
            )

    writer.close()
    print(
        f"\nTraining complete."
        f"  Best Val Dice: {best_metric:.4f} at epoch {best_metric_epoch}"
    )
