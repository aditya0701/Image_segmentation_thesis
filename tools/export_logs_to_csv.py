"""
Export training logs to a single tidy CSV:  model, run, step, metric, value

Sources:
  - MicroSAM / SwinUNETR : TensorBoard event files (scalars only)
  - nnUNet               : training_log_*.txt files (train_loss, val_loss, pseudo_dice)

Usage:
    python tools/export_logs_to_csv.py
"""

import glob
import os
import re
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ============================================================
MODELS_ROOT = "/work/scratch/rawat/models"
OUTPUT_CSV  = "/work/scratch/rawat/image-segmentation-thesis/tools/training_logs.csv"

TFEVENTS_SOURCES = {
    "microsam": {
        "root":      f"{MODELS_ROOT}/microsam_training_output_final_v2/logs",
        "tb_subdir": "",        # event files sit directly in the run dir
    },
    "swinunetr": {
        "root":      f"{MODELS_ROOT}/monai_swinunetr_final",
        "tb_subdir": "tb_logs", # event files are one level deeper
    },
}

NNUNET_ROOT = f"{MODELS_ROOT}/nnunet"
# ============================================================


def read_tfevents(event_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(event_dir, "events.out.tfevents*"))
    if not files:
        return pd.DataFrame()
    rows = []
    for path in files:
        ea = EventAccumulator(path)
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            for event in ea.Scalars(tag):
                rows.append({"step": event.step, "metric": tag, "value": event.value})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # keep last value when a step appears in multiple files (job restarts)
    df = df.sort_values("step").drop_duplicates(subset=["step", "metric"], keep="last")
    return df


def read_nnunet_txt(fold_dir: str) -> pd.DataFrame:
    """Parse nnUNet training_log_*.txt files into a tidy DataFrame."""
    log_files = sorted(glob.glob(os.path.join(fold_dir, "training_log_*.txt")))
    if not log_files:
        return pd.DataFrame()

    epoch, train_loss, val_loss, pseudo_dice = None, None, None, None
    rows = []

    for path in log_files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                m = re.search(r"Epoch (\d+)", line)
                if m:
                    epoch = int(m.group(1))
                    train_loss = val_loss = pseudo_dice = None
                    continue
                if epoch is None:
                    continue
                m = re.search(r"train_loss\s+([-\d.]+)", line)
                if m:
                    train_loss = float(m.group(1))
                m = re.search(r"val_loss\s+([-\d.]+)", line)
                if m:
                    val_loss = float(m.group(1))
                m = re.search(r"Pseudo dice \[([\d.]+)\]", line)
                if m:
                    pseudo_dice = float(m.group(1))
                # emit a row once we have all three for this epoch
                if train_loss is not None and val_loss is not None and pseudo_dice is not None:
                    rows.append({"step": epoch, "metric": "train/loss",    "value": train_loss})
                    rows.append({"step": epoch, "metric": "val/loss",      "value": val_loss})
                    rows.append({"step": epoch, "metric": "pseudo_dice",   "value": pseudo_dice})
                    train_loss = val_loss = pseudo_dice = None

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("step").drop_duplicates(subset=["step", "metric"], keep="last")
    return df


all_dfs = []

# --- TensorBoard sources (MicroSAM, SwinUNETR) ---
for model_name, cfg in TFEVENTS_SOURCES.items():
    root = cfg["root"]
    if not os.path.isdir(root):
        print(f"  [skip] {root} not found")
        continue
    for run_dir in sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)):
        run_name  = os.path.basename(run_dir)
        event_dir = os.path.join(run_dir, cfg["tb_subdir"]) if cfg["tb_subdir"] else run_dir
        df = read_tfevents(event_dir)
        if df.empty:
            print(f"  [empty] {model_name}/{run_name}")
            continue
        df["model"] = model_name
        df["run"]   = run_name
        all_dfs.append(df)
        print(f"  [ok] {model_name}/{run_name}  — {len(df)} rows, metrics: {sorted(df['metric'].unique())}")

# --- nnUNet (txt logs) ---
if os.path.isdir(NNUNET_ROOT):
    for dataset_dir in sorted(glob.glob(os.path.join(NNUNET_ROOT, "Dataset*"))):
        run_name = os.path.basename(dataset_dir)          # e.g. Dataset001_Calyx_All
        fold_dir = os.path.join(dataset_dir, "nnUNetTrainer__nnUNetPlans__3d_fullres", "fold_0")
        df = read_nnunet_txt(fold_dir)
        if df.empty:
            print(f"  [empty] nnunet/{run_name}")
            continue
        df["model"] = "nnunet"
        df["run"]   = run_name
        all_dfs.append(df)
        print(f"  [ok] nnunet/{run_name}  — {len(df)} rows, metrics: {sorted(df['metric'].unique())}")

if not all_dfs:
    print("No data found.")
else:
    out = pd.concat(all_dfs, ignore_index=True)[["model", "run", "step", "metric", "value"]]
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(out):,} rows -> {OUTPUT_CSV}")
    print("\nSummary:")
    print(out.groupby(["model", "run", "metric"]).size().rename("n_steps").reset_index().to_string(index=False))
