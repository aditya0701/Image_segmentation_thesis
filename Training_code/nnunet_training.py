"""
nnU-Net v2 Training Script
Drosophila microglomeruli synaptic bouton segmentation
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "NNUNET_RAW":          "/work/scratch/rawat/nnunet/nnUNet_raw",
    "NNUNET_PREPROCESSED": "/work/scratch/rawat/nnunet/nnUNet_preprocessed",
    "NNUNET_RESULTS":      "/work/scratch/rawat/models/nnunet",

    "DATASET_ID":          1,
    "FOLD":                0,
}

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT  (set before any nnU-Net import or subprocess call)
# ─────────────────────────────────────────────────────────────────────────────
os.environ["nnUNet_raw"]          = CONFIG["NNUNET_RAW"]
os.environ["nnUNet_preprocessed"] = CONFIG["NNUNET_PREPROCESSED"]
os.environ["nnUNet_results"]      = CONFIG["NNUNET_RESULTS"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def run(cmd: list) -> None:
    """Run a shell command, streaming output live. Exits on failure."""
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def dataset_tag(dataset_id: int) -> str:
    """Look up the actual folder name from the dataset ID — matches whatever
    prepare_nnunet_dataset.py created, regardless of the NAME used there."""
    from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name
    return convert_id_to_dataset_name(dataset_id)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="3d_fullres",
                        choices=["3d_fullres", "3d_lowres"])
    parser.add_argument("--dataset", type=int, default=CONFIG["DATASET_ID"],
                        help="Dataset ID — must match prepare_nnunet_dataset.py")
    parser.add_argument("--fold",    type=int, default=CONFIG["FOLD"])
    args = parser.parse_args()

    d    = str(args.dataset)
    fold = str(args.fold)
    cfg  = args.config

    # Resolve actual folder name from ID — matches whatever prepare_nnunet_dataset.py created
    import nnunetv2  # noqa: F401 — imported here to give a clean error if missing
    tag        = dataset_tag(args.dataset)
    prep_dir   = Path(CONFIG["NNUNET_PREPROCESSED"]) / tag
    splits_src = Path(CONFIG["NNUNET_RAW"]) / tag / "splits_final.json"

    print(f"Dataset  : {tag}")
    print(f"Config   : {cfg}")
    print(f"Fold     : {fold}")

    # ── Check preprocessing exists (must be run manually on nic before submitting)
    if not prep_dir.exists():
        sys.exit(
            f"ERROR: Preprocessed data not found at {prep_dir}\n"
            f"Run this on nic first:\n"
            f"  export nnUNet_raw={CONFIG['NNUNET_RAW']}\n"
            f"  export nnUNet_preprocessed={CONFIG['NNUNET_PREPROCESSED']}\n"
            f"  export nnUNet_results={CONFIG['NNUNET_RESULTS']}\n"
            f"  /work/scratch/rawat/installations/conda/envs/bw_env/bin/nnUNetv2_plan_and_preprocess "
            f"-d {d} --verify_dataset_integrity -np 1 -npfp 1\n"
            f"  cp {splits_src} {prep_dir}/splits_final.json"
        )

    splits_dst = prep_dir / "splits_final.json"
    if not splits_dst.exists():
        splits_dst.write_bytes(splits_src.read_bytes())
        print("Copied splits_final.json")

    # ── Train ────────────────────────────────────────────────────────────────
    run(["nnUNetv2_train", d, cfg, fold, "--npz"])

    print(f"\nTraining complete.")
    print(f"Results: {CONFIG['NNUNET_RESULTS']}/{tag}/{cfg}/fold_{fold}/")
