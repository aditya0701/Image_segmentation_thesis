"""
Build the INT8 calibration set from TRAINING volumes only (never the test set).

Picks 3 random slices from each training volume and a random 1024x1024 window
from each slice (the encoder input size), then prepares it exactly like a
pipeline tile: thesis percentile normalisation, then SAM normalisation.
The seed is fixed and every tile's source (file, z, y, x) is written to a JSON
file, so the set can be rebuilt and audited. Stages 3, 4 and 5 all use it.

Usage:
    D:/my_envs/sam-edge/python.exe edge_deploy/make_calibration_set.py
Writes:
    D:/Thesis/edge_results/calibration/calib_tiles.npy   (N, 3, 1024, 1024) float32
    edge_deploy/results/calibration_set.json             tile sources
"""
import hashlib
import json

import numpy as np
import tifffile
import torch

import common

TRAIN_IMAGES = common.DATA_ROOT.parents[1] / "train" / "train" / "images"
CALIB_DIR = common.OUTPUT_ROOT / "calibration"
CALIB_FILE = CALIB_DIR / "calib_tiles.npy"
SLICES_PER_VOLUME = 3


def sha16(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            d.update(chunk)
    return d.hexdigest()[:16]


def main():
    from micro_sam.automatic_segmentation import get_predictor_and_segmenter
    predictor, _ = get_predictor_and_segmenter(model_type=common.MODEL_TYPE, checkpoint=str(common.CHECKPOINT),
                                               device="cpu", is_tiled=True)
    rng = np.random.default_rng(common.SEED)

    # Guard against leakage: no training file may be byte-identical to a test file.
    test_hashes = {sha16(p) for p in (common.DATA_ROOT / "images").glob("*.tif")}

    tiles, sources = [], []
    for path in sorted(TRAIN_IMAGES.glob("*.tif")):
        assert sha16(path) not in test_hashes, f"{path.name} is identical to a test image"
        rgb = common.preprocess(tifffile.imread(path))          # (Z, Y, X, 3), 0..255
        Z, Y, X, _ = rgb.shape
        for z in sorted(rng.choice(Z, size=SLICES_PER_VOLUME, replace=False)):
            y, x = int(rng.integers(0, Y - 1024 + 1)), int(rng.integers(0, X - 1024 + 1))
            tile = rgb[z, y:y + 1024, x:x + 1024].astype(np.uint8)
            t = torch.as_tensor(predictor.transform.apply_image(tile)).permute(2, 0, 1)[None].float()
            tiles.append(predictor.model.preprocess(t).numpy()[0])
            sources.append({"file": path.name, "z": int(z), "y": y, "x": x})
        print(f"  {path.name}: slices {[s['z'] for s in sources[-SLICES_PER_VOLUME:]]}")

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    arr = np.stack(tiles).astype(np.float32)
    np.save(CALIB_FILE, arr)
    (common.RESULTS_DIR / "calibration_set.json").write_text(json.dumps(
        {"seed": common.SEED, "n_tiles": len(tiles), "source_dir": str(TRAIN_IMAGES),
         "shape": list(arr.shape), "tiles": sources}, indent=2), encoding="utf-8")
    print(f"Saved {arr.shape} to {CALIB_FILE} ({arr.nbytes / 1e6:.0f} MB); "
          f"value range {arr.min():.2f} to {arr.max():.2f}")


if __name__ == "__main__":
    main()
