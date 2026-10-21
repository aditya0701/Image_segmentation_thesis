"""
Shared code for all stages, so every stage uses identical preprocessing,
post-processing, scoring and timing. Only the image encoder changes between
stages (PyTorch, ONNX Runtime, TensorRT, OpenVINO).

The segmentation pipeline is copied from
Training_code/inference/batch_infer_microsam_base.py (the thesis pipeline),
and scoring uses compute_metrics() from tools/evaluate_segmentation.py unchanged.
"""
import csv
import importlib.metadata as md
import json
import platform
import random
import socket
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
import torch
from scipy.ndimage import generate_binary_structure, label as ndimage_label
from skimage.morphology import remove_small_objects

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))
from evaluate_segmentation import compute_metrics, to_3d  # noqa: E402  (thesis evaluation code)

# ── Paths and settings ───────────────────────────────────────────────────────
DATA_ROOT = Path(r"D:\Thesis\Thesis_Images\test\test")
CHECKPOINT_FULL = Path(r"D:\Thesis\microsam_B_all_checkpoints\best.pt")  # training checkpoint (pickle)
CHECKPOINT = CHECKPOINT_FULL.with_name("best_weights_only.pt")            # made by extract_weights.py
OUTPUT_ROOT = Path(r"D:\Thesis\edge_results")          # predicted masks (large, outside the repo)
RESULTS_DIR = Path(__file__).parent / "results"         # CSVs and logs (small, in the repo)
RESULTS_CSV = RESULTS_DIR / "results.csv"

# Test set: Brain1 + Brain2 PSF-deconvolved variant, 19 + 14 = 33 GT boutons.
TEST_PAIRS = [
    ("Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100.tif",
     "Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL_1100_mask.tif"),
    ("PN_KC_488_568_Brain2_CalyxR_LSMPlus_16bit.ome_deconv.tif",
     "PN_KC_488_568_Brain2_CalyxR_LSMPlus_16bit_mask.tif"),
]

# Same values as the thesis inference script.
MODEL_TYPE = "vit_b_lm"
TILE_SHAPE = (896, 896)
HALO = (64, 64)                       # tile + 2 * halo = 1024, the encoder input size
BLOB_REMOVAL_THRESHOLD = 862
ENCODER_INPUT_SHAPE = (1, 3, 1024, 1024)

# Scored at both thresholds; the headline one is chosen later from the thesis.
IOU_THRESHOLDS = (0.25, 0.5)

SEED = 0


def set_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── Pipeline (copied from the thesis inference script) ───────────────────────
def preprocess(raw_data):
    """Percentile-normalise to 0..255 and pack the 2 channels into an RGB stack (Z, Y, X, 3)."""
    if raw_data.ndim == 4:
        if raw_data.shape[0] <= 4:
            raw_data = np.transpose(raw_data, (1, 2, 3, 0))
        elif raw_data.shape[1] <= 4:
            raw_data = np.transpose(raw_data, (0, 2, 3, 1))
    elif raw_data.ndim == 3:
        raw_data = raw_data[..., np.newaxis]
    low, high = np.percentile(raw_data, (0.00001, 99.9999))
    if high > low:
        raw_data = np.clip((raw_data.astype(np.float32) - low) / (high - low), 0, 1) * 255
    else:
        raw_data = np.zeros_like(raw_data, dtype=np.float32)
    Z, Y, X, C = raw_data.shape
    rgb = np.zeros((Z, Y, X, 3), dtype=raw_data.dtype)
    rgb[..., :min(C, 2)] = raw_data[..., :min(C, 2)]
    return rgb


def segment_volume(predictor, segmenter, rgb):
    """Slice-by-slice tiled automatic segmentation, then 3D connected components and blob removal."""
    from micro_sam.automatic_segmentation import automatic_instance_segmentation

    stack = []
    for z in range(rgb.shape[0]):
        pred = automatic_instance_segmentation(
            predictor=predictor, segmenter=segmenter,
            input_path=rgb[z], ndim=2,
            tile_shape=TILE_SHAPE, halo=HALO, verbose=False,
        )
        stack.append(pred)
    structure = generate_binary_structure(3, 3)
    labeled, _ = ndimage_label(np.array(stack, dtype=np.uint32) > 0, structure=structure)
    clean = remove_small_objects(labeled.astype(bool), min_size=BLOB_REMOVAL_THRESHOLD, connectivity=1)
    clean, _ = ndimage_label(clean, structure=structure)
    return clean


def run_and_score(predictor, segmenter, run_name: str):
    """Segment every test volume, save the masks, and score them at each IoU threshold.

    Returns (rows, seconds_per_volume). One row per (image, threshold).
    """
    out_dir = OUTPUT_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, seconds, encoder_timing = [], {}, {}
    encoder = predictor.model.image_encoder
    for img_name, mask_name in TEST_PAIRS:
        rgb = preprocess(tifffile.imread(DATA_ROOT / "images" / img_name))
        t0 = time.perf_counter()
        pred = segment_volume(predictor, segmenter, rgb)
        seconds[img_name] = time.perf_counter() - t0
        if isinstance(encoder, ExternalEncoder) and encoder.times_ms:
            t = encoder.times_ms
            encoder_timing[img_name] = {
                "calls": len(t), "median_ms": round(statistics.median(t), 1),
                "p10_ms": round(statistics.quantiles(t, n=10)[0], 1), "p90_ms": round(statistics.quantiles(t, n=10)[-1], 1),
                "encoder_total_s": round(sum(t) / 1000), "volume_total_s": round(seconds[img_name]),
                "encoder_share": round(sum(t) / 1000 / seconds[img_name], 3)}
            encoder.times_ms = []
            (RESULTS_DIR / f"{run_name}_encoder_timing.json").write_text(json.dumps(encoder_timing, indent=2),
                                                                          encoding="utf-8")
        tifffile.imwrite(out_dir / f"{Path(img_name).stem}_masks.tif", pred.astype(np.uint16))

        gt = to_3d(tifffile.imread(DATA_ROOT / "masks" / mask_name))
        for thr in IOU_THRESHOLDS:
            rows.append({"image": img_name, "iou_threshold": thr, **compute_metrics(gt, pred, thr)})
        print(f"  {img_name}: {pred.max()} objects, {seconds[img_name]:.0f} s")
    return rows, seconds


def summarise_accuracy(rows, thr):
    """Combine the test images two ways.

    Pooled (micro): all boutons together, so recall = TP / 33 and mIoU is the mean
    over every matched bouton.
    Per-image mean (macro): the mean of each image's Recall and mean_iou_matched,
    as print_summary() in tools/evaluate_segmentation.py does for the thesis.
    """
    sel = [r for r in rows if r["iou_threshold"] == thr]
    tp = sum(r["TP"] for r in sel)
    fp = sum(r["FP"] for r in sel)
    fn = sum(r["FN"] for r in sel)
    n_gt = sum(r["n_gt"] for r in sel)
    miou = sum(r["mean_iou_matched"] * r["TP"] for r in sel) / tp if tp else 0.0
    return {"n_gt": n_gt, "TP": tp, "FP": fp, "FN": fn,
            "recall": round(tp / n_gt, 4) if n_gt else 0.0, "mIoU": round(miou, 4),
            "recall_per_image_mean": round(float(np.mean([r["Recall"] for r in sel])), 4),
            "mIoU_per_image_mean": round(float(np.mean([r["mean_iou_matched"] for r in sel])), 4)}


# ── Swapping the encoder (used by stages 2 to 5) ─────────────────────────────
MODELS_DIR = OUTPUT_ROOT / "models"                     # exported .onnx / OpenVINO files


class ExternalEncoder(torch.nn.Module):
    """Stands in for SAM's image encoder and forwards to another runtime.

    run_fn takes a float32 numpy array (1, 3, 1024, 1024) and returns the
    embedding (1, 256, 64, 64). micro_sam calls predictor.model.image_encoder
    for every tile (micro_sam/util.py, _compute_embeddings_batched), so
    replacing that module routes the whole pipeline through run_fn.
    """

    def __init__(self, run_fn, img_size: int = 1024):
        super().__init__()
        self.run_fn = run_fn
        self.img_size = img_size
        self.times_ms = []   # wall time of every call on real tiles, incl. tensor conversion

    def forward(self, x):
        t0 = time.perf_counter()
        out = torch.from_numpy(np.asarray(self.run_fn(x.detach().cpu().numpy().astype(np.float32))))
        self.times_ms.append((time.perf_counter() - t0) * 1000)
        return out


def swap_encoder(predictor, run_fn):
    """Replace the PyTorch encoder; the original is made to raise if anything still calls it."""
    original = predictor.model.image_encoder

    def _fail(*args, **kwargs):
        raise RuntimeError("The PyTorch encoder was called after being replaced.")

    original.forward = _fail
    predictor.model.image_encoder = ExternalEncoder(run_fn, original.img_size)
    return original


def real_encoder_inputs(predictor, n: int = 5):
    """n real encoder inputs, prepared exactly as micro_sam prepares a tile:
    a 1024x1024 crop of a test slice, ResizeLongestSide, then SAM normalisation and padding.
    Used only to check numerical agreement, never for calibration."""
    img_name, _ = TEST_PAIRS[0]
    rgb = preprocess(tifffile.imread(DATA_ROOT / "images" / img_name))
    zs = np.linspace(0, rgb.shape[0] - 1, n).astype(int)
    inputs = []
    for z in zs:
        tile = rgb[z, :1024, :1024].astype(np.uint8)
        t = torch.as_tensor(predictor.transform.apply_image(tile)).permute(2, 0, 1)[None].float()
        inputs.append(predictor.model.preprocess(t))
    return inputs


def compare_outputs(ref, test):
    """Numerical agreement between two embeddings (numpy arrays)."""
    ref, test = ref.astype(np.float64).ravel(), test.astype(np.float64).ravel()
    diff = np.abs(ref - test)
    cos = float(ref @ test / (np.linalg.norm(ref) * np.linalg.norm(test)))
    return {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
            "ref_abs_max": float(np.abs(ref).max()), "cosine": cos}


def file_size_mb(path: Path):
    """Size of a model file plus its external-data file, if it has one."""
    total = path.stat().st_size
    data = path.with_name(path.name + ".data")
    if data.exists():
        total += data.stat().st_size
    return round(total / 1e6)


# ── Timing and memory ─────────────────────────────────────────────────────────
REQUIRED_POWER = "AC, Best performance"


def benchmark(fn, warmup: int = 10, runs: int = 30):
    """Time fn() with warm-up. Returns latency stats in milliseconds.
    Refuses to run unless the laptop is on AC power in Best performance mode,
    and checks again afterwards, so throttled numbers are never recorded."""
    if power_state() != REQUIRED_POWER:
        raise RuntimeError(f"Benchmark refused: power is '{power_state()}', need '{REQUIRED_POWER}'.")
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    if power_state() != REQUIRED_POWER:
        raise RuntimeError(f"Benchmark invalid: power changed to '{power_state()}' during the runs.")
    q = statistics.quantiles(times, n=10)  # deciles: q[0] = p10, q[-1] = p90
    return {"median_ms": round(statistics.median(times), 1),
            "p10_ms": round(q[0], 1), "p90_ms": round(q[-1], 1),
            "min_ms": round(min(times), 1), "max_ms": round(max(times), 1),
            "runs": runs, "warmup": warmup}


def peak_memory_mb():
    """Peak resident memory of this process so far (Windows: peak working set)."""
    import psutil
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None) or info.rss
    return round(peak / 1024**2)


# ── Recording results ─────────────────────────────────────────────────────────
def software_versions(extra=()):
    names = ["torch", "micro_sam", "numpy", *extra]
    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = "n/a"
    return ";".join(f"{k}={v}" for k, v in out.items())


POWER_MODES = {"ded574b5-45a0-4f42-8737-46345c09c238": "Best performance",
               "00000000-0000-0000-0000-000000000000": "Balanced",
               "961cc777-2547-4f9d-8174-7d86181b8a7a": "Best power efficiency"}


def power_state():
    """'AC' or 'battery' plus the Windows 11 power mode, since laptop CPUs throttle on battery or in efficiency mode."""
    import psutil
    battery = psutil.sensors_battery()
    on_ac = battery is None or battery.power_plugged
    mode = "unknown"
    if platform.system() == "Windows":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes") as k:
                guid, _ = winreg.QueryValueEx(k, "ActiveOverlayAcPowerScheme" if on_ac else "ActiveOverlayDcPowerScheme")
                mode = POWER_MODES.get(guid, guid)
        except OSError:
            pass
    return f"{'AC' if on_ac else 'battery'}, {mode}"


def append_result(row: dict):
    """Append one row to results/results.csv. New columns are added to the header,
    and older rows get empty cells for them."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "host": socket.gethostname(), "cpu": platform.processor(), "power": power_state(), **row}
    existing = []
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    fields = list(existing[0].keys()) if existing else []
    fields += [k for k in row if k not in fields]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(existing + [row])
