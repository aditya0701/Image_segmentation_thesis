import numpy as np
import tifffile
import pandas as pd
from pathlib import Path
from scipy.optimize import linear_sum_assignment

RESULTS_ROOT = Path(r"D:\Thesis_images\Updated_final_results")
DATASET_ROOT = Path(r"D:\Thesis_images\Final_dataset")
OUTPUT_DIR = RESULTS_ROOT / "eval_csv"

VOXEL_VOL = 0.0709 * 0.0709 * 0.3  # µm³ per voxel

# Files whose names contain these substrings are post-processed variants — skip them
SKIP_SUFFIXES = ("_edges", "_inner_edges", "_shrunk", "_final", "_rough")

# GT filename suffixes and how to strip them to get the base name
GT_SUFFIXES = ("_mask.tif",)

# Prediction filename suffixes (tried in order)
PRED_SUFFIXES = ("_masks.tif", "_mask.tif")

# Wrongly-trained model versions superseded by retrained "batch 1" versions —
# excluded so only the retrained version's metrics appear (not both).
REPLACED_MODELS = {
    "vit_b_lm_neo_augmented_All_v1":      "vit_b_lm_neo_augmented_All_v2_batch1",
    "vit_l_lm_neo_augmented_All_v3":      "vit_l_lm_neo_augmented_All_v5_batch1",
    "vit_l_lm_neo_augmented_Original_v3": "vit_l_lm_neo_augmented_Original_v4_batch1",
}


def compute_iou_matrix(gt: np.ndarray, pred: np.ndarray):
    """Return (iou_matrix, gt_ids, pred_ids) using vectorized pair counting."""
    gt_ids = np.unique(gt[gt > 0])
    pred_ids = np.unique(pred[pred > 0])

    if gt_ids.size == 0 or pred_ids.size == 0:
        return np.zeros((len(gt_ids), len(pred_ids))), gt_ids, pred_ids

    gt_areas = {i: int(np.sum(gt == i)) for i in gt_ids}
    pred_areas = {i: int(np.sum(pred == i)) for i in pred_ids}

    overlap_mask = (gt > 0) & (pred > 0)
    gt_ov = gt[overlap_mask].astype(np.int64)
    pred_ov = pred[overlap_mask].astype(np.int64)

    max_pred_id = int(pred_ids.max()) + 1
    combined = gt_ov * max_pred_id + pred_ov
    unique_combined, counts = np.unique(combined, return_counts=True)
    gt_pairs = unique_combined // max_pred_id
    pred_pairs = unique_combined % max_pred_id

    # Build gt_id → row index and pred_id → col index maps
    gt_idx = {v: i for i, v in enumerate(gt_ids)}
    pred_idx = {v: i for i, v in enumerate(pred_ids)}

    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    for g, p, count in zip(gt_pairs, pred_pairs, counts):
        i, j = gt_idx[g], pred_idx[p]
        union = gt_areas[g] + pred_areas[p] - count
        iou_matrix[i, j] = count / union

    return iou_matrix, gt_ids, pred_ids


def match_instances(iou_matrix: np.ndarray, gt_ids: np.ndarray, pred_ids: np.ndarray, threshold: float = 0.5):
    """Hungarian matching; return (matched_pairs, tp, fp, fn).
    matched_pairs: list of (gt_id, pred_id, iou)
    """
    n_gt, n_pred = iou_matrix.shape
    if n_gt == 0 or n_pred == 0:
        return [], 0, n_pred, n_gt  # tp=0, fp=n_pred, fn=n_gt

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_pairs = []
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= threshold:
            matched_pairs.append((int(gt_ids[r]), int(pred_ids[c]), float(iou_matrix[r, c])))

    tp = len(matched_pairs)
    fp = n_pred - tp
    fn = n_gt - tp
    return matched_pairs, tp, fp, fn


def compute_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray, iou_threshold: float = 0.5) -> dict:
    iou_matrix, gt_ids, pred_ids = compute_iou_matrix(gt_mask, pred_mask)
    matched_pairs, tp, fp, fn = match_instances(iou_matrix, gt_ids, pred_ids, iou_threshold)

    n_gt = len(gt_ids)
    n_pred = len(pred_ids)

    count_error = abs(n_pred - n_gt) / n_gt if n_gt > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    matched_ious = [iou for _, _, iou in matched_pairs]
    sq = float(np.mean(matched_ious)) if matched_ious else 0.0
    pq = sq * f1

    vol_errors = []
    for gt_id, pred_id, _ in matched_pairs:
        v_gt = float(np.sum(gt_mask == gt_id)) * VOXEL_VOL
        v_pred = float(np.sum(pred_mask == pred_id)) * VOXEL_VOL
        if v_gt > 0:
            vol_errors.append(abs(v_pred - v_gt) / v_gt)
    mean_vol_error = float(np.mean(vol_errors)) if vol_errors else 0.0

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "count_error": round(count_error, 4),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "mean_iou_matched": round(sq, 4),
        "PQ": round(pq, 4),
        "mean_vol_error": round(mean_vol_error, 4),
    }


def infer_dataset_type(folder_name: str) -> str | None:
    name_lower = folder_name.lower()
    for dtype in ("all", "original", "dog", "psf"):
        if dtype in name_lower:
            return dtype
    return None


def build_gt_index(dataset_type: str) -> dict[str, Path]:
    """Return {gt_base_name: gt_path} for the given dataset type."""
    masks_dir = DATASET_ROOT / dataset_type / "test" / "masks"
    index = {}
    for f in masks_dir.glob("*.tif"):
        name = f.name
        for suffix in GT_SUFFIXES:
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                index[base] = f
                break
    return index


def find_matching_gt(pred_path: Path, gt_index: dict[str, Path]) -> Path | None:
    """Find GT whose base name is a prefix of the prediction's base name."""
    name = pred_path.name
    pred_base = name
    for suffix in PRED_SUFFIXES:
        if name.endswith(suffix):
            pred_base = name[: -len(suffix)]
            break

    # Exact match first
    if pred_base in gt_index:
        return gt_index[pred_base]

    # Prefix match: find all GT bases that are a prefix of pred_base
    matches = [base for base in gt_index if pred_base.startswith(base)]
    if len(matches) == 1:
        return gt_index[matches[0]]
    if len(matches) > 1:
        # Pick the longest matching prefix
        best = max(matches, key=len)
        return gt_index[best]
    return None


def to_3d(arr: np.ndarray) -> np.ndarray:
    """Squeeze and reduce to exactly (Z, Y, X) integer array."""
    arr = np.squeeze(arr)
    if arr.ndim == 4:
        arr = arr[0]  # take first channel; slice before astype to avoid OOM
    if arr.ndim != 3:
        raise ValueError(f"Cannot reduce array of shape {arr.shape} to 3D")
    return arr.astype(np.int32)


def should_skip(filename: str) -> bool:
    return any(s in filename for s in SKIP_SUFFIXES)


def run_batch_evaluation(iou_threshold: float = 0.5) -> pd.DataFrame:
    rows = []

    for family_dir in sorted(RESULTS_ROOT.iterdir()):
        if not family_dir.is_dir():
            continue
        model_family = family_dir.name

        # Support both nested (family/model/*.tif) and flat (family/*.tif) layouts
        subdirs = [d for d in family_dir.iterdir() if d.is_dir()]
        model_dirs = subdirs if subdirs else [family_dir]

        for model_dir in sorted(model_dirs):
            model_name = model_dir.name
            if model_name in REPLACED_MODELS:
                print(f"  [SKIP] {model_name} (wrongly trained — replaced by {REPLACED_MODELS[model_name]})")
                continue
            dataset_type = infer_dataset_type(model_name)
            if dataset_type is None:
                print(f"  [WARN] Cannot infer dataset type for {model_name}, skipping.")
                continue

            gt_index = build_gt_index(dataset_type)

            tif_files = sorted(model_dir.glob("*.tif"))
            pred_files = [f for f in tif_files if not should_skip(f.name)]

            for pred_path in pred_files:
                gt_path = find_matching_gt(pred_path, gt_index)
                if gt_path is None:
                    print(f"  [WARN] No GT found for {pred_path.name} ({model_name})")
                    continue

                print(f"  {model_family}/{model_name} | {pred_path.name} <- {gt_path.name}")
                gt = tifffile.imread(str(gt_path))
                pred = tifffile.imread(str(pred_path))

                # Ensure 3D (Z, Y, X) integer arrays
                gt = to_3d(gt)
                pred = to_3d(pred)

                metrics = compute_metrics(gt, pred, iou_threshold)
                rows.append({
                    "model_family": model_family,
                    "model_name": model_name,
                    "dataset_type": dataset_type,
                    "image": pred_path.stem,
                    "gt_file": gt_path.name,
                    **metrics,
                })

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, iou_threshold: float = 0.5):
    summary = (
        df.groupby(["model_family", "model_name", "dataset_type"])
        .agg(
            images=("image", "count"),
            n_gt=("n_gt", "sum"),
            n_pred=("n_pred", "sum"),
            count_error=("count_error", "mean"),
            TP=("TP", "sum"),
            FP=("FP", "sum"),
            FN=("FN", "sum"),
            Precision=("Precision", "mean"),
            Recall=("Recall", "mean"),
            F1=("F1", "mean"),
            mean_iou_matched=("mean_iou_matched", "mean"),
            PQ=("PQ", "mean"),
            mean_vol_error=("mean_vol_error", "mean"),
        )
        .reset_index()
    )
    for col in ("count_error", "Precision", "Recall", "F1", "mean_iou_matched", "PQ", "mean_vol_error"):
        summary[col] = summary[col].round(4)

    print("\n" + "=" * 100)
    print(f"EVALUATION SUMMARY (mean over test images, IoU threshold = {iou_threshold})")
    print("=" * 100)
    print(summary.to_string(index=False))
    print("=" * 100 + "\n")
    return summary


METRICS_DESCRIPTION_TEMPLATE = """\
# Evaluation Metrics — Instance Segmentation
IoU threshold for TP matching: {iou_threshold}
Voxel volume: {voxel_vol:.6f} µm³  (XY = 0.0709 µm, Z = 0.3 µm)

## Instance counts
  n_gt            Number of ground-truth instances in this image.
  n_pred          Number of predicted instances in this image.

## Count error (relative, dimensionless)
  count_error     |n_pred − n_gt| / n_gt
                  How far off the model is in total instance count, as a fraction
                  of the true count.  0 = perfect count.  0.2 = 20% over/under.
                  This is RELATIVE (normalised by n_gt), so it is comparable
                  across images with different numbers of boutons.

## Detection metrics (at IoU ≥ {iou_threshold}, Hungarian matching)
  Each predicted instance is matched to at most one GT instance via the
  Hungarian algorithm.  A match is accepted only if IoU ≥ {iou_threshold}.

  TP              True positives  — matched GT–pred pairs with IoU ≥ {iou_threshold}.
  FP              False positives — predicted instances with no GT match.
  FN              False negatives — GT instances with no predicted match.

  Precision       TP / (TP + FP)   — of all predictions, fraction that are correct.
  Recall          TP / (TP + FN)   — of all GT instances, fraction that were found.
  F1              2 · Precision · Recall / (Precision + Recall)
                  Harmonic mean of Precision and Recall.

## Spatial quality
  mean_iou_matched  Average IoU over all TP (matched) pairs.
                    Measures how well the spatial extent of matched instances agrees.
                    Also called Segmentation Quality (SQ) in the Panoptic Quality paper.

## Panoptic Quality
  PQ              mean_iou_matched × F1   (SQ × RQ)
                  Standard single-number metric for instance segmentation.
                  Penalises both wrong counts (via F1) and poor shape overlap
                  (via mean IoU).  Range [0, 1]; higher is better.

## Volume accuracy (matched pairs only)
  mean_vol_error  Mean relative volume error over TP pairs:
                    mean( |v_pred − v_gt| / v_gt )   for each matched (gt, pred) pair.
                  v_gt  and v_pred are bouton volumes in µm³ computed from voxel counts.
                  0 = model reproduces GT volumes exactly.
                  0.3 = model volumes are on average 30% off from GT volumes.
                  Only matched instances are included; FP/FN do not contribute.

## Summary CSV vs per-image CSV
  evaluation_results.csv   One row per prediction file (per image per model).
  evaluation_summary.csv   One row per model subfolder; numeric columns are
                           averaged across all test images for that model, except
                           TP/FP/FN/n_gt/n_pred which are summed.
"""


def write_metrics_description(output_dir: Path, iou_threshold: float = 0.5):
    text = METRICS_DESCRIPTION_TEMPLATE.format(voxel_vol=VOXEL_VOL, iou_threshold=iou_threshold)
    (output_dir / "metrics_description.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    print(f"Results root : {RESULTS_ROOT}")
    print(f"Dataset root : {DATASET_ROOT}")
    print(f"Output dir   : {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    iou_threshold = 0.25
    df = run_batch_evaluation(iou_threshold=iou_threshold)

    if df.empty:
        print("No results found. Check paths and file naming conventions.")
    else:
        out_detail = OUTPUT_DIR / "evaluation_results.csv"
        df.to_csv(out_detail, index=False)
        print(f"\nPer-image results saved to: {out_detail}")

        summary = print_summary(df, iou_threshold=iou_threshold)
        out_summary = OUTPUT_DIR / "evaluation_summary.csv"
        summary.to_csv(out_summary, index=False)
        print(f"Summary saved to: {out_summary}")

        write_metrics_description(OUTPUT_DIR, iou_threshold=iou_threshold)
        print(f"Metrics description saved to: {OUTPUT_DIR / 'metrics_description.txt'}")
