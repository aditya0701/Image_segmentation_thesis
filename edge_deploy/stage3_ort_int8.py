"""
Stage 3: INT8 post-training quantization with ONNX Runtime, run on CPU.

Starts from the verified Stage 2 FP32 model (torch.export exporter).

Variants:
  dynamic   Weights stored as INT8; activation ranges computed at run time for
            every input. No calibration data.
  static    Weights and activations INT8, activation ranges fixed in advance from
            the calibration set (57 training tiles, make_calibration_set.py).
            QDQ format, per-channel weight scales, MinMax calibration.

Modes:
  prep                      Shape inference + graph clean-up, recommended before quantizing.
  quantize --variant X      Write the INT8 model.
  check --variant X         Agreement with FP32 ONNX on 5 real test tiles (evaluation only;
                            these tiles are never used for calibration).
  bench --variant X         Latency (10 warm-up, 30 timed, batch 1), session.run only.
  eval --variant X          Full thesis pipeline with the INT8 encoder, scored on the 33 GT boutons.

Usage:
    D:/my_envs/sam-edge/python.exe edge_deploy/stage3_ort_int8.py prep
"""
import argparse
import json
import sys
import time

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_dynamic, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

import common
from make_calibration_set import CALIB_FILE
from stage2_onnx_export import onnx_path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

FP32_PATH = onnx_path("dynamo")
PREP_PATH = common.MODELS_DIR / "sam_vitb_encoder_fp32_prep.onnx"
CALIB_RANGES = common.RESULTS_DIR / "stage3_static_calibration_ranges.json"
VARIANTS = ("dynamic", "static", "static_matmul", "trt_matmul")

# static_matmul: quantize only the compute-heavy ops. LayerNorm, Softmax, GELU, the
# residual Adds and Einsum stay FP32, because rounding errors there accumulate
# across the 12 transformer blocks (full static had cosine 0.989 vs 0.9996 dynamic).
MATMUL_OPS = ["MatMul", "Gemm", "Conv"]

# trt_matmul: same as static_matmul, but in the form TensorRT accepts (Stage 4). TensorRT
# rejects non-zero zero points ("shiftIsAllZeros") and Int32 bias DequantizeLinear nodes,
# so activations and weights are symmetric and the biases stay float.
TRT_EXTRA_OPTIONS = {"ActivationSymmetric": True, "WeightSymmetric": True, "QuantizeBias": False}
VERSIONS = ("onnx", "onnxruntime")


def int8_path(variant):
    return common.MODELS_DIR / f"sam_vitb_encoder_int8_{variant}.onnx"


def make_session(path):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class TileReader(CalibrationDataReader):
    """Feeds the calibration tiles to the calibrator one at a time (batch size 1)."""

    def __init__(self):
        self.tiles = np.load(CALIB_FILE, mmap_mode="r")
        self.i = 0

    def get_next(self):
        if self.i >= len(self.tiles):
            return None
        x = np.ascontiguousarray(self.tiles[self.i][None], dtype=np.float32)
        self.i += 1
        print(f"  calibration tile {self.i}/{len(self.tiles)}", end="\r", flush=True)
        return {"image": x}


def prep():
    t0 = time.perf_counter()
    quant_pre_process(str(FP32_PATH), str(PREP_PATH))
    print(f"Pre-processed model: {PREP_PATH.name}, {common.file_size_mb(PREP_PATH)} MB, "
          f"{time.perf_counter() - t0:.0f} s")


def quantize(variant):
    out = int8_path(variant)
    t0 = time.perf_counter()
    if variant == "dynamic":
        quantize_dynamic(str(PREP_PATH), str(out), weight_type=QuantType.QInt8)
    else:
        reader = TileReader()
        quantize_static(
            str(PREP_PATH), str(out), reader,
            op_types_to_quantize=MATMUL_OPS if variant in ("static_matmul", "trt_matmul") else None,
            extra_options=TRT_EXTRA_OPTIONS if variant == "trt_matmul" else None,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
            per_channel=True, calibrate_method=CalibrationMethod.MinMax,
            # Do NOT set extra_options CalibMaxIntermediateOutputs: in onnxruntime 1.30 it
            # discards collected ranges instead of merging them (calibrate.py, collect_data).
            # The MinMax calibrator only keeps ReduceMin/ReduceMax scalars per tensor, so
            # keeping all 57 tiles costs little memory anyway.
            calibration_cache_path=str(CALIB_RANGES.with_stem(f"stage3_{variant}_calibration_ranges")),
        )
        print(f"\nCalibration tiles read: {reader.i}")
        assert reader.i == len(reader.tiles), "not every calibration tile was used"
    print(f"\n{out.name}: {common.file_size_mb(out)} MB (FP32: {common.file_size_mb(FP32_PATH)} MB), "
          f"{time.perf_counter() - t0:.0f} s")


def check(variant):
    from stage2_onnx_export import load_predictor
    predictor, _ = load_predictor()
    ref_sess, q_sess = make_session(FP32_PATH), make_session(int8_path(variant))
    checks = []
    for i, x in enumerate(common.real_encoder_inputs(predictor)):
        ref = ref_sess.run(None, {"image": x.numpy()})[0]
        out = q_sess.run(None, {"image": x.numpy()})[0]
        c = common.compare_outputs(ref, out)
        checks.append(c)
        print(f"  tile {i}: max|diff| {c['max_abs_diff']:.3f}, mean|diff| {c['mean_abs_diff']:.4f}, "
              f"cosine {c['cosine']:.5f} (output range up to {c['ref_abs_max']:.2f})")
    summary = {"variant": variant, "max_abs_diff": max(c["max_abs_diff"] for c in checks),
               "mean_abs_diff": float(np.mean([c["mean_abs_diff"] for c in checks])),
               "min_cosine": min(c["cosine"] for c in checks), "tiles": checks}
    (common.RESULTS_DIR / f"stage3_check_{variant}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  worst case: max|diff| {summary['max_abs_diff']:.3f}, min cosine {summary['min_cosine']:.5f}")


def bench(variant):
    session = make_session(int8_path(variant))
    x = np.random.default_rng(common.SEED).standard_normal(common.ENCODER_INPUT_SHAPE).astype(np.float32)
    lat = common.benchmark(lambda: session.run(None, {"image": x}))
    mem = common.peak_memory_mb()
    print(f"[{variant}] latency: {lat}\nPeak process memory: {mem} MB, power: {common.power_state()}")
    (common.RESULTS_DIR / f"stage3_bench_{variant}.json").write_text(
        json.dumps({**lat, "peak_mem_mb": mem, "power": common.power_state()}, indent=2), encoding="utf-8")


def evaluate(variant):
    from stage2_onnx_export import load_predictor
    common.set_seeds()
    run_name = f"stage3_onnxruntime_int8_{variant}_cpu"
    bench_file = common.RESULTS_DIR / f"stage3_bench_{variant}.json"
    lat = json.loads(bench_file.read_text()) if bench_file.exists() else {}
    predictor, segmenter = load_predictor()
    session = make_session(int8_path(variant))
    common.swap_encoder(predictor, lambda x: session.run(None, {"image": x})[0])
    rows, seconds = common.run_and_score(predictor, segmenter, run_name)
    (common.RESULTS_DIR / f"{run_name}_per_image.json").write_text(
        json.dumps({"rows": rows, "seconds_per_volume": seconds}, indent=2), encoding="utf-8")
    for thr in common.IOU_THRESHOLDS:
        acc = common.summarise_accuracy(rows, thr)
        print(f"IoU >= {thr}: {acc}")
        common.append_result({
            "stage": "3", "run": run_name, "device": "CPU", "runtime": "ONNX Runtime",
            "precision": f"INT8 {variant}", "encoder_size_mb": common.file_size_mb(int8_path(variant)),
            **{k: v for k, v in lat.items() if k not in ("peak_mem_mb", "power")},
            "peak_mem_encoder_mb": lat.get("peak_mem_mb"), "peak_mem_pipeline_mb": common.peak_memory_mb(),
            "pipeline_s_total": round(sum(seconds.values())), "iou_threshold": thr, **acc,
            "threads": "ORT default", "versions": common.software_versions(VERSIONS),
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prep", "quantize", "check", "bench", "eval"])
    parser.add_argument("--variant", choices=VARIANTS, default="dynamic")
    args = parser.parse_args()
    if args.mode == "prep":
        prep()
    else:
        {"quantize": quantize, "check": check, "bench": bench, "eval": evaluate}[args.mode](args.variant)
