"""
Stage 5: OpenVINO on this laptop's CPU, Arc GPU and NPU, then NNCF INT8.

Model variants (OpenVINO IR = .xml graph + .bin weights):
  fp32        converted from the Stage 2 FP32 ONNX, weights kept FP32
  fp16        same graph, weights stored FP16 (half the file size)
  int8_nncf   NNCF post-training INT8 (model_type=TRANSFORMER keeps sensitive ops
              such as Softmax/LayerNorm in float), same 57 calibration tiles as Stage 3
  int8_qdq    the Stage 3 ONNX Runtime static MatMul-only QDQ file, read directly

Compute precision is set explicitly per device (INFERENCE_PRECISION_HINT), because
OpenVINO's defaults differ: the GPU computes in FP16 by default, and the NPU only
supports FP16/INT8, so "fp32" on the NPU runs as FP16. The row records what ran.

Modes:
  convert                     ONNX -> IR (fp32, fp16)
  quantize                    NNCF INT8 -> IR (int8_nncf)
  check  --device D --variant V   agreement with ONNX Runtime FP32 on 5 test tiles
  bench  --device D --variant V   latency (10 warm-up, 30 timed, batch 1), compile time,
                                  and the wall-clock window for power analysis
  idle                        60 s with nothing running, the power baseline
  eval   --device D --variant V   full thesis pipeline, scored on the 33 GT boutons

A device that refuses to compile a model is a valid result: the error is saved.

Usage:
    D:/my_envs/sam-edge/python.exe edge_deploy/stage5_openvino.py convert
"""
import argparse
import json
import sys
import time
import traceback

import numpy as np
import openvino as ov

import common
from make_calibration_set import CALIB_FILE
from stage2_onnx_export import onnx_path
from stage3_ort_int8 import int8_path, make_session

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

OV_DIR = common.MODELS_DIR / "openvino"
DEVICES = ("CPU", "GPU", "NPU")
# int8_nncf_direct: PyTorch -> OpenVINO without ONNX, then the same NNCF settings
# (made by stage5_gpu_int8_debug.py direct). It compiles on the Arc GPU; the
# ONNX-route int8_nncf does not.
# int8_fc_onnx / int8_fc_direct: NNCF INT8 for linear layers only, attention MatMuls
# kept FP16 (stage5_gpu_int8_debug.py fc_only). int8_weights: weight-only INT8.
VARIANTS = ("fp32", "fp16", "int8_nncf", "int8_qdq", "int8_nncf_direct",
            "int8_fc_onnx", "int8_fc_direct", "int8_weights")
VERSIONS = ("openvino", "nncf")
POWER_WINDOWS = common.RESULTS_DIR / "stage5_power_windows.json"
# HWiNFO logs (2 s interval). Session 1 (12:53-13:17): microsam_test_log.CSV, its
# windows/energy/bench files are archived in results/stage5_session1/.
# Session 2: GPU FP16 vs the GPU-compilable INT8 (int8_nncf_direct).
HWINFO_LOG = r"C:\Users\adima\OneDrive\Documents\microsam_test_log_1.CSV"
SOAK_SECONDS = 60


def ir_path(variant):
    return OV_DIR / f"sam_vitb_encoder_{variant}.xml"


def model_file(variant):
    return int8_path("static_matmul") if variant == "int8_qdq" else ir_path(variant)


def model_size_mb(variant):
    path = model_file(variant)
    return common.file_size_mb(path) if variant == "int8_qdq" else round(
        (path.stat().st_size + path.with_suffix(".bin").stat().st_size) / 1e6)


def compute_precision(device, variant):
    """What we ask the device to compute in. INT8 models run their quantized ops in
    INT8 and the rest in this precision."""
    if device == "NPU":
        return "f16"
    if device == "GPU":
        return "f32" if variant == "fp32" else "f16"
    return "f32"


def compile_model(device, variant):
    core = ov.Core()
    config = {"PERFORMANCE_HINT": "LATENCY",
              "INFERENCE_PRECISION_HINT": compute_precision(device, variant)}
    t0 = time.perf_counter()
    compiled = core.compile_model(str(model_file(variant)), device, config)
    return compiled, time.perf_counter() - t0


def record_failure(device, variant, mode, err):
    out = common.RESULTS_DIR / f"stage5_failure_{device}_{variant}_{mode}.txt"
    out.write_text(f"{type(err).__name__}: {err}\n\n{traceback.format_exc()}", encoding="utf-8")
    print(f"[{device} {variant}] FAILED in {mode}: {type(err).__name__}. Saved to {out.name}")


def convert():
    OV_DIR.mkdir(parents=True, exist_ok=True)
    model = ov.convert_model(str(onnx_path("dynamo")))
    ov.save_model(model, str(ir_path("fp32")), compress_to_fp16=False)
    ov.save_model(model, str(ir_path("fp16")), compress_to_fp16=True)
    for v in ("fp32", "fp16"):
        print(f"{ir_path(v).name}: {model_size_mb(v)} MB")


def quantize():
    import nncf
    tiles = np.load(CALIB_FILE, mmap_mode="r")
    dataset = nncf.Dataset(range(len(tiles)), lambda i: np.ascontiguousarray(tiles[i][None], dtype=np.float32))
    model = ov.Core().read_model(str(ir_path("fp32")))
    t0 = time.perf_counter()
    q = nncf.quantize(model, dataset, subset_size=len(tiles), model_type=nncf.ModelType.TRANSFORMER)
    ov.save_model(q, str(ir_path("int8_nncf")), compress_to_fp16=False)
    print(f"{ir_path('int8_nncf').name}: {model_size_mb('int8_nncf')} MB, {time.perf_counter() - t0:.0f} s, "
          f"{len(tiles)} calibration tiles")


def check(device, variant):
    from stage2_onnx_export import load_predictor
    try:
        compiled, compile_s = compile_model(device, variant)
    except Exception as e:  # noqa: BLE001  (a refusal to compile is a result)
        return record_failure(device, variant, "compile", e)
    predictor, _ = load_predictor()
    ref_sess = make_session(onnx_path("dynamo"))
    checks = []
    for i, x in enumerate(common.real_encoder_inputs(predictor)):
        ref = ref_sess.run(None, {"image": x.numpy()})[0]
        out = compiled(x.numpy())[0]
        c = common.compare_outputs(ref, out)
        checks.append(c)
        print(f"  tile {i}: max|diff| {c['max_abs_diff']:.3g}, mean|diff| {c['mean_abs_diff']:.3g}, cosine {c['cosine']:.5f}")
    summary = {"device": device, "variant": variant, "compute_precision": compute_precision(device, variant),
               "compile_s": round(compile_s, 1), "max_abs_diff": max(c["max_abs_diff"] for c in checks),
               "min_cosine": min(c["cosine"] for c in checks), "tiles": checks}
    (common.RESULTS_DIR / f"stage5_check_{device}_{variant}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{device} {variant}] compile {compile_s:.1f} s, worst cosine {summary['min_cosine']:.5f}")


def save_window(key, start, end, extra=None):
    windows = json.loads(POWER_WINDOWS.read_text()) if POWER_WINDOWS.exists() else {}
    windows[key] = {"start": start, "end": end, **(extra or {})}
    POWER_WINDOWS.write_text(json.dumps(windows, indent=2), encoding="utf-8")


def bench(device, variant):
    try:
        compiled, compile_s = compile_model(device, variant)
    except Exception as e:  # noqa: BLE001
        return record_failure(device, variant, "compile", e)
    request = compiled.create_infer_request()
    x = np.random.default_rng(common.SEED).standard_normal(common.ENCODER_INPUT_SHAPE).astype(np.float32)
    start = time.time()
    lat = common.benchmark(lambda: request.infer({0: x}))
    end = time.time()
    mem = common.peak_memory_mb()
    save_window(f"{device}_{variant}_bench", start, end, {"runs": lat["runs"] + lat["warmup"]})

    # Sustained-load window for energy: HWiNFO logs every 2 s, so a short benchmark
    # gets too few power samples. Run back to back for 60 s and count inferences.
    soak_start, n = time.time(), 0
    while time.time() - soak_start < SOAK_SECONDS:
        request.infer({0: x})
        n += 1
    save_window(f"{device}_{variant}", soak_start, time.time(), {"runs": n})
    if common.power_state() != common.REQUIRED_POWER:
        raise RuntimeError("Power changed during the sustained-load window.")
    result = {**lat, "compile_s": round(compile_s, 1), "peak_mem_mb": mem,
              "compute_precision": compute_precision(device, variant), "power": common.power_state()}
    (common.RESULTS_DIR / f"stage5_bench_{device}_{variant}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[{device} {variant}] median {lat['median_ms']} ms (p10 {lat['p10_ms']}, p90 {lat['p90_ms']}), "
          f"compile {compile_s:.1f} s, peak mem {mem} MB")


def idle(key="idle_start", seconds=60):
    if common.power_state() != common.REQUIRED_POWER:
        raise RuntimeError(f"Idle baseline refused: power is '{common.power_state()}'.")
    print(f"Idle baseline ({key}): {seconds} s, keep the laptop untouched...")
    start = time.time()
    time.sleep(seconds)
    save_window(key, start, time.time())
    print("Idle window saved.")


def evaluate(device, variant):
    from stage2_onnx_export import load_predictor
    common.set_seeds()
    run_name = f"stage5_openvino_{variant}_{device.lower()}"
    bench_file = common.RESULTS_DIR / f"stage5_bench_{device}_{variant}.json"
    lat = json.loads(bench_file.read_text()) if bench_file.exists() else {}
    compiled, _ = compile_model(device, variant)
    request = compiled.create_infer_request()
    predictor, segmenter = load_predictor()
    common.swap_encoder(predictor, lambda x: request.infer({0: x})[0].copy())
    rows, seconds = common.run_and_score(predictor, segmenter, run_name)
    (common.RESULTS_DIR / f"{run_name}_per_image.json").write_text(
        json.dumps({"rows": rows, "seconds_per_volume": seconds}, indent=2), encoding="utf-8")
    for thr in common.IOU_THRESHOLDS:
        acc = common.summarise_accuracy(rows, thr)
        print(f"IoU >= {thr}: {acc}")
        common.append_result({
            "stage": "5", "run": run_name, "device": device, "runtime": "OpenVINO",
            "precision": f"{variant} (compute {compute_precision(device, variant)})",
            "encoder_size_mb": model_size_mb(variant),
            **{k: v for k, v in lat.items() if k in ("median_ms", "p10_ms", "p90_ms", "min_ms", "max_ms", "runs", "warmup")},
            "peak_mem_encoder_mb": lat.get("peak_mem_mb"), "peak_mem_pipeline_mb": common.peak_memory_mb(),
            "pipeline_s_total": round(sum(seconds.values())), "iou_threshold": thr, **acc,
            "threads": "OpenVINO LATENCY hint", "versions": common.software_versions(VERSIONS),
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["convert", "quantize", "check", "bench", "idle", "eval"])
    parser.add_argument("--device", choices=DEVICES, default="CPU")
    parser.add_argument("--variant", choices=VARIANTS, default="fp32")
    parser.add_argument("--idle-key", choices=["idle_start", "idle_end"], default="idle_start")
    args = parser.parse_args()
    if args.mode == "idle":
        idle(args.idle_key)
    elif args.mode in ("convert", "quantize"):
        {"convert": convert, "quantize": quantize}[args.mode]()
    else:
        {"check": check, "bench": bench, "eval": evaluate}[args.mode](args.device, args.variant)
