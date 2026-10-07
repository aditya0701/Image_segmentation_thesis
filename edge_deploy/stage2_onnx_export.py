"""
Stage 2: export the MicroSAM Base image encoder to ONNX and run it with ONNX Runtime on CPU.

Two exporters are compared, each writing its own file:
  dynamo   torch.export-based exporter (the default in recent PyTorch)
  legacy   TorchScript tracer (deprecated; records one run of the model)

Modes:
  export --exporter X   Export with a fixed 1x3x1024x1024 input, check the file, and
                        compare ONNX Runtime against PyTorch on 5 real tiles.
  compare               Side-by-side table of both exported graphs.
  bench --exporter X    Encoder latency in ONNX Runtime CPU (10 warm-up, 30 timed runs,
                        batch 1). Included: session.run only. The PyTorch model is not
                        loaded, so peak memory reflects ONNX Runtime (plus the torch import).
  eval --exporter X     Full thesis pipeline with the ONNX encoder swapped in (about
                        100 min), scored against the 33 GT boutons.

Usage:
    D:/my_envs/sam-edge/python.exe edge_deploy/stage2_onnx_export.py export --exporter dynamo
"""
import argparse
import json
import sys
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort
import torch

import common

# Windows writes redirected output (log files, pipes) as cp1252, which cannot encode
# characters such as emoji that libraries may print. UTF-8 can encode everything.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

EXPORTERS = ("dynamo", "legacy")
VERSIONS = ("onnx", "onnxruntime", "onnxscript")


def onnx_path(exporter):
    return common.MODELS_DIR / f"sam_vitb_encoder_fp32_{exporter}.onnx"


def load_predictor():
    from micro_sam.automatic_segmentation import get_predictor_and_segmenter
    return get_predictor_and_segmenter(model_type=common.MODEL_TYPE, checkpoint=str(common.CHECKPOINT),
                                       device="cpu", is_tiled=True)


def make_session(path):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def graph_info(path):
    model = onnx.load(str(path), load_external_data=False)
    ops = Counter(n.op_type for n in model.graph.node)
    return {"opset": max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
            "n_nodes": len(model.graph.node), "op_counts": dict(sorted(ops.items())),
            "file_size_mb": common.file_size_mb(path),
            "external_data": path.with_name(path.name + ".data").exists()}


def export(exporter):
    common.set_seeds()
    predictor, _ = load_predictor()
    encoder = predictor.model.image_encoder.eval()
    example = torch.randn(common.ENCODER_INPUT_SHAPE)
    path = onnx_path(exporter)
    common.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if exporter == "dynamo":
        # verbose=False switches off the exporter's progress prints.
        torch.onnx.export(encoder, (example,), str(path), dynamo=True, verbose=False,
                          input_names=["image"], output_names=["embedding"])
    else:
        torch.onnx.export(encoder, (example,), str(path), dynamo=False, opset_version=17,
                          input_names=["image"], output_names=["embedding"])

    onnx.checker.check_model(str(path))  # path form also handles files with external data
    info = graph_info(path)
    print(f"[{exporter}] {path.name}: {info['file_size_mb']} MB, opset {info['opset']}, "
          f"{info['n_nodes']} nodes, {len(info['op_counts'])} op types")

    # Numerical agreement on real tiles.
    session = make_session(path)
    checks = []
    with torch.inference_mode():
        for i, x in enumerate(common.real_encoder_inputs(predictor)):
            ref = encoder(x).numpy()
            out = session.run(None, {"image": x.numpy()})[0]
            c = common.compare_outputs(ref, out)
            checks.append(c)
            print(f"  tile {i}: max|diff| {c['max_abs_diff']:.2e}, mean|diff| {c['mean_abs_diff']:.2e}, "
                  f"cosine {c['cosine']:.8f}")
    summary = {"exporter": exporter, **info,
               "max_abs_diff": max(c["max_abs_diff"] for c in checks),
               "min_cosine": min(c["cosine"] for c in checks), "tiles": checks}
    (common.RESULTS_DIR / f"stage2_export_check_{exporter}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  worst case: max|diff| {summary['max_abs_diff']:.2e}, min cosine {summary['min_cosine']:.8f}")


def compare():
    infos = {e: graph_info(onnx_path(e)) for e in EXPORTERS}
    checks = {e: json.loads((common.RESULTS_DIR / f"stage2_export_check_{e}.json").read_text()) for e in EXPORTERS}
    print(f"{'':<16}" + "".join(f"{e:>14}" for e in EXPORTERS))
    for key in ("file_size_mb", "opset", "n_nodes", "external_data"):
        print(f"{key:<16}" + "".join(f"{str(infos[e][key]):>14}" for e in EXPORTERS))
    print(f"{'max_abs_diff':<16}" + "".join(f"{checks[e]['max_abs_diff']:>14.2e}" for e in EXPORTERS))
    print(f"{'min_cosine':<16}" + "".join(f"{checks[e]['min_cosine']:>14.8f}" for e in EXPORTERS))
    print("\nOperation counts (op: dynamo / legacy):")
    for op in sorted(set().union(*(i["op_counts"] for i in infos.values()))):
        print(f"  {op:<22} " + " / ".join(str(infos[e]["op_counts"].get(op, 0)) for e in EXPORTERS))


def bench(exporter):
    common.set_seeds()
    session = make_session(onnx_path(exporter))
    x = np.random.default_rng(common.SEED).standard_normal(common.ENCODER_INPUT_SHAPE).astype(np.float32)
    lat = common.benchmark(lambda: session.run(None, {"image": x}))
    mem = common.peak_memory_mb()
    print(f"[{exporter}] ONNX Runtime CPU latency: {lat}\nPeak process memory: {mem} MB, power: {common.power_state()}")
    (common.RESULTS_DIR / f"stage2_bench_{exporter}.json").write_text(
        json.dumps({**lat, "peak_mem_mb": mem, "power": common.power_state()}, indent=2), encoding="utf-8")


def evaluate(exporter):
    common.set_seeds()
    run_name = f"stage2_onnxruntime_fp32_cpu_{exporter}"
    bench_file = common.RESULTS_DIR / f"stage2_bench_{exporter}.json"
    lat = json.loads(bench_file.read_text()) if bench_file.exists() else {}
    predictor, segmenter = load_predictor()
    session = make_session(onnx_path(exporter))
    common.swap_encoder(predictor, lambda x: session.run(None, {"image": x})[0])
    rows, seconds = common.run_and_score(predictor, segmenter, run_name)
    (common.RESULTS_DIR / f"{run_name}_per_image.json").write_text(
        json.dumps({"rows": rows, "seconds_per_volume": seconds}, indent=2), encoding="utf-8")
    for thr in common.IOU_THRESHOLDS:
        acc = common.summarise_accuracy(rows, thr)
        print(f"IoU >= {thr}: {acc}")
        common.append_result({
            "stage": "2", "run": run_name, "device": "CPU", "runtime": "ONNX Runtime",
            "precision": "FP32", "encoder_size_mb": common.file_size_mb(onnx_path(exporter)),
            **{k: v for k, v in lat.items() if k not in ("peak_mem_mb", "power")},
            "peak_mem_encoder_mb": lat.get("peak_mem_mb"), "peak_mem_pipeline_mb": common.peak_memory_mb(),
            "pipeline_s_total": round(sum(seconds.values())), "iou_threshold": thr, **acc,
            "threads": "ORT default", "versions": common.software_versions(VERSIONS),
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["export", "compare", "bench", "eval"])
    parser.add_argument("--exporter", choices=EXPORTERS, default="dynamo")
    args = parser.parse_args()
    if args.mode == "compare":
        compare()
    else:
        {"export": export, "bench": bench, "eval": evaluate}[args.mode](args.exporter)
