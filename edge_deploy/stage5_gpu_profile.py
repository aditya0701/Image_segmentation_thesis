"""
Stage 5: per-layer GPU profile, FP16 vs INT8 (int8_nncf_direct).

OpenVINO's performance counters report, for every executed layer, its type, the
kernel that ran it (exec_type, which includes the precision) and its GPU time.
Summing these by kind shows where INT8 loses time: conversions/reorders,
quantize steps, or matrix multiplications that did not get INT8 kernels.

Usage:
    D:/my_envs/sam-edge/python.exe edge_deploy/stage5_gpu_profile.py
"""
import json
import sys
from collections import defaultdict

import numpy as np
import openvino as ov

import common
from stage5_openvino import compute_precision, model_file

sys.stdout.reconfigure(encoding="utf-8")

RUNS = 10   # profiles are averaged over 10 inferences after 3 warm-ups


def profile(variant):
    core = ov.Core()
    compiled = core.compile_model(str(model_file(variant)), "GPU",
                                  {"PERFORMANCE_HINT": "LATENCY", "PERF_COUNT": True,
                                   "INFERENCE_PRECISION_HINT": compute_precision("GPU", variant)})
    request = compiled.create_infer_request()
    x = np.random.default_rng(common.SEED).standard_normal(common.ENCODER_INPUT_SHAPE).astype(np.float32)
    for _ in range(3):
        request.infer({0: x})
    totals = defaultdict(float)          # (node_type, exec_type) -> microseconds
    for _ in range(RUNS):
        request.infer({0: x})
        for p in request.get_profiling_info():
            if p.status == ov.ProfilingInfo.Status.EXECUTED:
                totals[(p.node_type, p.exec_type)] += p.real_time.total_seconds() * 1e6 / RUNS
    return totals


def summarise(variant, totals):
    total_ms = sum(totals.values()) / 1000
    by_type = defaultdict(float)
    for (node_type, _), us in totals.items():
        by_type[node_type] += us
    print(f"\n=== GPU {variant}: {total_ms:.1f} ms summed over executed layers")
    for node_type, us in sorted(by_type.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {node_type:<28} {us / 1000:8.1f} ms  {100 * us / 1000 / total_ms:5.1f}%")
    print("  -- top kernels (node type | kernel):")
    for (node_type, exec_type), us in sorted(totals.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {us / 1000:8.1f} ms  {node_type:<22} | {exec_type}")
    return {"total_ms": round(total_ms, 1),
            "by_node_type_ms": {k: round(v / 1000, 2) for k, v in sorted(by_type.items(), key=lambda kv: -kv[1])},
            "by_kernel_ms": {f"{n} | {e}": round(v / 1000, 2) for (n, e), v in sorted(totals.items(), key=lambda kv: -kv[1])}}


def main():
    out = {}
    for variant in ("fp16", "int8_nncf_direct"):
        out[variant] = summarise(variant, profile(variant))
    (common.RESULTS_DIR / "stage5_gpu_profile.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
