"""
Stage 5 follow-up: why does the NNCF INT8 model fail to compile on the Arc GPU
(CL_BUILD_PROGRAM_FAILURE), and can it be made to work?

Modes:
  retry    Compile the existing NNCF INT8 model on the GPU 3 times, each in a fresh
           process (no model cache), to rule out a one-off failure.
  direct   Skip ONNX: PyTorch encoder -> OpenVINO (ov.convert_model) -> NNCF INT8 with
           the same settings and calibration tiles -> compile on GPU, check accuracy.

Usage:
    D:/my_envs/sam-edge/python.exe edge_deploy/stage5_gpu_int8_debug.py retry
"""
import argparse
import json
import subprocess
import sys
import time

import numpy as np
import openvino as ov

import common
from make_calibration_set import CALIB_FILE
from stage5_openvino import OV_DIR, ir_path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

LOG = common.RESULTS_DIR / "stage5_gpu_int8_debug.json"


def save(key, value):
    data = json.loads(LOG.read_text()) if LOG.exists() else {}
    data[key] = value
    LOG.write_text(json.dumps(data, indent=2), encoding="utf-8")


def try_compile(xml_path, device="GPU"):
    """Compile in a fresh Python process, so nothing is reused between attempts."""
    code = (f"import openvino as ov, time; t=time.perf_counter(); "
            f"ov.Core().compile_model(r'{xml_path}', '{device}', "
            f"{{'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': 'f16'}}); "
            f"print('COMPILED in %.1f s' % (time.perf_counter()-t))")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip()
    ok = "COMPILED" in out
    first_error = next((line for line in out.splitlines() if "error code" in line or "Error" in line), "")
    return ok, out.splitlines()[-1] if ok else first_error


def retry():
    results = []
    for i in range(3):
        ok, msg = try_compile(ir_path("int8_nncf"))
        results.append({"attempt": i + 1, "compiled": ok, "message": msg})
        print(f"attempt {i + 1}: {'OK' if ok else 'FAILED'}  {msg}")
    save("retry_existing_nncf_int8", results)


def direct():
    import nncf
    import torch
    from stage2_onnx_export import load_predictor

    predictor, _ = load_predictor()
    encoder = predictor.model.image_encoder.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        model = ov.convert_model(encoder, example_input=torch.randn(common.ENCODER_INPUT_SHAPE))
    fp32_xml = OV_DIR / "sam_vitb_encoder_fp32_direct.xml"
    ov.save_model(model, str(fp32_xml), compress_to_fp16=False)
    print(f"Direct conversion (no ONNX): {time.perf_counter() - t0:.0f} s, {len(model.get_ops())} ops")

    tiles = np.load(CALIB_FILE, mmap_mode="r")
    dataset = nncf.Dataset(range(len(tiles)), lambda i: np.ascontiguousarray(tiles[i][None], dtype=np.float32))
    q = nncf.quantize(model, dataset, subset_size=len(tiles), model_type=nncf.ModelType.TRANSFORMER)
    int8_xml = OV_DIR / "sam_vitb_encoder_int8_nncf_direct.xml"
    ov.save_model(q, str(int8_xml), compress_to_fp16=False)

    gpu_ok, gpu_msg = try_compile(int8_xml, "GPU")
    cpu_ok, cpu_msg = try_compile(int8_xml, "CPU")
    print(f"GPU compile: {'OK' if gpu_ok else 'FAILED'}  {gpu_msg}\nCPU compile: {'OK' if cpu_ok else 'FAILED'}  {cpu_msg}")
    save("direct_pytorch_to_openvino_nncf_int8", {"gpu_compiled": gpu_ok, "gpu_message": gpu_msg,
                                                   "cpu_compiled": cpu_ok, "cpu_message": cpu_msg})


FP32_BY_ROUTE = {"onnx": ir_path("fp32"), "direct": OV_DIR / "sam_vitb_encoder_fp32_direct.xml"}


def attention_matmuls(model):
    """MatMuls where neither input is a weight (Constant, possibly behind Convert/
    Transpose/Reshape/Unsqueeze/Multiply): query x key and attention x value.
    The GPU profile showed these lose their fast kernel in INT8 and drag the
    surrounding element-wise ops on the 4096 x 4096 attention maps to FP32."""
    def from_constant(out):
        n = out.get_node()
        while n.get_type_name() in ("Convert", "Transpose", "Reshape", "Unsqueeze", "Multiply"):
            n = n.input_value(0).get_node()
        return n.get_type_name() == "Constant"
    return [op.get_friendly_name() for op in model.get_ops() if op.get_type_name() == "MatMul"
            and not any(from_constant(op.input_value(i)) for i in range(2))]


def fc_only(route):
    """NNCF INT8 for the linear layers only; attention MatMuls stay FP16."""
    import nncf
    model = ov.Core().read_model(str(FP32_BY_ROUTE[route]))
    ignored = attention_matmuls(model)
    print(f"[{route}] excluding {len(ignored)} attention MatMuls from quantization")
    tiles = np.load(CALIB_FILE, mmap_mode="r")
    dataset = nncf.Dataset(range(len(tiles)), lambda i: np.ascontiguousarray(tiles[i][None], dtype=np.float32))
    q = nncf.quantize(model, dataset, subset_size=len(tiles), model_type=nncf.ModelType.TRANSFORMER,
                      ignored_scope=nncf.IgnoredScope(names=ignored))
    out = OV_DIR / f"sam_vitb_encoder_int8_fc_{route}.xml"
    ov.save_model(q, str(out), compress_to_fp16=False)
    ok, msg = try_compile(out, "GPU")
    print(f"[{route}] GPU compile: {'OK' if ok else 'FAILED'}  {msg}")
    save(f"fc_only_{route}", {"ignored_attention_matmuls": len(ignored), "gpu_compiled": ok, "gpu_message": msg})


def weights_only():
    """INT8 weight compression only: weights stored INT8, all maths in FP16."""
    import nncf
    model = ov.Core().read_model(str(ir_path("fp32")))
    q = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)
    out = OV_DIR / "sam_vitb_encoder_int8_weights.xml"
    ov.save_model(q, str(out), compress_to_fp16=False)
    ok, msg = try_compile(out, "GPU")
    print(f"weights-only INT8 GPU compile: {'OK' if ok else 'FAILED'}  {msg}")
    save("weights_only_int8", {"gpu_compiled": ok, "gpu_message": msg})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["retry", "direct", "fc_only", "weights_only"])
    parser.add_argument("--route", choices=["onnx", "direct"], default="onnx")
    args = parser.parse_args()
    if args.mode == "fc_only":
        fc_only(args.route)
    else:
        {"retry": retry, "direct": direct, "weights_only": weights_only}[args.mode]()
