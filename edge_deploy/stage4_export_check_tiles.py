"""
Stage 4 helper: save the 5 real test tiles used by the Stage 3 and Stage 5 'check' modes,
together with the FP32 ONNX Runtime reference embeddings, in one file.

Upload the .npz to Google Drive. On Colab, each TensorRT engine encodes the same tiles and
is compared with the reference (cosine, max abs diff), so the T4 numbers are directly
comparable with the CPU and OpenVINO rows. The tiles are test data used only for checking
numerical agreement, never for calibration.

Usage:
    D:/my_envs/sam-edge/python.exe edge_deploy/stage4_export_check_tiles.py
"""
import sys

import numpy as np

import common
from stage2_onnx_export import load_predictor, make_session, onnx_path

sys.stdout.reconfigure(encoding="utf-8")

OUT = common.OUTPUT_ROOT / "trt_check_tiles.npz"


def main():
    predictor, _ = load_predictor()
    session = make_session(onnx_path("dynamo"))
    inputs = [x.numpy().astype(np.float32) for x in common.real_encoder_inputs(predictor)]
    refs = [session.run(None, {"image": x})[0].astype(np.float32) for x in inputs]
    np.savez(OUT, images=np.concatenate(inputs), reference=np.concatenate(refs))
    print(f"{OUT.name}: images {np.concatenate(inputs).shape}, reference {np.concatenate(refs).shape}, "
          f"{OUT.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
