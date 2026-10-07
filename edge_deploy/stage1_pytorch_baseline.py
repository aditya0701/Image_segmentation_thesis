"""
Stage 1: PyTorch FP32 baseline on CPU.

1. Encoder benchmark: the MicroSAM Base image encoder alone, on a fixed
   1x3x1024x1024 input (10 warm-up runs, 30 timed runs, batch size 1).
   Included: the encoder forward pass only. Not included: preprocessing,
   prompt encoder, mask decoder, post-processing.
2. Accuracy: the full thesis pipeline on the 2 test volumes, scored with the
   thesis evaluation code. End-to-end time per volume is measured once
   (one run, not 30, because a volume takes minutes).

Every later stage replaces only the encoder and is compared against this.

Usage:
    D:/my_envs/sam-edge/python.exe edge_deploy/stage1_pytorch_baseline.py
"""
import json

import torch
from micro_sam.automatic_segmentation import get_predictor_and_segmenter

import common

RUN_NAME = "stage1_pytorch_fp32_cpu"


def main():
    common.set_seeds()
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}, power: {common.power_state()}")

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=common.MODEL_TYPE, checkpoint=str(common.CHECKPOINT),
        device="cpu", is_tiled=True,
    )
    encoder = predictor.model.image_encoder.eval()
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {n_params / 1e6:.1f} M")

    # 1. Encoder-only latency. The input content does not change FP32 speed,
    #    so a fixed random tensor (seeded) is used.
    x = torch.randn(common.ENCODER_INPUT_SHAPE)
    with torch.inference_mode():
        out = encoder(x)
        print(f"Encoder output shape: {tuple(out.shape)}")
        lat = common.benchmark(lambda: encoder(x))
    mem_after_encoder = common.peak_memory_mb()
    print(f"Encoder latency: {lat}")
    print(f"Peak process memory after encoder benchmark: {mem_after_encoder} MB")

    # 2. Full pipeline on the test volumes.
    print("Running full pipeline on test volumes...")
    rows, seconds = common.run_and_score(predictor, segmenter, RUN_NAME)
    mem_after_pipeline = common.peak_memory_mb()

    (common.RESULTS_DIR / f"{RUN_NAME}_per_image.json").write_text(
        json.dumps({"rows": rows, "seconds_per_volume": seconds}, indent=2), encoding="utf-8")

    for thr in common.IOU_THRESHOLDS:
        acc = common.summarise_accuracy(rows, thr)
        print(f"IoU >= {thr}: {acc}")
        common.append_result({
            "stage": "1", "run": RUN_NAME, "device": "CPU", "runtime": "PyTorch",
            # Encoder weights only (4 bytes per FP32 parameter), comparable to later .onnx/engine files.
            "precision": "FP32", "encoder_size_mb": round(n_params * 4 / 1e6),
            **lat, "peak_mem_encoder_mb": mem_after_encoder,
            "peak_mem_pipeline_mb": mem_after_pipeline,
            "pipeline_s_total": round(sum(seconds.values())),
            "iou_threshold": thr, **acc,
            "threads": torch.get_num_threads(), "versions": common.software_versions(),
        })
    print(f"Results appended to {common.RESULTS_CSV}")


if __name__ == "__main__":
    main()
