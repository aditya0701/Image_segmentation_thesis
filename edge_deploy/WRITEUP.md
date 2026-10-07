# Deploying the MicroSAM Base encoder: ONNX, INT8, OpenVINO and TensorRT

**Goal.** Take the image encoder of a fine-tuned MicroSAM Base model (ViT-B, about 90 M parameters,
input 1x3x1024x1024) and measure what ONNX, quantization and OpenVINO really buy, without
changing the segmentation result.

**Hardware.** HP OmniBook Ultra Flip 14: Core Ultra 7 258V (Lunar Lake), Arc 140V iGPU, NPU,
32 GB RAM, Windows 11, AC power, "Best performance", package power limit 17 W.
Software: PyTorch 2.13, ONNX Runtime 1.30, OpenVINO 2026.4.1, NNCF 3.4.0.

**Method.** Batch 1, 10 warm-up runs, 30 timed runs, median with p10/p90, encoder only
(pipeline times are stated separately). The benchmark code refuses to run unless the laptop is on
AC in "Best performance". INT8 calibration used 57 tiles from the 19 training volumes only
(hash-checked against the test set). Accuracy: full tiled pipeline on 33 ground-truth boutons from
two test volumes. A detection counts as a match at IoU >= 0.25, the threshold used for this
project; IoU 0.5 is shown as a stricter check.

## Results (encoder, median latency)

| Device | Runtime | Precision | Latency | Speedup vs baseline | Size | Min cosine vs FP32 | TP/FP/FN at IoU 0.25 | TP/FP/FN at IoU 0.5 (stricter check) |
|---|---|---|---|---|---|---|---|---|
| CPU | PyTorch | FP32 (baseline) | 5,605 ms | 1.00x | 359 MB | 1 | 32/12/1 | 30/14/3 |
| CPU | ONNX Runtime | FP32 | 3,772 ms | 1.49x | 368 MB | 1.0 (max diff 6e-6) | 32/12/1 | 30/14/3 |
| CPU | ONNX Runtime | INT8 dynamic | 2,079 ms | 2.70x | 109 MB | 0.9996 | 32/12/1 | 30/14/3 |
| CPU | ONNX Runtime | INT8 static, MatMul only | 2,399 ms | 2.34x | 110 MB | 0.989 | 32/10/1 | 31/11/2 |
| CPU | ONNX Runtime | INT8 static, all ops | 3,063 ms | 1.83x | 101 MB | 0.989 | 31/9/2 | 29/11/4 |
| CPU | OpenVINO | FP32 | 3,840 ms | 1.46x | 368 MB | not recorded | not run | not run |
| CPU | OpenVINO | INT8 (NNCF) | 1,889 ms | 2.97x | 110 MB | 0.9998 | 32/8/1 | 30/10/3 |
| GPU | OpenVINO | FP32 | 624 ms | 8.98x | 368 MB | not recorded | not run | not run |
| GPU | OpenVINO | **FP16** | **241 ms** | **23.3x** | 184 MB | 0.999997 | 32/12/1 | 30/14/3 |
| GPU | OpenVINO | INT8 (QDQ from ONNX) | 680 ms | 8.24x | 110 MB | 0.988 | 32/10/1 | 31/11/2 |
| GPU | OpenVINO | INT8 (NNCF, direct route) | 592 ms | 9.47x | 101 MB | 0.9998 | 32/8/1 | 30/10/3 |
| GPU | OpenVINO | INT8 linear layers only | 465 ms | 12.1x | not recorded | 0.9998 | not run | not run |
| GPU | OpenVINO | INT8 weights only | 371 ms | 15.1x | not recorded | 0.99998 | not run | not run |
| NPU | OpenVINO | FP16, INT8 | failed | none | | | | |

Sources: `results/results.csv`, `results/stage5_bench_*.json`, `results/stage5_check_*.json`.

## TensorRT 11 on an NVIDIA T4 (Google Colab, a different machine)

Only ratios inside this table are comparable. Absolute times must not be compared with the laptop rows.
Tesla T4, TensorRT 11.3.0.99, torch 2.11 (for CUDA events), batch 1, 10 warm-up and 30 timed runs.
Time is GPU execution only (CUDA events); host-to-device copies are not included. GPU clocks stayed at
1500 to 1590 MHz (maximum 1590), so no throttling. Accuracy is the cosine against the FP32 ONNX Runtime
output on 5 real test tiles (numerical agreement only; no detections were scored for these engines).

| Precision | Latency (p10 to p90) | Speedup vs T4 FP32 | Engine size | Layers in that precision | Min cosine | Worst max diff |
|---|---|---|---|---|---|---|
| FP32 | 314.5 ms (308.9 to 318.9) | 1.00x | 370 MB | n/a | 1.00000 | 7.6e-6 |
| FP16 | 108.3 ms (106.7 to 109.3) | 2.90x | 200 MB | 226 of 228 half | 0.99998 | 0.035 |
| INT8 (symmetric QDQ, MatMul/Gemm/Conv) | 98.7 ms (98.2 to 99.5) | 3.18x (1.10x vs FP16) | 112 MB | 194 of 226 INT8 | 0.98447 | 0.401 |

Sources: `results/stage4_trt_all.json`, `results/stage4_trt_real_tile_check.json`, `stage4_tensorrt_colab.ipynb`.

## Findings

1. **ONNX export alone gives 1.49x on CPU** with voxel-identical masks (max output difference 6.35e-6).
   The gain comes from graph optimisation (fusion, constant folding), not from lower precision.
2. **Dynamic INT8 is the best ONNX Runtime CPU option**: 2.70x vs PyTorch, 3.4x smaller than ONNX FP32,
   same detections as FP32 at both thresholds. Static INT8 was less accurate (cosine 0.989) and
   quantizing every operator (LayerNorm, Softmax, Add) was slower and worse than quantizing only
   the matrix multiplications.
3. **The integrated GPU in FP16 is the clear winner**: 241 ms, 23x faster than the PyTorch CPU baseline,
   detections identical to FP32. Energy per encoder inference (HWiNFO package power, 60 s windows)
   is about 5 J vs 66 J for CPU FP32, about 13x less.
4. **INT8 did not beat FP16 on the Arc GPU.** Every INT8 variant was slower (371 to 680 ms vs 241 ms).
   Per-layer profiling showed the attention matrix multiplications get no INT8 kernel and the
   surrounding Add/Multiply on the 4096x4096 attention maps fall back to FP32. Weight-only INT8
   was also slower, consistent with a compute-bound encoder (about 4,096 patches reuse every weight).
5. **Amdahl's law limits the end-to-end gain.** The full two-volume pipeline went from 5,941 s to
   2,248 s (2.6x) with the GPU FP16 encoder, because the encoder is only about 14% of GPU pipeline time.
6. **Failures are results too.** The NNCF INT8 model built from ONNX fails to compile on the Arc GPU
   (`CL_BUILD_PROGRAM_FAILURE -11`, 3 of 3 fresh runs); converting from PyTorch directly works.
   The NPU compiled but produced wrong output or NaN, then `ZE_RESULT_ERROR_DEVICE_LOST`, also after
   a driver update. The cause is untested (watchdog hypothesis).

7. **On the T4, FP16 is 2.9x faster than FP32 and INT8 is only 1.10x faster than FP16**, with clearly lower
   agreement (cosine 0.984 vs 0.99998). The INT8 engine agrees with the same model run in ONNX Runtime
   (0.984), so the loss comes from the symmetric MinMax quantization, not from TensorRT. FP16 is the better
   choice on this GPU too. A likely reason for the small INT8 gain (not tested): every layer that is not
   INT8 (LayerNorm, Softmax, GELU, Adds) stays in FP32 in this model, and FP32 is slow on a T4.
8. **TensorRT 11 changed the workflow.** The `FP16` and `INT8` builder flags are gone and every network is
   strongly typed, so precision has to be inside the ONNX file. `trtexec` is not in the pip package.
   The Stage 3 QDQ model was rejected twice: Int32 bias `DequantizeLinear` nodes, and non-zero activation
   zero points. A new symmetric model with float biases (`stage3_ort_int8.py quantize --variant trt_matmul`)
   was needed. ModelOpt AutoCast ran out of RAM on Colab and was replaced by `onnxconverter-common`.

## Limits of this study

- One laptop, one model, 33 ground-truth objects from two volumes. Differences of one or two
  detections (for example fewer false positives with NNCF INT8) are too small to call an accuracy
  improvement.
- The pipeline time of the ONNX FP32 run partly overlapped a Modern Standby period and is not used
  for any timing claim.
- Energy comes from a software sensor (HWiNFO, 2 s sampling), not a power meter.
- These are laptop CPU, GPU and NPU results, not measurements on an embedded edge device. The T4 numbers
  come from a shared Colab machine and cover GPU execution only.
- TensorRT engines were checked on 5 tiles for numerical agreement only. Detections (TP/FP/FN) were not
  scored for them, and the INT8 model used for TensorRT is a different quantization (symmetric) from the
  ONNX Runtime and OpenVINO INT8 rows.

## Not done yet

- Detections for the TensorRT engines (would need the engine inside the full pipeline).
- INT8 + FP16 mixed engine on the T4, to test why INT8 gains so little.
- NPU watchdog tests, percentile/entropy calibration, C++ inference program.
