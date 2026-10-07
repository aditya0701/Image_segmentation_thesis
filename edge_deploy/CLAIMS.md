# CLAIMS: what may and may not be said about this project

Every claim on the left is backed by a file in `edge_deploy/results/`. Keep claims exactly this big.

## May claim (measured)

| Claim | Evidence |
|---|---|
| ONNX Runtime FP32 is 1.49x faster than PyTorch FP32 for the encoder on this laptop CPU, with identical masks | `results.csv` stage 2, `stage2_compare.txt` |
| Dynamic INT8 (ONNX Runtime) runs the encoder in 2,079 ms vs 5,605 ms (2.7x), with a 109 MB model vs 359 MB, and the same 32/12/1 detections at IoU 0.25 | `results.csv` stage 3 |
| OpenVINO FP16 on the Arc 140V iGPU runs the encoder in 241 ms (23x vs the PyTorch CPU baseline) with detections identical to FP32 | `stage5_bench_GPU_fp16.json`, `results.csv` |
| The full pipeline on two volumes was 2.6x faster (5,941 s to 2,248 s) with the GPU FP16 encoder, limited by Amdahl's law | `results.csv` stage 1 and 5 |
| Encoder energy was about 5 J per inference on GPU FP16 vs 66 J on CPU FP32 (software sensor, HWiNFO package power) | `stage5_energy.csv`, session 1 and 2 |
| On the Arc 140V with OpenVINO 2026.4.1, no INT8 variant I tried beat FP16 (371 to 680 ms vs 241 ms) | `stage5_bench_GPU_*.json` |
| The cause, from per-layer profiling: attention MatMuls get no INT8 kernel, Add/Multiply fall back to FP32 | `stage5_gpu_profile.json` |
| NNCF INT8 on CPU: 1,889 ms (2.97x vs PyTorch), cosine 0.9998 | `stage5_bench_CPU_int8_nncf.json` |
| The ONNX-route NNCF INT8 model fails to compile on this GPU (3 of 3 fresh runs); the direct PyTorch route compiles | `stage5_failure_GPU_int8_*_compile.txt` |
| The NPU compiled the model but returned wrong output or NaN, then device lost, also after a driver update | `stage5_check_NPU_*_log.txt`, `stage5_before_driver_update/` |
| Calibration used only training tiles (hash-checked against the test set) | `calibration_log.txt` |
| On a Colab T4 with TensorRT 11.3, the encoder ran in 314.5 ms (FP32), 108.3 ms (FP16, 2.9x) and 98.7 ms (INT8, 3.2x vs FP32, 1.10x vs FP16), GPU execution only, no clock throttling | `stage4_trt_all.json` |
| The T4 FP16 engine agrees with FP32 (cosine 0.99998) and the INT8 engine less so (0.984), on 5 real test tiles; the INT8 engine matches the same model in ONNX Runtime | `stage4_trt_real_tile_check.json`, `stage3_check_trt_matmul.json` |
| TensorRT 11 removed the FP16/INT8 builder flags and requires strongly typed networks; the Stage 3 QDQ model was rejected (Int32 bias DequantizeLinear, non-zero zero points) and had to be re-quantized symmetrically | `stage4_session_log.ipynb`, `stage3_quantize_trt_matmul_log.txt` |

## May NOT claim

- "Edge device" or "embedded" results. Say "on a laptop CPU, GPU and NPU".
- That INT8 is faster or better in general. It was faster on CPU and slower on this GPU.
- That INT8 improved accuracy or reduced false positives. A difference of 2 to 4 false positives
  on 33 objects is noise at this sample size.
- Any NPU speedup or NPU numbers. There are none.
- That the NPU failure is caused by a watchdog timeout. That is a hypothesis, untested.
- The pipeline time of the ONNX FP32 run (stage 2) as a timing result. It overlapped Modern Standby.
- Energy in joules as a precise value. It is a software estimate with 2 s sampling.
- That TensorRT INT8 is "fast" or "better": it was 10% faster than FP16 on the T4 and clearly less accurate.
- Detection results (TP/FP/FN) or segmentation quality for any TensorRT engine. Only cosine on 5 tiles exists.
- That the TensorRT INT8 model equals the ONNX Runtime or OpenVINO INT8 models. It is a different,
  symmetric quantization with float biases.
- Why INT8 gained little on the T4 (FP32 leftovers are a hypothesis, not tested).
- That "T4 is faster than the Arc GPU" or any T4 vs laptop comparison of absolute times. Different machines.
- T4 times as end-to-end latency: they exclude host-to-device copies and pre/post-processing.
- Results for other models, other laptops or other drivers.
- Accuracy numbers without naming the threshold. The project threshold is IoU 0.25; IoU 0.5 is a
  stricter check. Never switch thresholds between rows to make a variant look better.
- Absolute latency from a different machine (Colab T4) as comparable to the laptop numbers.
  Only ratios within one machine may be compared.
