# Edge deployment project: MicroSAM Base encoder (handover context)

Read this first. It replaces the long earlier chat (session 9ed4c6f4, ended 2026-10-07).
Full old transcript, only if a detail is missing:
`C:\Users\adima\.claude\projects\d--Personal-Projects-Image-segmentation-thesis\9ed4c6f4-e598-4781-ad00-5b8bae6f43c8.jsonl`

## Who and why
Aditya (M.Sc. Data Science, RWTH Aachen; strong Python/PyTorch; fine-tuned MicroSAM in his thesis)
is learning ONNX, quantization, TensorRT and OpenVINO through one small, honest project, written up
as a case study for automated driving, robotics and edge-AI jobs. Every CV claim must be backed by a
measured result.

## Working rules (follow exactly)
- Claude writes the code, Aditya reviews it. Teach as you go: before each step, 2 to 3 sentences on
  what it does and why, define new terms, explain each tool from scratch (role and alternatives).
  Quiz at the end of each stage. Stop after each stage, summarise, wait for him.
- For future C++ or ROS 2 work: he writes the code, Claude tutors.
- Ask before downloading large files, installing drivers or changing system settings.
- When something fails: show the error and the diagnosis before trying fixes.
- Plain English. No em dashes anywhere (text, comments, write-ups).
- Small, readable, reproducible scripts: one per stage, results CSV, fixed seeds (SEED 0).
- Do not run benchmarks while he games or browses. Before timing: Firefox closed, AC power,
  "Best performance" (common.benchmark refuses otherwise). Screen-off on AC is set to Never
  (screen-off triggers Modern Standby on this laptop).
- git push needs him (auth fails from Claude). Commit only when asked.

## Measurement and honesty rules
- Batch 1, warm-up runs, at least 30 timed runs, report median and p10/p90, state what is included.
- Record all software and hardware versions. Never claim an unmeasured improvement.
- Report it when INT8 hurts accuracy or speed. Say "on laptop CPU, GPU and NPU", never "edge device".
- Calibration data only from training data, never test.
- IoU threshold decided 2026-10-07: 0.25 is the headline number. Still score 0.5 as a stricter check.

## Setup
- Model: MicroSAM Base `vit_b_lm`, fine-tuned checkpoint MicroSAM_B_all. Image encoder only,
  fixed input 1x3x1024x1024.
- This laptop (Laptop B, DragoX): HP OmniBook Ultra Flip 14, Core Ultra 7 258V, Arc 140V, NPU,
  32 GB RAM, Windows 11. PL1 17 W. Drivers: GPU 32.0.101.8626, NPU 32.0.100.5540.
- Laptop A (GTX 1650 Ti) is NOT needed. Stage 4 runs on Google Colab (free T4).
- Env: conda `D:\my_envs\sam-edge` (python 3.11, micro_sam 1.8.14, torch 2.13 CPU, numpy 2,
  onnxruntime 1.30, openvino, nncf). Run scripts with `D:/my_envs/sam-edge/python.exe`.
  Windows console is cp1252: scripts call `sys.stdout.reconfigure(encoding="utf-8")`.
- Repo: github.com/aditya0701/Image_segmentation_thesis, branch `edge-deploy` (pushed at 8d0c5f7).
- Data and paths (see common.py):
  - Test: `D:\Thesis\Thesis_Images\test\test`. Brain1 Airyscan (19 GT) + Brain2 PSF deconv
    (14 GT, mask channel 0) = 33 boutons.
  - Train: `D:\Thesis\Thesis_Images\train\train` (19 volumes). Calibration: 57 tiles at
    `D:\Thesis\edge_results\calibration\calib_tiles.npy` (sha256 guard against test overlap).
  - Checkpoint: `D:\Thesis\microsam_B_all_checkpoints\best_weights_only.pt`.
  - Models: `D:\Thesis\edge_results\models` (ONNX, INT8 ONNX, `openvino\` IRs).
  - Results: `edge_deploy/results/` (results.csv plus per-run JSON and logs).
- Pipeline settings: tile (896,896), halo (64,64), blob removal threshold 862.

## Scripts (edge_deploy/)
common.py (paths, pipeline, scoring, benchmark with power guard, swap_encoder, versions),
stage0_env_check.py, check_imports.py, extract_weights.py, stage1_pytorch_baseline.py,
stage2_onnx_export.py, make_calibration_set.py, stage3_ort_int8.py, stage5_openvino.py,
stage5_power_analysis.py (HWiNFO CSV), stage5_gpu_int8_debug.py, stage5_gpu_profile.py,
stage4_export_check_tiles.py (5 real tiles + FP32 reference for Colab), stage4_tensorrt_colab.ipynb
(clean Colab notebook; stage4_session_log.ipynb is the raw session). Docs: WRITEUP.md, CLAIMS.md.

## Results so far (encoder latency = median, batch 1)
- Stage 1, PyTorch FP32 CPU: 5,605 ms (p10 5,179, p90 6,155), 359 MB, peak 3.7 GB encoder /
  8.6 GB pipeline. IoU 0.25: TP/FP/FN 32/12/1, recall 0.970, mIoU 0.754. IoU 0.5: 30/14/3,
  recall 0.909, mIoU 0.777. Pipeline 5,941 s.
- Stage 2, ONNX (dynamo exporter, opset 20, 368 MB; legacy kept for comparison): max diff 6.35e-6,
  ORT CPU 3,772 ms (1.49x), masks voxel-identical to Stage 1.
- Stage 3, ORT INT8 CPU: dynamic 2,079 ms, 109 MB, cos 0.9996, 32/12/1, pipeline 3,204 s.
  Static full 3,063 ms, cos 0.989, 31/9/2 (worse recall). Static MatMul-only 2,399 ms, 32/10/1.
  ORT 1.30 bug: CalibMaxIntermediateOutputs discards ranges, so it is not used.
- Stage 5, OpenVINO (AC, 17 W):
  - CPU: fp32 3,840, fp16 3,770, NNCF INT8 1,889 ms (cos 0.99982, eval 32/8/1), QDQ 2,498 ms.
  - GPU: fp32 624, fp16 241 ms (cos 1.0, eval identical to FP32, pipeline 2,248 s = 2.6x),
    QDQ 680 ms. Encoder is only about 14% of GPU pipeline time (Amdahl).
  - Energy (HWiNFO package power): GPU fp16 about 5 J per inference vs CPU fp32 66 J.
  - GPU INT8: ONNX-route NNCF model fails to compile (CL_BUILD_PROGRAM_FAILURE -11, every time).
    Direct route (PyTorch -> ov.convert_model -> NNCF) compiles: 592 ms, 12.0 J, eval 32/8/1.
    Profile: attention MatMul gets no fast kernel, Add/Multiply fall back to f32.
    2026-10-07 bench: fp16 240.9, INT8 weights-only 370.8, INT8 linear-only 465.0 ms.
    Conclusion: no INT8 variant beats FP16 on the Arc GPU (encoder is compute-bound).
  - NPU: compiles, then wrong output or NaN, then ZE_RESULT_ERROR_DEVICE_LOST, also after the
    driver update. Documented as a failure (watchdog/TDR hypothesis untested).
  - Session 1 and 2 raw files archived in results/stage5_session1/ and stage5_session2/.

- Stage 4, TensorRT 11.3 on Colab T4 (GPU execution only, ratios within the T4 only):
  FP32 314.5 ms, FP16 108.3 ms (2.90x, cos 0.99998), INT8 98.7 ms (3.18x, 1.10x vs FP16, cos 0.984).
  TensorRT 11 has no FP16/INT8 flags (strongly typed only). FP16 = onnxconverter-common conversion.
  INT8 needs a symmetric QDQ model with float biases: `stage3_ort_int8.py quantize --variant trt_matmul`
  (the old static_matmul model was rejected: Int32 bias DQ, non-zero zero points). ModelOpt AutoCast ran
  out of RAM on Colab. No detections scored for TensorRT engines (cosine on 5 tiles only).

## Next steps
1. Optional T4 experiment: INT8 + FP16 mixed engine (non-INT8 layers are FP32 now; hypothesis for the
   small INT8 gain). Optional: TensorRT engine inside the full pipeline for TP/FP/FN.
2. Keep WRITEUP.md and CLAIMS.md in step with any new result.
3. Optional: NPU watchdog tests, OpenVINO bug report for the ONNX-route GPU INT8 failure,
   Percentile/Entropy calibration, later a C++ inference program (he codes) and use of the
   optimised encoder in his napari bouton viewer.
