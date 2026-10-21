# Learning notes: deploying the MicroSAM encoder (ONNX, INT8, OpenVINO)

A step-by-step study guide to what we did, why, what we measured, and what went wrong.
Use it in a fresh chat: paste one section and ask "quiz me on this" or "go deeper on X".
All numbers were measured on Laptop B (Core Ultra 7 258V, Arc 140V GPU, NPU, 32 GB RAM),
on AC power in "Best performance" mode, batch size 1.

---

## The big picture in one paragraph

A model trained in PyTorch is not yet a deployed model. Deployment means: freeze the model
into a portable graph (ONNX), let an inference engine optimise it for one piece of hardware
(ONNX Runtime, OpenVINO, TensorRT), optionally shrink the number format from 32-bit floats
to 16-bit floats or 8-bit integers (quantization), and then prove with measurements that it
is faster, smaller or cheaper in energy without breaking accuracy. We did this for the image
encoder of MicroSAM Base (a ViT-B, about 90 million parameters, input 1x3x1024x1024), because
it is the heavy part of the segmentation pipeline.

---

## Stage 0: a reproducible environment

**What we did:** created a conda environment (`sam-edge`) from a YAML file, recorded exact
package versions (lock files) and a hardware report as JSON.

**Concepts**
- *conda vs pip:* conda installs compiled libraries (vigra, nifty) that pip cannot; pip has
  the newest ML packages (onnxruntime, openvino, nncf). We used conda-forge first, then pip.
- *ABI break:* numpy 2 changed its binary interface. Packages compiled against numpy 1
  crash when numpy 2 is installed. That is exactly what broke our first environment: conda
  picked an old micro_sam built for numpy 1, then pip upgraded numpy. Fix: install everything
  numpy-dependent from conda against numpy 2.
- *Lock file:* the exact versions that worked, so the result can be reproduced later.

**Checkpoint lesson:** `best.pt` was a pickled training checkpoint that referenced Python
functions from the training script (`ChannelSafeIntensityAug` etc.). Unpickling needs those
objects to exist. We loaded it with placeholder classes and saved a weights-only file
(1,350 MB down to 419 MB), then verified the weights were bit-identical. Interview point:
pickles execute code and depend on code; ship `state_dict` or safetensors instead.

**Interview questions**
- Why record versions of every library in a benchmark?
- What is an ABI and why does a numpy major version break compiled packages?
- Why is `torch.load` on an untrusted pickle a security risk?

---

## Stage 1: the PyTorch FP32 baseline

**Why a baseline:** every later claim is "X times faster than this" or "same accuracy as
this". Without a careful baseline, nothing after it means anything.

**How to time a model correctly**
- *Warm-up runs (10):* the first runs are slow (memory allocation, caches, kernel selection).
- *Timed runs (30+), report the median and p10/p90:* the median ignores rare spikes; the
  spread shows how stable it is. The mean is pulled by outliers.
- *Batch size 1:* the realistic case for one image at a time.
- *State what is included:* encoder only, or the full pipeline with tiling and post-processing.
- *Control the power state:* our first run on battery gave 6.5 s instead of 5.6 s. The
  benchmark function now refuses to run unless the laptop is on AC in "Best performance".

**Results (CPU, PyTorch FP32)**
- Encoder: 5,605 ms median (p10 5,179, p90 6,155). Size 359 MB. Peak memory 3.7 GB.
- Accuracy on 33 ground-truth boutons:
  - IoU >= 0.25: 32 true positives, 12 false positives, 1 missed; recall 0.970, mIoU 0.754.
  - IoU >= 0.5: 30/14/3; recall 0.909, mIoU 0.777.
- Full pipeline on two volumes: 5,941 s.

**Concepts**
- *IoU (intersection over union):* overlap of predicted and true object divided by their
  union. A match counts if IoU is above a threshold; we report two thresholds.
- *Micro vs macro averaging:* pool all objects then compute (micro), or compute per image
  then average (macro). Both are reported because they differ when images have different
  numbers of objects.
- *Peak working set:* the most physical RAM the process used.

**Interview questions**
- Why median and not mean? Why warm-up?
- Why can mIoU go up when recall goes down? (Fewer, but better, matches.)
- What can make a laptop benchmark unreliable? (Power mode, thermals, background apps, sleep.)

---

## Stage 2: export to ONNX and verify

**What ONNX is:** a file format that describes a neural network as a graph of standard
operations (MatMul, Add, Softmax, ...) plus its weights. Any runtime that understands ONNX
can run it, independent of PyTorch.

**What we did**
- Exported the encoder twice: the new `torch.export` (dynamo) exporter and the legacy
  TorchScript tracer. Dynamo gave 564 nodes (opset 20), legacy 3,219 nodes (opset 17).
  Fewer nodes means a cleaner graph that is easier for runtimes to optimise.
- The weights are stored as *external data* (`.onnx.data`) because ONNX files use protobuf,
  which has a 2 GB limit, and external data is the standard pattern for big models.
- Verified the export: same input through PyTorch and ONNX Runtime, compared the outputs.
  Max absolute difference 6.35e-6, cosine similarity 1.0. Then ran the full pipeline: masks
  voxel-identical to PyTorch.

**Result:** ONNX Runtime on CPU: 3,772 ms vs 5,605 ms (1.49x faster) with identical output.
Why faster without changing precision: graph optimisations (operator fusion, constant
folding, better memory layout and threading) that eager PyTorch does not do.

**Concepts**
- *Opset:* the version of the ONNX operator set; newer opsets have more/better operators.
- *Constant folding:* precompute parts of the graph that only depend on weights.
- *Operator fusion:* merge several small ops into one kernel to save memory traffic.
- *Verification metrics:* max abs diff (worst element), mean abs diff, cosine similarity
  (direction of the whole output). Always verify on real inputs, not only random noise.

**Pitfall we hit:** the exporter printed an emoji and the Windows console (cp1252 encoding)
crashed. Fix: quiet the exporter and switch stdout to UTF-8.

**Interview questions**
- Tracing vs `torch.export`: what can go wrong with tracing (data-dependent control flow)?
- How do you prove an exported model is correct? Which tolerance is "good enough"?
- Why is ONNX Runtime faster than PyTorch for the same FP32 maths?

---

## Stage 3: INT8 quantization with ONNX Runtime (CPU)

**Core idea:** store and compute numbers as 8-bit integers instead of 32-bit floats. 4x
smaller weights, and CPUs have special instructions for fast INT8 dot products (VNNI).

**The maths**
- real_value ≈ scale × (int8_value - zero_point).
- *Scale* sets the step size; *zero point* shifts the range so that 0.0 is exactly
  representable. *Symmetric* = zero point 0; *asymmetric* = range not centred on 0.
- *Per-tensor vs per-channel:* one scale for a whole weight matrix, or one per output channel
  (more accurate for weights).
- Two error types: *rounding error* (values between steps) and *clipping error* (values
  outside the chosen range). A wide range has less clipping but more rounding, and the reverse.
- *Outliers:* transformers have a few huge activation values; they stretch the range and make
  all other values coarse. This is why ViTs/LLMs are harder to quantize than CNNs.

**Two flavours**
- *Dynamic quantization:* weights INT8 ahead of time; activation ranges computed on the fly
  for each input. No calibration data, very accurate, some runtime overhead.
- *Static quantization:* activation ranges fixed ahead of time from *calibration data*.
  Faster in principle, but if the ranges do not fit the real data, accuracy drops.
- *QDQ format:* the graph gets QuantizeLinear/DequantizeLinear pairs around ops. It is the
  portable way to carry INT8 info; OpenVINO and TensorRT read it too.
- *Calibration rule:* only training data (57 tiles from 19 training volumes), never test data,
  otherwise the result is leaked. We checked by hash that no tile came from the test set.

**Results (CPU)**
| Variant | Latency | Size | Cosine | IoU 0.25 TP/FP/FN |
|---|---|---|---|---|
| FP32 ONNX | 3,772 ms | 368 MB | 1.0 | 32/12/1 |
| Dynamic INT8 | 2,079 ms | 109 MB | 0.9996 | 32/12/1 |
| Static INT8, all ops | 3,063 ms | 101 MB | 0.989 | 31/9/2 |
| Static INT8, MatMul/Gemm/Conv only | 2,399 ms | 110 MB | 0.989 | 32/10/1 |

**Lessons**
- Quantizing everything (LayerNorm, Softmax, Add) was both slower and less accurate: those
  ops are sensitive, and the extra quantize/dequantize steps cost time. Quantize only the
  heavy matrix multiplications.
- Dynamic was the best CPU option: 1.8x faster, 3.4x smaller, same detections.
- A library bug (ORT 1.30): the option `CalibMaxIntermediateOutputs` silently discarded the
  collected ranges. We found it because the warning said "No data is collected", diagnosed it,
  and added an assert that all 57 tiles were read. Lesson: verify the calibration really ran.

**Interview questions**
- Explain scale and zero point. Symmetric vs asymmetric, per-tensor vs per-channel.
- Dynamic vs static quantization: trade-offs. What is calibration and how do you pick data?
- Why are transformers hard to quantize? Name techniques (SmoothQuant, mixed precision,
  keeping LayerNorm/Softmax in float, percentile calibration, QAT).
- PTQ vs QAT (post-training quantization vs quantization-aware training)?

---

## Stage 5: OpenVINO and NNCF on CPU, GPU and NPU

**What OpenVINO is:** Intel's inference engine. It converts a model to its own format
(*IR*: an .xml graph plus a .bin of weights) and compiles it for a device: CPU, integrated
GPU or NPU. *NNCF* is its companion library for compression (quantization, weight compression).

**Precision: storage vs compute (key idea)**
- `compress_to_fp16` only changes how weights are *stored* on disk (184 MB instead of 368 MB).
- The device chooses the *compute* precision. The CPU computed in FP32 anyway, so the "FP16"
  IR was no faster on CPU (3,770 vs 3,840 ms). The GPU computes in FP16 by default. The NPU
  only supports FP16 and INT8. We set `INFERENCE_PRECISION_HINT` explicitly and recorded it.

**NNCF INT8:** `nncf.quantize(model_type=TRANSFORMER)` keeps sensitive ops in float and
applies *SmoothQuant* (moves activation outliers into the weights, which are easier to
quantize) and *fast bias correction* (fixes the average shift INT8 introduces).

**Results (encoder median)**
| Device | Variant | Latency | Notes |
|---|---|---|---|
| CPU | FP32 | 3,840 ms | |
| CPU | NNCF INT8 | 1,889 ms | cosine 0.99982; IoU 0.25: 32/8/1 (fewer false positives) |
| GPU | FP32 | 624 ms | |
| GPU | FP16 | **241 ms** | cosine 1.0, detections identical to FP32 |
| GPU | INT8 (several kinds) | 371 to 680 ms | always slower than FP16, see below |
| NPU | all | failed | wrong output, then device lost |

- GPU FP16 is 23x faster than the PyTorch CPU baseline for the encoder.
- But the full pipeline got only 2.6x faster (5,941 s to 2,248 s). **Amdahl's law:** on the GPU
  the encoder is only about 14% of pipeline time; the rest (tiling, decoder, post-processing)
  did not change, so even an infinitely fast encoder could save at most 14% more.

**Energy**
- Measured with HWiNFO (logs package power every 2 s). For each device, ran inference back to
  back for 60 s, averaged power, then: joules per inference = watts × seconds / inferences.
  Also logged idle windows to subtract the baseline.
- GPU FP16 about 5 J per inference vs CPU FP32 66 J: about 13x less energy.
- Caveat to state in an interview: software sensor, not a power meter; 2 s sampling.
- Pitfall: HWiNFO added new sensor columns during logging, shifting rows. We only read
  columns that come before the point where new ones appear, and asserted it.

**Why INT8 was slower on the Arc GPU (the investigation)**
1. The NNCF INT8 model made from ONNX would not compile on the GPU
   (`CL_BUILD_PROGRAM_FAILURE`), 3 of 3 times in fresh processes, so it is not a one-off.
2. Converting PyTorch straight to OpenVINO (skipping ONNX) produced a graph that compiled,
   but it ran at 592 ms vs 241 ms for FP16.
3. *Per-layer profiling* (`PERF_COUNT`) showed why: the attention matrix multiplications got
   no optimised INT8 kernel ("undef"), and the element-wise Add/Multiply around the large
   4096x4096 attention maps fell back to FP32. The linear layers did get real INT8 kernels.
4. Linear layers only in INT8 (attention kept FP16): 465 ms. Better, still slower than FP16.
5. Weight-only INT8 (weights stored INT8, maths in FP16): 371 ms. Also slower.

**The concept behind point 5: compute-bound vs memory-bound**
- *Memory-bound:* the chip waits for data (weights) to arrive. Smaller weights help. Typical
  for LLMs generating one token at a time.
- *Compute-bound:* the chip is busy doing arithmetic. Smaller weights do not help, and
  decompressing them costs time. Our encoder processes 4,096 patches per tile, so each weight
  is reused thousands of times: high *arithmetic intensity*, compute-bound.
- Conclusion: INT8 pays off on the CPU (fast INT8 instructions), not on this GPU with this
  software stack. FP16 is the right choice for the Arc GPU. This is a measured result, not a
  failure.

**NPU failure (documented, valid result)**
- It compiled, then gave wrong output on some tiles, NaN in INT8, then
  `ZE_RESULT_ERROR_DEVICE_LOST`, also after a driver update.
- Hypothesis (untested): one inference on this large input takes too long and the driver's
  watchdog resets the device (similar to Windows TDR for GPUs). Tests would be: time a single
  inference, check the event log, try a smaller input such as 512x512.

**Interview questions**
- Storage precision vs compute precision: why did an FP16 file not speed up the CPU?
- Why might INT8 be slower than FP16 on a GPU? How would you find out? (profile per layer)
- Compute-bound vs memory-bound; arithmetic intensity; when does weight-only quantization help?
- Explain Amdahl's law with your own numbers.
- How did you measure energy, and what are the limitations?
- What do you do when a device fails? (reproduce, isolate, document, report upstream)

---

## Stage 4 preview: TensorRT (next, on Colab T4)

TensorRT is NVIDIA's inference compiler. It takes ONNX and builds an *engine* for one exact
GPU: it fuses layers, picks a precision per layer (FP32/FP16/INT8), and *auto-tunes* by timing
several kernels per layer and keeping the fastest. Engines are not portable across GPU models
or TensorRT versions. Tools: `trtexec` (build and benchmark from the command line),
Polygraphy (compare TensorRT output against ONNX Runtime layer by layer), ModelOpt (NVIDIA's
quantization toolkit), Nsight Systems (timeline profiler). The T4 has INT8 Tensor Cores, so
INT8 may behave very differently from the Arc GPU, which is the interesting comparison.

---

## Engineering habits worth talking about

- Change one thing at a time and keep the old file to compare against (legacy vs dynamo).
- Verify every conversion numerically and end-to-end on real data, not only random input.
- Guard against invalid measurements in code (power check, tile counter assert, hash check).
- When something fails: show the error, diagnose, then fix (the ORT calibration bug, the
  HWiNFO column shift, the GPU compile failure).
- Report negative results honestly (INT8 slower on GPU, NPU failure). They show judgement.
- Keep claims exactly as big as the evidence: "on a laptop CPU, GPU and NPU", not "edge".

## Two-minute interview story

"I took the ViT-B image encoder of a fine-tuned MicroSAM model and deployed it through ONNX,
ONNX Runtime and OpenVINO on an Intel Lunar Lake laptop. I set up a strict benchmark: batch 1,
warm-up, 30 runs, median and spread, power state checked in code. ONNX export alone gave 1.5x
on CPU with identical masks. INT8 quantization gave about 2x on CPU at 3.4x smaller size, with
the same detections. On the integrated Arc GPU, FP16 brought the encoder from 5.6 s to 241 ms
and cut energy per inference about 13x. INT8 was slower than FP16 on that GPU; I profiled it
per layer and found the attention matmuls had no INT8 kernel and nearby ops fell back to FP32.
The end-to-end pipeline improved only 2.6x, because the encoder became a small share of total
time, which is Amdahl's law. The NPU failed with a device-lost error, which I documented."
