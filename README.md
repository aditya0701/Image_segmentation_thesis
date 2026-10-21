<div align="center">

# 🧠 Microglomeruli Segmentation

### Instance segmentation of synaptic boutons in *Drosophila melanogaster* brains from fluorescent confocal microscopy

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-ee4c2c)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](#license)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-models-yellow)](https://huggingface.co/aditya0701/Drosophilla_melanogaster_Bouton_3d_segmentation)
[![Annotation Tool](https://img.shields.io/badge/GitHub-annotation%20tool-181717?logo=github)](https://github.com/aditya0701/Fluorescent-Microscopy-Image-Segmentation-and-Quantification)

</div>

---

## What this is

This repository contains the training, inference, and evaluation code for my thesis on segmenting **microglomeruli** — synaptic bouton structures — in confocal microscopy volumes of the *Drosophila* brain. Several segmentation backbones are trained and compared: **MicroSAM**, **Cellpose 3D**, **nnU-Net v2**, and **SwinUNETR**.

## Sample results

A MicroSAM (`vit_l_lm`) model fine-tuned on this data, run on a held-out volume:

<div align="center">

| Raw input | Ground truth | Prediction |
|:---:|:---:|:---:|
| ![Raw input](assets/gifs/original.gif) | ![Ground truth](assets/gifs/ground_truths.gif) | ![Prediction](assets/gifs/predicitons.gif) |

</div>

## Overview

Models are trained on image/mask `.tif` slices (or volumes for the 3D pipelines) across several preprocessing variants of the same underlying dataset, and evaluated with instance-matching metrics.

| Dataset key | Description |
|---|---|
| `all` | Full combined dataset |
| `dog` | Difference-of-Gaussians preprocessed |
| `original` | Unprocessed raw images |
| `psf` | PSF-deconvolved images |

## Repository layout

```
Training_code/
  training_microsam.py          # MicroSAM (vit_l_lm) fine-tuning, 2D, with Kornia augmentations
  training_microsam_continual.py
  train_microsam_base.py
  training_cellpose3d.py        # Cellpose 3D training
  training_cellpose3d_xyz.py
  nnunet_training.py            # nnU-Net v2 training wrapper
  swinunetr_training.py         # SwinUNETR 3D training (MONAI)
  prepare_nnunet_dataset.py     # Converts TIFFs -> nnU-Net's NIfTI dataset format
  run*.sh, *.sub                # Shell / HTCondor job launch scripts (LFB cluster)
  inference/                    # Batch inference scripts + notebooks per model
tools/
  evaluate_segmentation.py      # Instance-matching evaluation metrics
  czi2tif.py                    # Zeiss .czi -> .tif conversion
  images_2_slices.py            # Volume -> 2D slice extraction
  annotate_img.py, visual_*.py, interactive_mask_viewer.py  # Annotation / visualization utilities
  downscale_*.py, shrink_mask.py, thresholding.py, region_growing.py, edge_detection.py  # Pre/post-processing utilities
  *.ipynb                       # Exploratory notebooks (data verification, diagnostics, seed detection)
weights/                        # Pretrained / fine-tuned model checkpoints
assets/gifs/                    # Sample input / ground-truth / prediction visualizations
```

## Pretrained weights

The final, selected model checkpoints (i.e. the ones reported in the thesis, not rejected experiment runs) are published on Hugging Face:

🤗 **[aditya0701/Drosophilla_melanogaster_Bouton_3d_segmentation](https://huggingface.co/aditya0701/Drosophilla_melanogaster_Bouton_3d_segmentation)**

## Edge deployment

The fine-tuned MicroSAM image encoder was also deployed with ONNX, ONNX Runtime, OpenVINO and TensorRT 11, on a laptop CPU, integrated GPU and NPU and on a Colab T4. Every number is measured, with accuracy checked against the FP32 model.

- On the laptop's Intel Arc GPU, FP16 runs the encoder **23x faster** than the PyTorch CPU baseline (5.6 s to 241 ms), with identical detections.
- INT8 on CPU gives **2.7x** at a third of the model size. INT8 was not faster than FP16 on the GPUs tested.
- Read **[edge_deploy/WRITEUP.md](edge_deploy/WRITEUP.md)** for results and findings, **[edge_deploy/CLAIMS.md](edge_deploy/CLAIMS.md)** for what is and is not claimed, and **[edge_deploy/LEARNING_NOTES.md](edge_deploy/LEARNING_NOTES.md)** for the step-by-step study notes.

## Related resources

For an end-user-facing tool built on top of these models — letting users run segmentation and manually delete unwanted/false-positive boutons — see:

🛠️ **[Fluorescent-Microscopy-Image-Segmentation-and-Quantification](https://github.com/aditya0701/Fluorescent-Microscopy-Image-Segmentation-and-Quantification)**

## Setup

This project was developed across two machines, so two setup paths are supported:

**Mamba/conda** (annotation & visualization machine, napari-based tooling), defined in `environment.yml`:

```bash
mamba env create -f environment.yml
```

**pip** (training/inference machine), via `requirement.txt`:

```bash
pip install -r requirement.txt
```

`micro_sam` is not reliably published on PyPI — install it via conda/mamba instead (see comment in `requirement.txt`).

## Training

Each model has a dedicated training script under `Training_code/`. Most accept a `--dataset` flag selecting one of `all | dog | original | psf`, and a `--batch_size` flag.

Example — MicroSAM fine-tuning:

```bash
python Training_code/training_microsam.py --batch_size 1 --dataset all
```

On the LFB HTCondor cluster, training is launched via the corresponding `run*.sh` wrapper and `.sub` submit file, e.g.:

```bash
bash Training_code/run.sh all
```

Each training script expects data laid out as:

```
<DATA_ROOT>/train/images/*.tif
<DATA_ROOT>/train/masks/*.tif
<DATA_ROOT>/validation/images/*.tif   # optional — falls back to an auto split by brain volume
<DATA_ROOT>/validation/masks/*.tif
```

## Inference

Batch inference scripts live in `Training_code/inference/` (one per model backbone), alongside notebooks for interactive inspection of predictions.

## Evaluation

`tools/evaluate_segmentation.py` matches predicted instances to ground truth via linear sum assignment and computes per-dataset metrics (with physical voxel volume taken into account).

## Citation

If you use these weights in your work, please cite the accompanying thesis.

```bibtex
@mastersthesis{rawat2026microsam,
  author  = {Aditya Rawat},
  title   = {Image Segmentation and Quantification of Synaptic Markers in the Mushroom Body Calyx of Drosophila melanogaster},
  school  = {RWTH Aachen University, Software Engineering Group},
  year    = {2026}
}
```

## License

This model is released under the MIT License. Please refer to the [LICENSE](LICENSE) file in this repository for full terms.

## Acknowledgments

This work was carried out at the Software Engineering Group, RWTH Aachen University, under the supervision of Prof. Dr. Abigail Morrison, with Prof. Dr.-Ing. Johannes Stegmaier serving as second examiner. The Imaris reference segmentations used for comparison were provided by collaborating biologists in the Tavosanis laboratory. Hardware for training was provided by the Institute of Imaging & Computer Vision (LfB), RWTH Aachen University.
