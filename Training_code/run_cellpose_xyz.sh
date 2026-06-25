#!/usr/bin/env bash
# run_cellpose_xyz.sh — HTCondor wrapper for CellposeSAM XY+XZ+YZ training

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
echo "GPU has $vram MB free memory."

if [ -z "$1" ]; then
    echo "ERROR: no dataset argument. Usage: run_cellpose_xyz.sh <all|dog|original|psf>"
    exit 1
fi
DATASET=$1
echo "Dataset: $DATASET"

/work/scratch/rawat/installations/conda/envs/thesis_microglomeruli/bin/python -u \
    /work/scratch/rawat/image-segmentation-thesis/Training_code/training_cellpose3d_xyz.py \
    --dataset $DATASET
