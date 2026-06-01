#!/usr/bin/env bash
# ------------------------------------------------------------
#  run_cellpose.sh – HTCondor wrapper for Cellpose 3D training
# ------------------------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
echo "GPU has $vram MB free memory."

# Dataset MUST be passed as first argument: condor_submit cellpose_train.tbi -a "args=dog"
# Valid values: all | dog | original | psf
if [ -z "$1" ]; then
    echo "ERROR: no dataset argument given. Usage: run_cellpose.sh <all|dog|original|psf>"
    exit 1
fi
DATASET=$1
echo "Dataset: $DATASET"

/work/scratch/rawat/installations/conda/envs/thesis_microglomeruli/bin/python -u \
    /work/scratch/rawat/image-segmentation-thesis/Training_code/training/training_cellpose3d.py \
    --dataset $DATASET
