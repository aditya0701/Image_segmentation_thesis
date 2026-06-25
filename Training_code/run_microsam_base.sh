#!/usr/bin/env bash
# ------------------------------------------------------------
#  run_microsam_base.sh – HTCondor wrapper for vit_b_lm training
# ------------------------------------------------------------

export CONDA_PREFIX=/work/scratch/rawat/installations/Miniconda3-latest-Linux-x86_64.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
echo "GPU has $vram MB free memory."

BATCH=1
echo "Selected Batch Size: $BATCH"

# $1 = dataset (all | dog | original | psf)
# $2 = optional "--augmented"
DATASET=${1:-all}
AUG_FLAG=${2:-}
echo "Dataset: $DATASET | Aug flag: ${AUG_FLAG:-none}"

/work/scratch/rawat/installations/conda/envs/thesis_microglomeruli/bin/python -u \
    /work/scratch/rawat/image-segmentation-thesis/Training_code/train_microsam_base.py \
    --batch_size $BATCH --dataset $DATASET $AUG_FLAG
