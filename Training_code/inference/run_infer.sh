#!/usr/bin/env bash
# Wrapper for Condor inference jobs.
# Usage: run_infer.sh <path/to/batch_infer_*.py>
# Routes to the correct conda env automatically.

SCRIPT=$1
if [ -z "$SCRIPT" ]; then
    echo "ERROR: no script argument given."
    exit 1
fi

INFER_DIR="/work/scratch/rawat/image-segmentation-thesis/Training_code/inference"
if [[ "$SCRIPT" != /* ]]; then
    SCRIPT="$INFER_DIR/$SCRIPT"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)
echo "GPU free memory: ${vram} MB"
echo "Running: $SCRIPT"

# nnUNet requires bw_env; everything else uses thesis_microglomeruli
if [[ "$SCRIPT" == *"nnunet"* ]]; then
    export nnUNet_n_proc_DA=1
    export nnUNet_compile=0
    export BLOSC_NTHREADS=1
    export BLOSC2_NTHREADS=1
    PYTHON=/work/scratch/rawat/installations/conda/envs/bw_env/bin/python
else
    PYTHON=/work/scratch/rawat/installations/conda/envs/thesis_microglomeruli/bin/python
fi

$PYTHON -u "$SCRIPT"
