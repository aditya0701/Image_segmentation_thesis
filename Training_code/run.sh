#!/usr/bin/env bash
# ------------------------------------------------------------
#  run.sh – wrapper for HTCondor jobs on the LFB cluster
# ------------------------------------------------------------


# 2️⃣  Activate the conda environment
# Check this path! It must match where you installed micro_sam
export CONDA_PREFIX=/work/scratch/rawat/installations/Miniconda3-latest-Linux-x86_64.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# source /work/scratch/rawat/installations/conda/envs/thesis_microglomeruli


# We use 'nvidia-smi' to check free memory
vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)

echo "GPU has $vram MB free memory."

# Logic: If >32GB use Batch 4, If >20GB use Batch 2, Else Batch 1
if [ "$vram" -gt 20000 ]; then
    BATCH=2
else
    BATCH=1
fi

echo "Selected Batch Size: $BATCH"

# 3. Run Python with the selected batch size
# -u means unbuffered, so you see prints immediately
python  /work/scratch/rawat/image-segmentation-thesis/Training_code/training_microsam.py $BATCH