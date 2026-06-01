#!/usr/bin/env bash
# ------------------------------------------------------------
#  nnunet_train.sh – wrapper for HTCondor jobs on the LFB cluster
# ------------------------------------------------------------


# 2️⃣  Activate the conda environment
# Check this path! It must match where you installed miniconda
export CONDA_PREFIX=/work/scratch/rawat/installations/Miniconda3-latest-Linux-x86_64.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export nnUNet_n_proc_DA=1        # single DA worker avoids blosc2 thread-safety crash
export nnUNet_compile=0          # disable torch.compile (uses extra memory during JIT)
export BLOSC_NTHREADS=1          # limit blosc2 internal threads to avoid heap corruption
export BLOSC2_NTHREADS=1


# Dataset ID passed as argument: condor_submit ... -a "args=2"
# Falls back to the hardcoded value if no argument given
DATASET_ID=${1:-2}
FOLD=0


# 3️⃣  Run Python with the selected config
# -u means unbuffered, so you see prints immediately
/work/scratch/rawat/installations/conda/envs/bw_env/bin/python -u /work/scratch/rawat/image-segmentation-thesis/Training_code/training/nnunet_training.py --config 3d_fullres --dataset $DATASET_ID --fold $FOLD
