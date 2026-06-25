#!/usr/bin/env bash
# nnUNet batch inference — all 4 datasets
# Runs in bw_env on a Condor GPU node (requires sm_75+)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export nnUNet_n_proc_DA=1
export nnUNet_compile=0
export BLOSC_NTHREADS=1
export BLOSC2_NTHREADS=1

/work/scratch/rawat/installations/conda/envs/bw_env/bin/python -u \
    /work/scratch/rawat/image-segmentation-thesis/Training_code/inference/batch_infer_nnunet.py
