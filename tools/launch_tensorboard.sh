#!/bin/bash
# ============================================================
# TensorBoard launcher for thesis final model training logs
# Usage: bash launch_tensorboard.sh
# ============================================================
#
# WHAT YOU CAN CHANGE:
#
#   MODEL   -> which model group to view
#               "microsam"   : MicroSAM final runs (vit_b v1 + vit_l v3, all 4 preprocessings)
#               "swinunetr"  : SwinUNETR final runs (all 4 preprocessings)
#
#   PORT    -> browser port (default 6006). Change if that port is already in use.
#
# ============================================================

MODEL="microsam"   # <-- change to "microsam" or "swinunetr"
PORT=6006

# ============================================================
MODELS_ROOT="/work/scratch/rawat/models"
TENSORBOARD_BIN="/work/scratch/rawat/installations/conda/envs/thesis_microglomeruli/bin/tensorboard"

case "$MODEL" in
    microsam)
        LOG_DIR="$MODELS_ROOT/microsam_training_output_final_v2/logs"
        ;;
    swinunetr)
        LOG_DIR="$MODELS_ROOT/monai_swinunetr_final"
        ;;
    *)
        echo "ERROR: Unknown model '$MODEL'. Use 'microsam' or 'swinunetr'."
        exit 1
        ;;
esac

if [ ! -d "$LOG_DIR" ]; then
    echo "ERROR: Log directory not found: $LOG_DIR"
    exit 1
fi

echo "Model  : $MODEL"
echo "Logs   : $LOG_DIR"
echo "Port   : $PORT"
echo ""
echo "Runs found:"
find "$LOG_DIR" -name "events.out.tfevents*" -printf "  %h\n" | sort -u
echo ""
echo "Opening TensorBoard at http://localhost:$PORT"
echo "(Press Ctrl+C to stop)"
echo ""

"$TENSORBOARD_BIN" --logdir "$LOG_DIR" --port "$PORT" --bind_all
