import sys
import torch
from pathlib import Path

# Import Official Tools
from napari_cellseg3d.dev_scripts import colab_training as c
from napari_cellseg3d.config import WNetTrainingWorkerConfig, WandBConfig, WeightsInfo

# ================= CONFIGURATION =================
# 1. DATA: Point to the folder you created in Step 1
TRAIN_DATA_FOLDER = Path("/home/students/rawat/czi_output/Processed_images_float32_GreenAndRed")

# 2. RESULTS: Where to save the model weights (.pth)
RESULTS_FOLDER = Path("/work/scratch/rawat/models/wnet3d_spa_0_5_int_1_ncuts2_GreenAndRed")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
# =================================================

def train_official():
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"🚀 Starting Official WNet Training")
    print(f"📂 Data Source: {TRAIN_DATA_FOLDER}")
    
    # 1. SETUP DATASETS
    # Because we pre-processed the data, we can use this simple loader
    train_dict = c.create_dataset_dict_no_labs(str(TRAIN_DATA_FOLDER))
    
    if len(train_dict) == 0:
        print("❌ Error: No images found in training folder!")
        return

    # 2. CONFIGURE TRAINING
    # "Strict" Mode to fix the whole-image segmentation bug
    train_config = WNetTrainingWorkerConfig(
        device=DEVICE,
        max_epochs=100,          # WNet converges fast. 30 is usually enough.
        learning_rate=1e-5,     # Slow & Steady
        validation_interval=2,
        batch_size=2,           # Safe for 12GB RAM
        num_workers=0,          # 0 = Main process only (Best for debugging)
        
        # --- PATHS ---
        weights_info=WeightsInfo(),
        results_path_folder=str(RESULTS_FOLDER),
        train_data_dict=train_dict,
        eval_volume_dict=None,
        
        # --- THE CRITICAL FIXES ---
        num_classes=2,          # Foreground vs Background
        
        # Spatial Sigma: 0.5 (Was 4.0)
        spatial_sigma=0.5,      
        
        # Intensity Sigma: 2.0 (Was 1.0)
        intensity_sigma=1.0,    
        
        ncuts_radius=4,         # Radius for local graph connections
        patch_size=(64, 64, 64) # Sliding window size during training
    )

    # 3. DISABLE WANDB
    wandb_config = WandBConfig(
        mode="disabled",
        save_model_artifact=False,
    )

    # 4. START WORKER
    try:
        worker = c.get_colab_worker(
            worker_config=train_config, 
            wandb_config=wandb_config
        )
        
        print("⏳ Beginning training loop...")
        for result in worker.train():
            # This loop yields the loss dictionary every epoch
            epoch = result["epoch"]
            loss = result["loss"]
            print(f"   Epoch {epoch}: Loss = {loss:.4f}")
            
    except Exception as e:
        print(f"\n❌ Training Crashed: {e}")
        if "cuda" in str(e).lower() and DEVICE == "cuda":
            print("💡 TIP: Try changing DEVICE to 'cpu' in the script.")

    print(f"✅ Training Complete. Check {RESULTS_FOLDER} for .pth files.")

if __name__ == "__main__":
    train_official()