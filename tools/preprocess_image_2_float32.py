import os
import tifffile
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURATION =================
INPUT_FOLDER = r"/home/students/rawat/czi_output/processed_images"

# Folder for Model A (e.g., Green Channel only)
OUTPUT_FOLDER = r"/home/students/rawat/czi_output/Processed_images_float32_GreenAndRed" 

# WHICH CHANNELS TO KEEP?
# [0] = Keep only 1st channel (Green)
# [0, 1] = Keep first two channels
# [1] = Keep only 2nd channel (Red)
CHANNELS_TO_KEEP = [0,1] 

# YOUR NORMALIZATION SETTINGS
PERCENTILES = (0.0001, 99.999) 
# =================================================

def normalize_and_save(path, save_dir):
    # 1. Load the raw uint16 image
    img = tifffile.imread(path)
    
    # 2. Handle Dimensions (Ensure C, Z, Y, X)
    # This logic fixes the "random dimension order" of microscopes
    if img.ndim == 3: # (Z, Y, X) -> Add Channel dim
        img = img[np.newaxis, ...]
    elif img.ndim == 4:
        # If channel is last (Z, Y, X, C), move to front
        if img.shape[-1] <= 4: 
            img = np.transpose(img, (3, 0, 1, 2))
        # If channel is second (Z, C, Y, X), move to front
        elif img.shape[1] <= 4:
            img = np.transpose(img, (1, 0, 2, 3))
            
    # 3. Select Channels
    # Only keep the indices you specified in CONFIG
    try:
        img = img[CHANNELS_TO_KEEP, ...]
    except IndexError:
        print(f"Skipping {path.name}: Image has fewer channels than requested.")
        return

    # 4. Apply Your Custom Normalization
    img = img.astype(np.float32) # Convert to float
    
    for c in range(img.shape[0]):
        # Calculate your robust percentiles
        low, high = np.percentile(img[c], PERCENTILES)
        
        # Avoid division by zero if image is empty
        if high > low:
            # Clip values outside the range
            img[c] = np.clip(img[c], low, high)
            # Scale to 0-1
            img[c] = (img[c] - low) / (high - low)
        else:
            img[c] = img[c] * 0 # Empty channel stays empty

    # 5. Save as Float32
    save_path = save_dir / path.name
    tifffile.imwrite(save_path, img)

def main():
    in_path = Path(INPUT_FOLDER)
    out_path = Path(OUTPUT_FOLDER)
    out_path.mkdir(parents=True, exist_ok=True)
    
    files = list(in_path.glob("*.tif"))
    print(f"Processing {len(files)} files...")
    print(f"Keeping Channels: {CHANNELS_TO_KEEP}")
    print(f"Normalization: {PERCENTILES} -> Float32")
    
    for p in tqdm(files):
        normalize_and_save(p, out_path)
        
    print(f"✅ Done. Normalized data saved to: {out_path}")

if __name__ == "__main__":
    main()