import os
import glob
import numpy as np
import tifffile

# --- CONFIGURATION ---
# Path to the folder containing your images
TARGET_FOLDER = "/home/students/rawat/training_data/data/train/images"

print(f"Scanning folder: {TARGET_FOLDER}")
print("------------------------------------------------")

# Get all .tif files
files = sorted(glob.glob(os.path.join(TARGET_FOLDER, "*.tif")))

count_converted = 0
count_skipped = 0

for file_path in files:
    try:
        # 1. Load the image
        img = tifffile.imread(file_path)
        fname = os.path.basename(file_path)

        # 2. Check if it needs conversion (Only convert if NOT uint8)
        if img.dtype != np.uint8:
            
            # --- YOUR NORMALIZATION LOGIC ---
            low, high = np.percentile(img, (1, 99))
            
            if high > low:
                # Normalize to 0-1 based on percentiles
                img_norm = (img.astype(float) - low) / (high - low)
                # Clip to 0-1, scale to 255, convert to uint8
                img_u8 = (np.clip(img_norm, 0, 1) * 255).astype(np.uint8)
            else:
                # Fallback if image is completely flat/empty
                img_u8 = img.astype(np.uint8)
            # -------------------------------

            # 3. Overwrite the existing file
            tifffile.imwrite(file_path, img_u8)
            
            print(f"✅ Converted & Overwritten: {fname} ({img.dtype} -> {img_u8.dtype})")
            count_converted += 1
            
        else:
            # print(f"⏭️  Skipping: {fname} (Already uint8)")
            count_skipped += 1

    except Exception as e:
        print(f"❌ Error processing {os.path.basename(file_path)}: {e}")

print("------------------------------------------------")
print(f"🎉 Process Complete.")
print(f"🔄 Converted: {count_converted} files")
print(f"⏭️  Skipped:   {count_skipped} files")