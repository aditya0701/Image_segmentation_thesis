import tifffile
import numpy as np

# --- PATHS ---
# Path to the "wrong" 4D file you just made
WRONG_MASK_PATH = r"/home/students/rawat/czi_output/Annotations/PN_KC_Calyx_CS-_GFP_mCherry568_anti-synapsin633_brain1/Brain1_Auto3D.tif"

# Path where we will save the "fixed" 3D file
FIXED_MASK_PATH = r"/home/students/rawat/czi_output/Annotations/PN_KC_Calyx_CS-_GFP_mCherry568_anti-synapsin633_brain1/Brain1_Auto3D.tif"

print("Loading 4D mask...")
mask_4d = tifffile.imread(WRONG_MASK_PATH)
print(f"Current shape: {mask_4d.shape}")  # (92, 1431, 1431, 3)

# --- THE FIX ---
# We just take the first channel (index 0).
# The syntax [..., 0] means "keep all Z, all Y, all X, but take only channel 0"
mask_3d = mask_4d[..., 0]

print(f"Fixed shape:   {mask_3d.shape}")  # (92, 1431, 1431)

print("Saving fixed file...")
tifffile.imwrite(FIXED_MASK_PATH, mask_3d)
print("Done! You can now open the '_Fixed.tif' file in Napari.")