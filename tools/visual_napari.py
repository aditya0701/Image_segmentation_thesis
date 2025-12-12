import tifffile
import numpy as np
import napari

# --- PATHS ---
IMAGE_PATH = r'/home/students/rawat/czi_output/PN_KC_Calyx_CS-_GFP_mCherry568_anti-synapsin633_brain1.ome.tif'
# Use the path where you saved your auto-generated results
MASK_PATH = r'/home/students/rawat/czi_output/Annotations/PN_KC_Calyx_CS-_GFP_mCherry568_anti-synapsin633_brain1/Brain1_Auto3D.tif'

print("Loading images...")
img = tifffile.imread(IMAGE_PATH)
mask = tifffile.imread(MASK_PATH)

print(f"Image shape: {img.shape}")
print(f"Mask shape: {mask.shape}")

# Initialize Viewer
viewer = napari.Viewer()

# 1. Add the Raw Image (Split into 3 Channels)
# channel_axis=0 splits the first dimension (3) into three separate layers
viewer.add_image(
    img, 
    channel_axis=0, 
    colormap=['green', 'red', 'magenta'],
    name=['GFP', 'mCherry', 'anti-synapsin'],
    scale=(1, 1, 1) # Ensures pixels are treated as cubes
)

# 2. Add the Annotation Mask (as a Labels layer)
# We add this as a single layer. Napari automatically handles the Z-stack 
# overlay because the shape (92, 1431, 1431) matches the shape of your image channels.
# viewer.add_labels(
#     mask, 
#     name='Bouton Annotations',
#     opacity=0.6,
#     scale=(1, 1, 1)
# )

print("Napari launched. Use the slider at the bottom to scroll through Z-slices.")
napari.run()