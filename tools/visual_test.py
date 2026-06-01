import tifffile
import napari
import numpy as np

# --- PATH ---
# Point to the new folder with 2-channel images
IMAGE_PATH = r"/home/students/rawat/training_data_new/data/train/images/PN_KC_488_568_Brain1_CalyxL_LSMPlus_16bit_slice_0000.tif"

# --- CONFIGURATION ---
# The channels are now 0 and 1 in the new files.
CHANNEL_NAMES = ["Boutons", "Kenyon Cells", "Channel 3"]
CHANNEL_COLORS = ["magenta", "green", "blue"]


# --- INSPECT AXES ---
with tifffile.TiffFile(IMAGE_PATH) as tif:
    # These images might not have axes metadata, so we handle that
    try:
        axes = tif.series[0].axes
        print(f"Axes: {axes}")
    except (ValueError, IndexError):
        axes = 'ZCYX' # Assume ZCYX if no axes metadata is found
        print(f"Axes metadata not found. Assuming: {axes}")
    print(f"Shape: {tif.series[0].shape}")

# --- LOAD ---
# The images are already processed, so we just load them.
img = tifffile.imread(IMAGE_PATH).astype('float32')
print(f"Loaded Shape: {img.shape}, Dtype: {img.dtype}")


# --- VIEW ---
# The channel axis is now 2 in the (Y, X, C) shaped image
viewer = napari.Viewer()
viewer.add_image(
    img,
    channel_axis=2,
    name=CHANNEL_NAMES,
    colormap=CHANNEL_COLORS,
    blending='additive',
)

print("Use sliders at the bottom to navigate dimensions.")
napari.run()