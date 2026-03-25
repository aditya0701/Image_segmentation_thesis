import tifffile
import napari

# --- PATH ---
IMAGE_PATH = r"/home/students/rawat/Dataset/Surface_Masks/Surface_Masks/airyscan_2ndT/Brain1_1340_Cont_AiryscanSR_Fly1_CalyxL.tif"

# --- INSPECT AXES ---
with tifffile.TiffFile(IMAGE_PATH) as tif:
    axes = tif.series[0].axes
    print(f"Axes: {axes}")
    print(f"Shape: {tif.series[0].shape}")

# --- LOAD ---
img = tifffile.imread(IMAGE_PATH)[1].astype('float32')  # selects the second time point
print(f"Shape: {img.shape}, Dtype: {img.dtype}")

# --- VIEW ---
# The T axis is gone, so we need to adjust the axes string
channel_axis = axes.index("C") if "C" in axes else None

viewer = napari.Viewer()
viewer.add_image(
    img,
    channel_axis=channel_axis,
    name="Airyscan"
)

print("Use sliders at the bottom to navigate dimensions.")
napari.run()