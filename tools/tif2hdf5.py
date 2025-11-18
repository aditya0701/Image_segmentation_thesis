import os
import glob
import tifffile
import h5py
import numpy as np

def convert_tif_to_h5(input_folder, output_folder):
    """
    Converts all .tif files in the input_folder to .h5 files
    in the output_folder.
    """
    # 1. Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # 2. Find all .tif and .tiff files in the input folder
    search_path = os.path.join(input_folder, '*.tif*')
    tif_files = glob.glob(search_path)
    
    if not tif_files:
        print(f"No TIFF files found in: {input_folder}")
        return

    print(f"Found {len(tif_files)} TIFF file(s) to convert.")
    
    # 3. Process each TIFF file
    for tif_path in tif_files:
        # Get just the file name (e.g., 'image_01.tif')
        base_name = os.path.basename(tif_path)
        # Create the output file name (e.g., 'image_01.h5')
        h5_name = os.path.splitext(base_name)[0] + '.h5'
        # Define the full path for the output HDF5 file
        h5_path = os.path.join(output_folder, h5_name)
        
        print(f"\nProcessing: {base_name}...")
        
        try:
            # --- Conversion Logic ---
            
            # Read the TIFF file into a NumPy array
            tiff_data = tifffile.imread(tif_path)

            # Write the array to a new HDF5 file
            with h5py.File(h5_path, 'w') as hf:
                # Store the data in a dataset named 'image_data'
                # Use chunking and gzip compression for better performance/smaller files
                hf.create_dataset(
                    'image_data', 
                    data=tiff_data, 
                    chunks=True, 
                    compression="gzip"
                )
            
            print(f"SUCCESS: Saved to {h5_path}")
            
        except Exception as e:
            print(f"ERROR converting {base_name}: {e}")
            
# --- CONFIGURATION ---
# IMPORTANT: Change these two paths to your specific folder locations.
INPUT_FOLDER = r'C:\Users\YourName\Desktop\InputTIFs'
OUTPUT_FOLDER = r'C:\Users\YourName\Desktop\OutputH5s'

# --- RUN THE FUNCTION ---
if __name__ == "__main__":
    convert_tif_to_h5(INPUT_FOLDER, OUTPUT_FOLDER)