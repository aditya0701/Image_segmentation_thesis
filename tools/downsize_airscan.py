import os
import tifffile
import numpy as np
from tqdm import tqdm

def process_airyscan_images(input_folder, output_folder):
    """
    Processes Airyscan images to extract the second time point.

    Args:
        input_folder (str): Path to the folder containing the input TIFF images.
        output_folder (str): Path to the folder where the processed images will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    all_files = os.listdir(input_folder)
    image_files = [f for f in all_files if f.endswith(('.tif', '.tiff')) and '_Mask' not in f]

    print(f"Found {len(image_files)} images to process.")

    for filename in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            with tifffile.TiffFile(input_path) as tif:
                if 'T' not in tif.series[0].axes or tif.series[0].shape[tif.series[0].axes.find('T')] < 2:
                    print(f"Skipping {filename}: Does not have at least two time points.")
                    continue
            
            # Load the whole image first
            img_data = tifffile.imread(input_path)

            # Select the second time point (index 1)
            processed_img = img_data[1].astype(np.float32)

            # Save the processed image
            tifffile.imwrite(output_path, processed_img, imagej=True)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    # --- PATHS ---
    INPUT_IMAGE_FOLDER = r"/home/students/rawat/Dataset/Surface_Masks/Surface_Masks/"
    OUTPUT_FOLDER_NAME = "airyscan_2ndT"
    
    # Create the full path for the output directory
    output_directory = os.path.join(INPUT_IMAGE_FOLDER, OUTPUT_FOLDER_NAME)

    # Process the images
    process_airyscan_images(INPUT_IMAGE_FOLDER, output_directory)

    print("\nProcessing complete.")
    print(f"Processed images are saved in: {output_directory}")
