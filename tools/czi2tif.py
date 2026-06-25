#!/usr/bin/env python3
"""
process_images.py
Reads images from a folder, takes the first channel, and duplicates it across
all three channels of a new RGB image, saving the result to a new folder.
"""
import sys
from pathlib import Path
import numpy as np
import tifffile
from tqdm import tqdm

def process_and_save_image(input_path: Path, output_path: Path):
    """
    Reads an image, duplicates its first channel into a 3-channel image,
    and saves it.
    """
    try:
        # Read the input image
        image_data = tifffile.imread(input_path)

        # Ensure the image has a channel dimension
        if image_data.ndim == 2:
            # If it's a 2D grayscale image, add a channel axis
            image_data = image_data[..., np.newaxis]
        
        if image_data.shape[-1] == 0:
            print(f"  Skipping (no channels): {input_path.name}")
            return False

        # Take the first channel (channel 0)
        first_channel = image_data[..., 0]

        # Create a new 3-channel image by stacking the first channel 3 times
        # This assumes the output should be in (Height, Width, Channels) format
        new_image = np.stack([first_channel, first_channel, first_channel], axis=-1)

        # Save the new image
        tifffile.imwrite(output_path, new_image)
        
        # print(f"  Processed and saved: {output_path.name}")
        return True

    except Exception as e:
        print(f"  Failed to process {input_path.name}: {e}", file=sys.stderr)
        return False

def main():
    print("\nImage Channel Duplicator\n")
    
    # --- Configuration ---
    input_dir = Path("/home/students/rawat/training_data_new/data/train/images/")
    output_dir = Path("/home/students/rawat/training_data_new/data/train/images_green/")

    if not input_dir.is_dir():
        print(f"Error: Input directory not found at {input_dir}")
        sys.exit(1)

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input folder:  {input_dir}")
    print(f"Output folder: {output_dir}")

    # Find all .tif files in the input directory
    image_files = list(input_dir.glob("*.tif"))
    if not image_files:
        print("No .tif files found in the input directory.")
        sys.exit(0)
        
    print(f"Found {len(image_files)} images to process.")

    success_count = 0
    for image_path in tqdm(image_files, desc="Processing images"):
        # Define the output path for the new image
        output_path = output_dir / image_path.name
        
        if process_and_save_image(image_path, output_path):
            success_count += 1

    print(f"\nDone: {success_count}/{len(image_files)} images processed successfully.")

if __name__ == "__main__":
    main()