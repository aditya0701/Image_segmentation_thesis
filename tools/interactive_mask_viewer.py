"""
Interactive Mask Image Viewer with Napari
Navigate through mask images using keyboard controls
Shows image shape and checks mask uniqueness
"""

import os
import numpy as np
import tifffile
import napari
from pathlib import Path
from typing import List, Tuple, Dict, Any


class MaskImageViewer:
    """Modular interactive viewer for mask images"""
    
    def __init__(self, folder_path: str):
        """
        Initialize the viewer with a folder path
        
        Args:
            folder_path: Path to folder containing mask images
        """
        self.folder_path = Path(folder_path)
        self.supported_formats = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
        self.image_list = []
        self.current_index = 0
        self.viewer = None
        self.image_data = None
        self.current_image_path = None
        
        # Load images
        self._load_images()
        
    def _load_images(self) -> None:
        """Load all supported image formats from the folder"""
        if not self.folder_path.exists():
            raise ValueError(f"Folder not found: {self.folder_path}")
        
        # Get all image files
        for ext in self.supported_formats:
            self.image_list.extend(
                sorted(self.folder_path.glob(f"*{ext}"))
            )
            self.image_list.extend(
                sorted(self.folder_path.glob(f"*{ext.upper()}"))
            )
        
        # Remove duplicates while preserving order
        seen = set()
        unique_images = []
        for img in self.image_list:
            if str(img) not in seen:
                seen.add(str(img))
                unique_images.append(img)
        
        self.image_list = unique_images
        
        if not self.image_list:
            raise ValueError(
                f"No images found in {self.folder_path}\n"
                f"Supported formats: {self.supported_formats}"
            )
        
        print(f"Found {len(self.image_list)} images")
        
    def analyze_mask(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze mask properties
        
        Args:
            image: Image array to analyze
            
        Returns:
            Dictionary with analysis results
        """
        unique_values = np.unique(image)
        mask_ids = unique_values[unique_values != 0]  # Exclude background (0)
        
        analysis = {
            'shape': image.shape,
            'dtype': image.dtype,
            'min_value': int(np.min(image)),
            'max_value': int(np.max(image)),
            'total_unique_values': int(unique_values.size),
            'all_unique_values': unique_values,
            'num_masks': int(mask_ids.size),
            'mask_ids': mask_ids,
            'has_duplicates': len(np.unique(image)) < np.prod(image.shape),
            'mask_overlap_check': self._check_mask_overlap(image),
        }
        return analysis
    
    def _check_mask_overlap(self, image: np.ndarray) -> str:
        """
        Check if masks overlap (each pixel should belong to only one mask)
        
        Args:
            image: Mask image to analyze
            
        Returns:
            String describing overlap status
        """
        unique = np.unique(image)
        # In proper segmentation, each pixel has one mask ID (no overlap)
        # If pixels have same value repeated, it's valid
        if image.size > 0:
            return "No overlaps (each pixel has one mask ID)"
        return "Unknown"
    
    def load_image(self, file_path: Path) -> np.ndarray:
        """
        Load image from file
        
        Args:
            file_path: Path to image file
            
        Returns:
            Loaded image array
        """
        if file_path.suffix.lower() in ['.tif', '.tiff']:
            image = tifffile.imread(str(file_path))
        else:
            # For PNG, JPG, etc.
            import cv2
            image = cv2.imread(str(file_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if image is None:
                image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        
        return image
    
    def print_image_info(self) -> None:
        """Print detailed information about current image"""
        if self.current_image_path is None:
            return
        
        analysis = self.analyze_mask(self.image_data)
        
        print("\n" + "="*70)
        print(f"Image {self.current_index + 1}/{len(self.image_list)}")
        print(f"File: {self.current_image_path.name}")
        print("="*70)
        print(f"Shape: {analysis['shape']}")
        print(f"Data Type: {analysis['dtype']}")
        print(f"Value Range: [{analysis['min_value']}, {analysis['max_value']}]")
        print(f"Total Unique Values: {analysis['total_unique_values']}")
        print(f"All Unique Values: {analysis['all_unique_values']}")
        print(f"Number of Masks (excluding background): {analysis['num_masks']}")
        print(f"Mask IDs: {analysis['mask_ids']}")
        print(f"Masks are Unique: {not analysis['has_duplicates']}")
        print(f"Overlap Status: {analysis['mask_overlap_check']}")
        print("="*70)
        print("\nKeyboard Controls:")
        print("  [RIGHT/→]  - Next image")
        print("  [LEFT/←]   - Previous image")
        print("  [Q]        - Quit viewer")
        print("="*70 + "\n")
    

    
    def display_image(self, index: int) -> None:
        """
        Display image at specified index
        
        Args:
            index: Index of image to display
        """
        if index < 0 or index >= len(self.image_list):
            return
        
        self.current_index = index
        self.current_image_path = self.image_list[index]
        
        # Load image
        self.image_data = self.load_image(self.current_image_path)
        
        # Print analysis
        self.print_image_info()
        
        # Update viewer
        if self.viewer is not None:
            self.viewer.layers.clear()
            self.viewer.add_image(
                self.image_data,
                name=f"Mask {self.current_index + 1}",
                colormap='magma'
            )
            # Update title
            self.viewer.title = (
                f"Mask Viewer - [{self.current_index + 1}/{len(self.image_list)}] "
                f"{self.current_image_path.name}"
            )
    
    def next_image(self) -> None:
        """Display next image"""
        if self.current_index < len(self.image_list) - 1:
            self.display_image(self.current_index + 1)
        else:
            print("Already at the last image")
    
    def previous_image(self) -> None:
        """Display previous image"""
        if self.current_index > 0:
            self.display_image(self.current_index - 1)
        else:
            print("Already at the first image")
    
    def close_viewer(self) -> None:
        """Close the napari viewer"""
        if self.viewer is not None:
            self.viewer.close()
    
    def run(self) -> None:
        """Start the interactive viewer"""
        print(f"Starting Mask Viewer for: {self.folder_path}")
        print(f"Total images to view: {len(self.image_list)}\n")
        
        # Create napari viewer
        self.viewer = napari.Viewer()
        
        # Display first image
        self.display_image(0)
        
        # Connect keyboard events
        self.viewer.bind_key('Right', lambda v: self.next_image())
        self.viewer.bind_key('Left', lambda v: self.previous_image())
        self.viewer.bind_key('q', lambda v: self.close_viewer())
        
        # Start event loop
        napari.run()


def main():
    """Main entry point - Edit this to change the folder path"""
    
    # ============ EDIT THIS PATH ============
    # Set your mask image folder path here
    MASK_FOLDER = r"/home/students/rawat/training_data_new/data/train/labels/"
    # ========================================
    
    try:
        viewer = MaskImageViewer(MASK_FOLDER)
        viewer.run()
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
