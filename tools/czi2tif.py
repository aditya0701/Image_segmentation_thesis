#!/usr/bin/env python3
"""
czi_to_ome_tiff_safe.py
Convert CZI → OME-TIFF preserving ALL dimensions (T, C, Z, Y, X, S, etc.)
Uses aicsimageio – reliable for confocal microscopy data.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter

def convert_czi_to_ome_tiff(input_path: Path, output_path: Path, overwrite: bool = False):
    if output_path.exists() and not overwrite:
        print(f"Skipping (exists): {output_path.name}")
        return False

    print(f"Reading: {input_path.name}")
    try:
        # aicsimageio auto-detects and loads ALL dimensions
        img = AICSImage(input_path)
        
        # Print full dimension info (for debugging)
        print(f"  Shape: {img.shape}")
        print(f"  Dims:  {img.dims.order}")  # e.g., 'TCZYX', 'SCZYX', etc.
        print(f"  dtype: {img.dtype}")
        print(f"  Size:  {input_path.stat().st_size / 1024:.1f} KB")

        # Get full 5D+ array: (T, C, Z, Y, X) – extra dims are squeezed or kept
        data = img.get_image_data("TCZYX", S=0, M=0)  # S=0, M=0 → first scene/mosaic
        # If you have scenes, loop over S and save separately!

        print(f"  Writing: {output_path.name} | Shape: {data.shape} | dtype: {data.dtype}")

        # Write as OME-TIFF with full metadata
        OmeTiffWriter.save(
            data,
            output_path,
            dim_order="TCZYX",  # Force standard order
            overwrite=overwrite
        )
        print(f"  Success: {output_path.stat().st_size / 1024:.1f} KB")
        return True

    except Exception as e:
        print(f"  Failed: {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert CZI to OME-TIFF (preserves dimensions)")
    parser.add_argument("input", help="Input folder with .czi files")
    parser.add_argument("-o", "--output", help="Output folder (default: same as input)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output) if args.output else in_dir

    if not in_dir.is_dir():
        print(f"Error: {in_dir} not found")
        sys.exit(1)

    czi_files = list(in_dir.glob("*.czi")) + list(in_dir.glob("*.CZI"))
    if not czi_files:
        print("No .czi files found")
        sys.exit(0)

    out_dir.mkdir(parents=True, exist_ok=True)
    success = 0

    for czi_path in czi_files:
        out_path = out_dir / (czi_path.stem + ".ome.tif")
        if convert_czi_to_ome_tiff(czi_path, out_path, args.overwrite):
            success += 1

    print(f"\nDone: {success}/{len(czi_files)} converted")

if __name__ == "__main__":
    main()