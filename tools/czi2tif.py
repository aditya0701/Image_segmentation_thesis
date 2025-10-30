#!/usr/bin/env python3
"""
czi2tif.py

Batch converter from ZEISS CZI to multi-page TIFF.
Converts ALL .czi files in a folder automatically.
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# 1. CZI reader – only czifile (pure Python, reliable)
# ------------------------------------------------------------
try:
    from czifile import CziFile
except Exception:               # pragma: no cover
    CziFile = None

try:
    import tifffile
except Exception:               # pragma: no cover
    print("tifffile is required. Install with: poetry add tifffile", file=sys.stderr)
    sys.exit(2)


def read_czi(path: Path):
    """Read a CZI file and return (array, axes-string)."""
    if CziFile is None:
        raise RuntimeError("czifile is required. Run: poetry add czifile")

    with CziFile(str(path)) as czi:
        array = czi.asarray()
        axes = getattr(czi, "axes", None)
        if axes is None:
            axes = "?" * array.ndim
        return np.asarray(array), "".join(axes).upper()


# ------------------------------------------------------------
# 2. Helper: pack arbitrary CZI dimensions into (frames, Y, X)
# ------------------------------------------------------------
def pack_frames(array, axes):
    axes = axes.upper()

    # locate Y and X
    if "Y" in axes and "X" in axes:
        y_idx, x_idx = axes.index("Y"), axes.index("X")
    else:
        y_idx, x_idx = array.ndim - 2, array.ndim - 1
        axes = ("?" * (array.ndim - 2)) + "YX"

    # transpose so that Y and X are the last two axes
    if not (y_idx == array.ndim - 2 and x_idx == array.ndim - 1):
        other = [i for i in range(array.ndim) if i not in (y_idx, x_idx)]
        order = other + [y_idx, x_idx]
        array = np.transpose(array, axes=order)
        axes = "".join(axes[i] for i in order)

    # reshape to (frames, Y, X)
    if array.ndim == 2:
        frames = array.reshape(1, array.shape[0], array.shape[1])
        frames_axes = ""
    else:
        leading = array.shape[:-2]
        nframes = int(np.prod(leading))
        frames = array.reshape((nframes, array.shape[-2], array.shape[-1]))
        frames_axes = axes[:-2]

    return frames, frames_axes


# ------------------------------------------------------------
# 3. Write multi-page TIFF
# ------------------------------------------------------------
def write_tiff(frames, out_path, compress="zlib", bigtiff=False):
    compression = None if compress in (None, "none", "None") else compress
    tifffile.imwrite(
        str(out_path),
        frames,
        photometric="minisblack",
        compression=compression,
        bigtiff=bigtiff,
    )


# ------------------------------------------------------------
# 4. Process a single file
# ------------------------------------------------------------
def process_file(
    in_path: Path,
    out_dir: Path,
    compress="zlib",
    bigtiff=False,
    overwrite=False,
):
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_path = out_dir / (in_path.stem + ".tif")

    if out_path.exists() and not overwrite:
        print(f"Skipping {in_path.name} (output exists, use --overwrite)")
        return False

    arr, axes = read_czi(in_path)
    frames, _ = pack_frames(arr, axes)

    # ---- simple dtype normalisation -------------------------------------------------
    if np.issubdtype(frames.dtype, np.floating):
        if frames.max() <= 1.0:
            frames = (frames * 65535.0).astype(np.uint16)
        elif frames.max() <= 255:
            frames = frames.astype(np.uint8)
        else:
            frames = frames.astype(np.uint16)
    elif not np.issubdtype(frames.dtype, np.integer):
        frames = frames.astype(np.uint16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_tiff(frames, out_path, compress=compress, bigtiff=bigtiff)

    print(
        f"Success: {in_path.name} → {out_path.name}  "
        f"(frames={frames.shape[0]}, Y={frames.shape[1]}, X={frames.shape[2]})"
    )
    return True


# ------------------------------------------------------------
# 5. CLI
# ------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        prog="czi2tif",
        description="Convert ALL .czi files in a folder to multi-page TIFF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  poetry run python czi2tif.py /path/to/czi/files
  poetry run python czi2tif.py ./data -o ./output --overwrite
        """,
    )
    p.add_argument("input_folder", help="Folder containing .czi files")
    p.add_argument("-o", "--output", help="Output folder (default: same as input)")
    p.add_argument(
        "--compress",
        choices=["zlib", "lzma", "none"],
        default="zlib",
        help="TIFF compression",
    )
    p.add_argument("--bigtiff", action="store_true", help="Write BigTIFF")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = p.parse_args(argv)

    input_folder = Path(args.input_folder).resolve()
    if not input_folder.is_dir():
        print(f"Error: Not a directory: {input_folder}", file=sys.stderr)
        sys.exit(1)

    output_folder = Path(args.output).resolve() if args.output else input_folder

    # case-insensitive .czi search
    czi_files = list(input_folder.glob("*.czi")) + list(input_folder.glob("*.CZI"))
    if not czi_files:
        print(f"No .czi files found in: {input_folder}")
        sys.exit(0)

    print(f"\nFound {len(czi_files)} CZI file(s)")
    print(f"Input : {input_folder}")
    print(f"Output: {output_folder}")
    print(f"Compression: {args.compress}")
    print("-" * 60)

    success = 0
    failed = []

    for i, czi_path in enumerate(czi_files, 1):
        print(f"\n[{i}/{len(czi_files)}] {czi_path.name}")
        try:
            if process_file(
                czi_path,
                output_folder,
                compress=args.compress,
                bigtiff=args.bigtiff,
                overwrite=args.overwrite,
            ):
                success += 1
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            failed.append(czi_path.name)

    print("\n" + "=" * 60)
    print(f"Done: {success}/{len(czi_files)} files converted")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print("=" * 60)


if __name__ == "__main__":
    main()