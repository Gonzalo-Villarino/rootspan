"""Save one matching image/mask crop, without resizing or model training.

Run from the project folder with labelme-env active:
    python 03_Notebooks/02_make_one_patch.py

The mask is an indexed PNG: its stored numbers remain 0, 1, 2 and 255.
A display palette makes it look coloured in an image viewer. Future training
must read the mask's indices, NOT convert it to RGB or divide its labels by 255.
Codes 0 (unannotated) and 255 (uncertain overlap) are not known background;
exclude both explicitly from future training loss/metrics.
"""

import argparse
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image, PngImagePlugin


PROJECT = Path(__file__).resolve().parents[1]
# One square near the manually annotated transition in the first (top) root.
# Coordinates are (left, top, right, bottom), with right/bottom excluded.
CROP_BOX = (7280, 0, 8304, 1024)
IMAGE_NAME = "MT25_rep3_patch_001_image.png"
MASK_NAME = "MT25_rep3_patch_001_mask.png"


def make_pair(image_path, mask_path, output_dir, box):
    image_path, mask_path, output_dir = (Path(p).resolve() for p in (image_path, mask_path, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"The output folder already exists; nothing was overwritten: {output_dir}")
    if len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError("The crop needs four integer coordinates.")
    left, top, right, bottom = box
    previous_limit = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = 200_000_000
        with Image.open(image_path) as source_image, Image.open(mask_path) as source_mask:
            if source_image.size != source_mask.size:
                raise ValueError("The image and mask must have identical dimensions.")
            width, height = source_image.size
            if width * height > 200_000_000:
                raise ValueError("This lesson supports images up to 200 million pixels.")
            if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                raise ValueError("The crop must lie entirely inside the image and mask.")
            if source_image.mode != "RGB" or source_mask.mode not in ("L", "P"):
                raise ValueError("Expected an RGB image and an 8-bit label-code mask.")
            print("Cutting the same square from your image and mask...", flush=True)
            image_patch = source_image.crop(box)
            mask_patch = source_mask.crop(box)
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
    labels = np.array(mask_patch)
    if not set(np.unique(labels)).issubset({0, 1, 2, 255}):
        raise ValueError("The mask contains unexpected label codes.")
    # Attach colours without changing any stored pixel-label numbers.
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[0], palette[1], palette[2], palette[255] = (25, 25, 25), (230, 60, 60), (40, 190, 80), (240, 70, 220)
    mask_patch.putpalette(palette.ravel().tolist())
    image_info, mask_info = PngImagePlugin.PngInfo(), PngImagePlugin.PngInfo()
    for info in (image_info, mask_info):
        info.add_text("Original image", image_path.name)
        info.add_text("Original mask", mask_path.name)
        info.add_text("Crop box left,top,right,bottom", ",".join(map(str, box)))
        info.add_text("Preparation", "Matching crop; no resizing or smoothing; no model training")
    mask_info.add_text("Label codes", "0=unannotated; 1=Periderm; 2=Endoderm; 255=uncertain overlap")
    mask_info.add_text("Future training", "Read palette indices; exclude 0 and 255 from loss/metrics")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".patch-building-", dir=output_dir.parent))
    try:
        image_patch.save(temporary / IMAGE_NAME, pnginfo=image_info)
        mask_patch.save(temporary / MASK_NAME, pnginfo=mask_info, bits=8)
        # Check that saving the viewable mask preserved every numerical label.
        with Image.open(temporary / MASK_NAME) as saved:
            if saved.mode != "P" or not np.array_equal(np.asarray(saved), labels):
                raise ValueError("Mask pixel codes did not survive saving correctly.")
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Created one matching pair: {right - left} x {bottom - top} pixels each.")
    print(f"Image patch: {output_dir / IMAGE_NAME}")
    print(f"Mask patch:  {output_dir / MASK_NAME}")
    print("Original files unchanged. No model has been trained.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=PROJECT / "02_Data/02_My_First_Annotations/MT25_rep3_practice.png")
    parser.add_argument("--mask", type=Path, default=PROJECT / "04_Results/01_Practice_Mask/MT25_rep3_practice_GV_mask.png")
    parser.add_argument("--output", type=Path, default=PROJECT / "04_Results/02_First_Patch")
    parser.add_argument("--box", type=int, nargs=4, default=CROP_BOX, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    args = parser.parse_args()
    try:
        make_pair(args.image, args.mask, args.output, tuple(args.box))
    except (ValueError, OSError, Image.DecompressionBombError) as error:
        parser.exit(1, f"Stopped: {error}\n")


if __name__ == "__main__":
    main()
