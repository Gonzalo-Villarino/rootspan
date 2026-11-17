"""Turn saved Labelme polygons into a separate pixel mask (no model training).

Run from the project folder after activating labelme-env:
    python 03_Notebooks/01_make_pixel_mask.py

Pixel codes: 0 unannotated, 1 Periderm, 2 Endoderm, 255 conflicting labels.
Both 0 and 255 need explicit exclusion from a future training loss; neither
is a verified background class. The small colour preview is for viewing only.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image, ImageDraw


PROJECT = Path(__file__).resolve().parents[1]
LABEL_CODES = {"Periderm": 1, "Endoderm": 2}
CODE_NAMES = {0: "Unannotated", 1: "Periderm", 2: "Endoderm", 255: "Overlap: uncertain"}


def validate(data):
    """Check the saved outlines before creating any output files."""
    width, height = data.get("imageWidth"), data.get("imageHeight")
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError("The JSON must contain positive integer image dimensions.")
    if width * height > 200_000_000:
        raise ValueError("This learning script supports images up to 200 million pixels.")
    shapes = data.get("shapes")
    if not isinstance(shapes, list) or not shapes:
        raise ValueError("No saved outlines were found in this JSON.")
    for i, shape in enumerate(shapes, 1):
        if shape.get("label") not in LABEL_CODES:
            raise ValueError(f"Outline {i} has an unexpected tissue name: {shape.get('label')!r}.")
        if shape.get("shape_type", "polygon") != "polygon":
            raise ValueError(f"Outline {i} is not a polygon.")
        points = shape.get("points", [])
        if len(points) < 3:
            raise ValueError(f"Outline {i} needs at least three points.")
        for point in points:
            if len(point) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in point):
                raise ValueError(f"Outline {i} has an invalid point.")
            x, y = point
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"Outline {i} has a point outside the image.")
    return width, height


def make_mask(data):
    """Fill polygons using Pillow, as the installed Labelme polygon helper does.

    Build the union of each tissue separately. Bit flags preserve cross-tissue
    conflicts independently of polygon order: 1 | 2 = 3, later changed to 255.
    """
    width, height = validate(data)
    mask = np.zeros((height, width), dtype=np.uint8)
    for label, code in LABEL_CODES.items():
        region = Image.new("1", (width, height), 0)
        draw = ImageDraw.Draw(region)
        for shape in data["shapes"]:
            if shape["label"] == label:
                draw.polygon([tuple(p) for p in shape["points"]], outline=1, fill=1)
        inside = np.array(region, dtype=bool)
        mask[inside] |= code
        del inside, draw, region
    mask[mask == 3] = 255
    return mask


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(annotation_path, output_dir):
    annotation_path, output_dir = annotation_path.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output folder already exists; nothing was overwritten: {output_dir}")
    source_hash = sha256(annotation_path)
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    data.pop("imageData", None)  # We only need coordinates, labels, and dimensions.
    width, height = validate(data)
    if not isinstance(data.get("imagePath"), str) or not data["imagePath"]:
        raise ValueError("The JSON does not name its original image.")
    image_path = (annotation_path.parent / data["imagePath"]).resolve()
    # Inspect the original image header without loading or modifying its pixels.
    # Its known 147-million-pixel size exceeds Pillow's default warning threshold.
    old_limit = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = max(old_limit or 0, width * height)
        with Image.open(image_path) as original:
            if original.size != (width, height):
                raise ValueError("The JSON dimensions do not match the original image.")
    finally:
        Image.MAX_IMAGE_PIXELS = old_limit
    image_hash = sha256(image_path)
    print("Filling your saved outlines. No model is being trained.", flush=True)
    mask = make_mask(data)
    counts = {code: int(np.count_nonzero(mask == code)) for code in CODE_NAMES}
    if sha256(annotation_path) != source_hash or sha256(image_path) != image_hash:
        raise ValueError("An input file changed while the program was running. Please run again.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".mask-building-", dir=output_dir.parent))
    try:
        stem = annotation_path.stem
        mask_name = f"{stem}_mask.png"
        preview_name = f"{stem}_mask_preview.png"
        mask_image = Image.fromarray(mask)
        mask_image.save(temporary / mask_name)
        # Nearest-neighbour resizing preserves codes in the viewing-only preview.
        scale = min(1, 1600 / max(width, height))
        small = mask_image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.NEAREST)
        palette = np.zeros((256, 3), dtype=np.uint8)
        palette[0], palette[1], palette[2], palette[255] = (25, 25, 25), (230, 60, 60), (40, 190, 80), (240, 70, 220)
        Image.fromarray(palette[np.asarray(small)]).save(temporary / preview_name)
        metadata = {
            "annotation_file": str(annotation_path), "annotation_sha256": source_hash,
            "image_file": str(image_path), "image_sha256": image_hash,
            "script_sha256": sha256(Path(__file__)),
            "width": width, "height": height,
            "saved_outlines": dict(Counter(s["label"] for s in data["shapes"])),
            "mask_file": mask_name, "mask_format": "8-bit grayscale PNG; values are class codes, not brightness measurements",
            "pixel_codes": CODE_NAMES, "pixel_counts": counts,
            "exclude_from_future_training": [0, 255],
            "overlap_policy": "Union within each tissue; pixels covered by both tissues receive 255. Polygon order does not decide tissue.",
            "unannotated_note": "0 is unannotated, not confirmed background.",
            "preview_file": preview_name,
            "preview_note": "Smaller colour picture for viewing only; thin or small regions may be hidden by downsampling. Use the full-size mask and its pixel counts for pixel labels.",
            "preview_colours": {"dark grey": "Unannotated", "red": "Periderm", "green": "Endoderm", "pink": "Overlap: uncertain"},
            "model_trained": False,
        }
        (temporary / "mask_details.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Mask saved: {output_dir / mask_name}")
    print(f"Colour preview: {output_dir / preview_name}")
    print(f"Same image dimensions: {width} x {height} pixels")
    for code, label in CODE_NAMES.items():
        print(f"  {code}: {label} = {counts[code]:,} pixels")
    print("The small preview can hide thin regions; the full-size mask and counts preserve them.")
    if counts[255]:
        print("Some pixels are inside both tissue outlines. They are marked 255 (uncertain), even if hidden in the small preview.")
    print("Original image and annotations unchanged. No model has been trained.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=PROJECT / "02_Data/02_My_First_Annotations/MT25_rep3_practice_GV.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "04_Results/01_Practice_Mask")
    args = parser.parse_args()
    try:
        convert(args.annotations, args.output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Stopped: {error}\n")


if __name__ == "__main__":
    main()
