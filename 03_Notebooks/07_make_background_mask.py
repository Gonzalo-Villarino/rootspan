"""Make ONE pixel mask containing Periderm, Endoderm and Background.

From the project folder with ml-env active, run:
    python 03_Notebooks/07_make_background_mask.py

The mask stores 0=unannotated, 1=Periderm, 2=Endoderm, 3=Background,
255=conflicting annotations. Only explicitly labelled regions receive a class.
This fills saved human outlines; it does not train or run a model.
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
LABEL_CODES = {"Periderm": 1, "Endoderm": 2, "Background": 3}
# Internal flags are separate from the saved class numbers. In particular,
# Periderm + Endoderm overlap must NOT accidentally become Background (3).
LABEL_FLAGS = {"Periderm": 1, "Endoderm": 2, "Background": 4}
CODE_NAMES = {
    0: "Unannotated (ignored)",
    1: "Periderm",
    2: "Endoderm",
    3: "Background",
    255: "Conflicting labels (ignored)",
}
PREVIEW_COLOURS = {
    0: (48, 53, 59),
    1: (230, 60, 60),
    2: (40, 190, 80),
    3: (235, 195, 45),
    255: (240, 70, 220),
}


def validate(data):
    """Reject invalid or unexpected annotations before writing output."""
    if not isinstance(data, dict):
        raise ValueError("The annotation JSON must contain an object.")
    width, height = data.get("imageWidth"), data.get("imageHeight")
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError("The JSON needs positive integer image dimensions.")
    if width * height > 200_000_000:
        raise ValueError("This learning program supports up to 200 million pixels.")
    shapes = data.get("shapes")
    if not isinstance(shapes, list) or not shapes:
        raise ValueError("No saved outlines were found.")
    for i, shape in enumerate(shapes, 1):
        if not isinstance(shape, dict) or shape.get("label") not in LABEL_CODES:
            raise ValueError(f"Outline {i} must be labelled Periderm, Endoderm or Background.")
        kind = shape.get("shape_type", "polygon")
        points = shape.get("points")
        if kind not in ("polygon", "rectangle"):
            raise ValueError(f"Outline {i} must be a polygon or rectangle, not {kind!r}.")
        if not isinstance(points, list) or (kind == "polygon" and len(points) < 3) or (kind == "rectangle" and len(points) != 2):
            raise ValueError(f"Outline {i} has the wrong number of points.")
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in point):
                raise ValueError(f"Outline {i} has an invalid point.")
            x, y = point
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"Outline {i} has a point outside the image.")
        if len({x for x, y in points}) < 2 or len({y for x, y in points}) < 2:
            raise ValueError(f"Outline {i} has no width or height.")
    return width, height


def make_mask(data):
    """Fill class unions with Pillow, preserving all cross-class conflicts.

    Drawing uses the saved floating-point coordinates, as in our first mask
    program. Same-class overlap stays that class. Any two or three different
    classes at a pixel become 255, independently of drawing order.
    """
    width, height = validate(data)
    flags = np.zeros((height, width), dtype=np.uint8)
    for label, flag in LABEL_FLAGS.items():
        region = Image.new("1", (width, height), 0)
        draw = ImageDraw.Draw(region)
        for shape in data["shapes"]:
            if shape["label"] != label:
                continue
            pts = [tuple(p) for p in shape["points"]]
            if shape.get("shape_type", "polygon") == "rectangle":
                (x1, y1), (x2, y2) = pts
                draw.rectangle((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), outline=1, fill=1)
            else:
                draw.polygon(pts, outline=1, fill=1)
        inside = np.array(region, dtype=bool)
        flags[inside] |= flag
        del inside, draw, region
    # Indices: none, P, E, P+E, B, P+B, E+B, P+E+B.
    lookup = np.array([0, 1, 2, 255, 3, 255, 255, 255], dtype=np.uint8)
    return lookup[flags]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(annotation_path, output_dir):
    annotation_path, output_dir = annotation_path.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output folder already exists; nothing overwritten: {output_dir}")
    source_hash = sha256(annotation_path)
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    width, height = validate(data)
    shape_counts = Counter(s["label"] for s in data["shapes"])
    missing = set(LABEL_CODES) - set(shape_counts)
    if missing:
        raise ValueError(f"This lesson needs saved examples of all three classes. Missing: {', '.join(sorted(missing))}.")
    data.pop("imageData", None)  # Embedded image bytes are unnecessary here.
    image_name = data.get("imagePath")
    if not isinstance(image_name, str) or not image_name:
        raise ValueError("The JSON does not name its microscopy image.")
    image_path = (annotation_path.parent / image_name).resolve()
    old_limit = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = max(old_limit or 0, width * height)
        with Image.open(image_path) as original:
            if original.size != (width, height):
                raise ValueError("The image dimensions do not match the annotations.")
    finally:
        Image.MAX_IMAGE_PIXELS = old_limit
    image_hash = sha256(image_path)
    print("Filling your saved Periderm, Endoderm and Background outlines...", flush=True)
    print("Creating ONE mask with three classes. No model is being trained.", flush=True)
    mask = make_mask(data)
    counts = {code: int(np.count_nonzero(mask == code)) for code in CODE_NAMES}
    for label, code in LABEL_CODES.items():
        if counts[code] == 0:
            raise ValueError(f"No unambiguous {label} pixels remain; check the outlines and overlaps.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".background-mask-building-", dir=output_dir.parent))
    try:
        stem = annotation_path.stem
        mask_name, preview_name = f"{stem}_mask.png", f"{stem}_mask_preview.png"
        mask_image = Image.fromarray(mask)
        mask_image.save(temporary / mask_name)
        scale = min(1, 1600 / max(width, height))
        small = mask_image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.NEAREST)
        palette = np.zeros((256, 3), dtype=np.uint8)
        for code, colour in PREVIEW_COLOURS.items():
            palette[code] = colour
        Image.fromarray(palette[np.asarray(small)]).save(temporary / preview_name)
        metadata = {
            "annotation_file": str(annotation_path), "annotation_sha256": source_hash,
            "image_file": str(image_path), "image_sha256": image_hash,
            "script_file": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__)),
            "width": width, "height": height, "saved_outlines": dict(shape_counts),
            "mask_file": mask_name, "mask_format": "8-bit grayscale PNG; pixel values are label codes, not image brightness",
            "pixel_codes": CODE_NAMES, "pixel_counts": counts, "trainable_class_codes": [1, 2, 3],
            "exclude_from_future_training": [0, 255],
            "overlap_policy": "Union within each class; pixels in two or three different classes receive 255, independently of outline order.",
            "unannotated_note": "0 means unannotated, not Background. Background=3 appears only inside explicitly labelled Background outlines.",
            "preview_file": preview_name,
            "preview_colours": {"grey": "Unannotated", "red": "Periderm", "green": "Endoderm", "yellow": "Background", "pink": "Conflicting labels"},
            "preview_note": "Reduced-size viewing picture only. Small or thin regions may disappear. Use the full-size numeric mask for training and counts.",
            "model_trained": False,
        }
        (temporary / "mask_details.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (temporary / "READ_ME.txt").write_text(
            "ONE MASK, THREE CLASSES\n\n"
            "The *_mask.png file is the full-size numeric answer key: 1 Periderm, 2 Endoderm, 3 Background.\n"
            "0 is unannotated and 255 means conflicting labels. Both must be ignored during training.\n"
            "The *_mask_preview.png is only a smaller colour picture: red Periderm, green Endoderm,\n"
            "yellow Background, grey unannotated, pink conflict. Small regions can disappear in this preview.\n"
            "mask_details.json records the inputs, hashes, class codes and pixel counts.\n\n"
            "This program filled your saved outlines. It did not train or change a model or update patches.\n"
            "The earlier two-class training scripts cannot use this new mask without adaptation.\n",
            encoding="utf-8",
        )
        if sha256(annotation_path) != source_hash or sha256(image_path) != image_hash:
            raise ValueError("An input changed during conversion. Please run again.")
        if output_dir.exists():
            raise FileExistsError("Output folder appeared during conversion; nothing overwritten.")
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Mask saved: {output_dir / mask_name}")
    print(f"Colour preview: {output_dir / preview_name}")
    print(f"Same image dimensions: {width} x {height} pixels")
    for code, label in CODE_NAMES.items():
        print(f"  {code}: {label} = {counts[code]:,} pixels")
    print("Preview colours: red Periderm; green Endoderm; yellow Background; grey unannotated; pink conflict.")
    print("The small preview can hide thin regions; the full-size mask preserves them.")
    print("Unannotated pixels remain unannotated. Only your Background outlines supply background labels.")
    print("No model trained or patches updated. Original files and earlier results unchanged.")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=PROJECT / "02_Data/03_Background_Annotations/MT25_rep3_background_GV.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "04_Results/06_Background_Mask")
    args = parser.parse_args()
    try:
        convert(args.annotations, args.output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Stopped: {error}\n")


if __name__ == "__main__":
    main()
