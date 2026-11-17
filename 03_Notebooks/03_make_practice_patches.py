"""Cut one annotated slide and its mask into matching 1024-pixel squares.

Keep every grid square containing at least one Periderm/Endoderm label. No
resizing, smoothing, augmentation, training, or train/test split is performed.
All patches share one source slide: they are not independent slides.

Read saved masks as palette indices, NOT RGB. Codes 0 (unannotated, including
edge padding) and 255 (uncertain overlap) require explicit exclusion from
future loss/metrics. This exercise does not yet teach a background class.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile

import numpy as np
from PIL import Image, PngImagePlugin


PROJECT = Path(__file__).resolve().parents[1]
CODES = {0: "Unannotated or padding", 1: "Periderm", 2: "Endoderm", 255: "Uncertain overlap"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_rgb_png(path):
    """Avoid silently down-converting a 16-bit microscopy PNG in Pillow."""
    with Path(path).open("rb") as stream:
        header = stream.read(33)
    if len(header) != 33 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("The original image must be a PNG for this lesson.")
    _, _, depth, colour, _, _, _ = struct.unpack(">IIBBBBB", header[16:29])
    if depth != 8 or colour != 2:
        raise ValueError("This lesson requires an 8-bit-per-channel RGB PNG; no automatic conversion is performed.")


def make_patches(image_path, mask_path, output_dir, size=1024):
    image_path, mask_path, output_dir = (Path(p).resolve() for p in (image_path, mask_path, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"Output folder already exists; nothing was overwritten: {output_dir}")
    if type(size) is not int or not (1 <= size <= 4096):
        raise ValueError("Patch size must be a positive integer no larger than 4096.")
    verify_rgb_png(image_path)
    image_hash, mask_hash = sha256(image_path), sha256(mask_path)
    previous_limit = Image.MAX_IMAGE_PIXELS
    temporary = None
    try:
        Image.MAX_IMAGE_PIXELS = 200_000_000
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise ValueError("The image and mask dimensions do not match.")
            width, height = image.size
            if width * height > 200_000_000:
                raise ValueError("This lesson supports images up to 200 million pixels.")
            if image.mode != "RGB" or mask.mode not in ("L", "P"):
                raise ValueError("Expected an RGB image and an 8-bit label-code mask.")
            histogram = mask.histogram()
            if any(count and code not in CODES for code, count in enumerate(histogram)):
                raise ValueError("The source mask contains unexpected class codes.")
            if histogram[1] + histogram[2] == 0:
                raise ValueError("The mask contains no Periderm or Endoderm pixels.")
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".practice-patches-", dir=output_dir.parent))
            (temporary / "images").mkdir()
            (temporary / "masks").mkdir()
            palette = np.zeros((256, 3), dtype=np.uint8)
            palette[0], palette[1], palette[2], palette[255] = (25, 25, 25), (230, 60, 60), (40, 190, 80), (240, 70, 220)
            records, skipped = [], 0
            retained = {code: 0 for code in CODES}
            print("Cutting matching squares; keeping those with your labelled tissue...", flush=True)
            for row, top in enumerate(range(0, height, size)):
                for column, left in enumerate(range(0, width, size)):
                    right, bottom = min(left + size, width), min(top + size, height)
                    box = (left, top, right, bottom)
                    labels = np.asarray(mask.crop(box))
                    counts = {code: int(np.count_nonzero(labels == code)) for code in CODES}
                    if counts[1] + counts[2] == 0:
                        skipped += 1
                        continue
                    valid_width, valid_height = right - left, bottom - top
                    # Last row/column may be short: add empty space, never stretch.
                    image_patch = Image.new("RGB", (size, size), (0, 0, 0))
                    image_patch.paste(image.crop(box), (0, 0))
                    padded = np.zeros((size, size), dtype=np.uint8)
                    padded[:valid_height, :valid_width] = labels
                    mask_patch = Image.fromarray(padded)
                    mask_patch.putpalette(palette.ravel().tolist())
                    stem = f"{image_path.stem}_r{row:03d}_c{column:03d}"
                    filename = stem + ".png"
                    image_info, mask_info = PngImagePlugin.PngInfo(), PngImagePlugin.PngInfo()
                    for info in (image_info, mask_info):
                        info.add_text("Source image", image_path.name)
                        info.add_text("Source image SHA256", image_hash)
                        info.add_text("Source mask SHA256", mask_hash)
                        info.add_text("Crop box left,top,right,bottom", ",".join(map(str, box)))
                        info.add_text("Padding right,bottom", f"{size-valid_width},{size-valid_height}")
                    mask_info.add_text("Label codes", "0=unannotated or padding; 1=Periderm; 2=Endoderm; 255=uncertain overlap")
                    mask_info.add_text("Future training", "Read palette indices; exclude codes 0 and 255 from loss and metrics")
                    image_patch.save(temporary / "images" / filename, pnginfo=image_info)
                    mask_patch.save(temporary / "masks" / filename, pnginfo=mask_info, bits=8)
                    with Image.open(temporary / "masks" / filename) as saved:
                        if saved.mode != "P" or not np.array_equal(np.asarray(saved), padded):
                            raise ValueError(f"Saved mask labels changed for {filename}.")
                    for code in CODES:
                        retained[code] += counts[code]
                    records.append({
                        "id": stem, "row": row, "column": column,
                        "image": f"images/{filename}", "mask": f"masks/{filename}",
                        "source_crop_box": list(box), "valid_width": valid_width,
                        "valid_height": valid_height,
                        "padding_right": size - valid_width, "padding_bottom": size - valid_height,
                        "padding_pixels": size * size - valid_width * valid_height,
                        "source_pixel_counts_excluding_padding": counts,
                    })
            # Every labelled tissue pixel must be included once, even at edges.
            if any(retained[code] != histogram[code] for code in (1, 2)):
                raise ValueError("Some annotated tissue pixels were lost or duplicated.")
            if sha256(image_path) != image_hash or sha256(mask_path) != mask_hash:
                raise ValueError("A source file changed during processing. Please run again.")
            manifest = {
                "purpose": "Single-slide learning exercise; no model trained and no dataset split",
                "source_image": str(image_path), "source_mask": str(mask_path),
                "source_image_sha256": image_hash, "source_mask_sha256": mask_hash,
                "script_sha256": sha256(Path(__file__)), "source_size": [width, height],
                "patch_size": size, "grid_squares": len(records) + skipped,
                "kept_pairs": len(records), "skipped_squares_without_tissue_labels": skipped,
                "selection_rule": "Keep a grid square if it contains at least one pixel with code 1 or 2",
                "padding_rule": "Non-overlapping source grid; right/bottom edges padded with black image pixels and mask code 0; no resizing",
                "label_codes": CODES, "ignore_codes_for_future_training": [0, 255],
                "mask_reading": "Indexed PNG: read numerical palette indices, not RGB colours",
                "source_label_counts": {code: histogram[code] for code in CODES},
                "retained_source_label_counts_excluding_padding": retained,
                "split_caution": "All patches come from ONE slide. Do not treat random patch splits as evidence of performance on independent slides. The separate first-patch demonstration is not added here.",
                "patches": records,
            }
            (temporary / "patch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.rename(output_dir)
        temporary = None
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    padded_count = sum(bool(r["padding_pixels"]) for r in records)
    print(f"Created {len(records)} matching patch pairs, each {size} x {size} pixels.")
    print(f"Skipped {skipped} squares without Periderm or Endoderm labels.")
    print(f"Added empty space at the slide edge for {padded_count} pairs; no stretching.")
    print(f"Image patches: {output_dir / 'images'}")
    print(f"Mask patches:  {output_dir / 'masks'}")
    print(f"Record of patch locations: {output_dir / 'patch_manifest.json'}")
    print("All Periderm and Endoderm labels were retained once. Original files unchanged.")
    print("No model trained. All examples still come from your one slide.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=PROJECT / "02_Data/02_My_First_Annotations/MT25_rep3_practice.png")
    parser.add_argument("--mask", type=Path, default=PROJECT / "04_Results/01_Practice_Mask/MT25_rep3_practice_GV_mask.png")
    parser.add_argument("--output", type=Path, default=PROJECT / "04_Results/03_Practice_Patches")
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()
    try:
        make_patches(args.image, args.mask, args.output, args.size)
    except (ValueError, OSError, Image.DecompressionBombError) as error:
        parser.exit(1, f"Stopped: {error}\n")


if __name__ == "__main__":
    main()
