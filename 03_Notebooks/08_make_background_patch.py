"""Prepare the SAME training patch with the updated three-class answer key.

Run from the project folder with ml-env active:
    python 03_Notebooks/08_make_background_patch.py

Copy the previously trained image patch unchanged. Crop its exact location
from the updated numeric mask. Do not resize, train, or replace old results.
The numeric mask stores 1=Periderm, 2=Endoderm, 3=Background; ignore 0 and 255.
The separate RGB colour preview is for viewing, not for model training.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
CODES = {0: "Unannotated (ignored)", 1: "Periderm", 2: "Endoderm",
         3: "Background", 255: "Conflicting labels (ignored)"}
COLOURS = {0: (48, 53, 59), 1: (230, 60, 60), 2: (40, 190, 80),
           3: (235, 195, 45), 255: (240, 70, 220)}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pixel_counts(labels):
    return {str(code): int(np.count_nonzero(labels == code)) for code in CODES}


def prepare_pair(trial_path, mask_details_path, output_dir):
    trial_path, mask_details_path, output_dir = (
        Path(p).resolve() for p in (trial_path, mask_details_path, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"Output folder already exists; nothing was overwritten: {output_dir}")

    trial = read_json(trial_path)
    provenance = trial["provenance"]
    details = read_json(mask_details_path)
    image_path = Path(provenance["image"]).resolve()
    old_mask_path = Path(provenance["mask"]).resolve()
    manifest_path = image_path.parent.parent / "patch_manifest.json"
    source_image_path = Path(details["image_file"]).resolve()
    full_mask_path = (mask_details_path.parent / details["mask_file"]).resolve()
    inputs = {str(p): sha256(p) for p in (
        trial_path, mask_details_path, image_path, old_mask_path,
        manifest_path, source_image_path, full_mask_path)}
    for path, expected in [(image_path, provenance["image_sha256"]),
                           (old_mask_path, provenance["mask_sha256"]),
                           (manifest_path, provenance["manifest_sha256"]),
                           (source_image_path, details["image_sha256"])]:
        if inputs[str(path)] != expected:
            raise ValueError(f"An input differs from its saved record: {path}")

    manifest = read_json(manifest_path)
    if manifest["source_image_sha256"] != details["image_sha256"]:
        raise ValueError("The updated mask and old training patch refer to different source images.")
    matches = [p for p in manifest["patches"] if p["id"] == provenance["patch_id"]]
    if len(matches) != 1:
        raise ValueError("The training patch must appear exactly once in its manifest.")
    record = matches[0]
    box = tuple(provenance["source_crop_box"])
    if (len(box) != 4 or any(type(v) is not int for v in box)
            or list(box) != record["source_crop_box"] or record["padding_pixels"] != 0):
        raise ValueError("This lesson needs the original, unpadded training crop.")
    if ((manifest_path.parent / record["image"]).resolve() != image_path
            or (manifest_path.parent / record["mask"]).resolve() != old_mask_path):
        raise ValueError("Patch filenames disagree with the manifest.")
    left, top, right, bottom = box
    size = (right - left, bottom - top)

    print("Preparing the same image patch with your updated answer key...", flush=True)
    previous_limit = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = 200_000_000
        with Image.open(full_mask_path) as full_mask, Image.open(source_image_path) as source_image:
            width, height = full_mask.size
            if (full_mask.mode != "L" or source_image.mode != "RGB"
                    or source_image.size != full_mask.size
                    or [width, height] != manifest["source_size"]
                    or (width, height) != (details["width"], details["height"])
                    or width * height > 200_000_000):
                raise ValueError("Expected matching RGB image and 8-bit numeric mask dimensions.")
            if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                raise ValueError("The crop extends outside the source image.")
            histogram = full_mask.histogram()
            actual_counts = {str(i): n for i, n in enumerate(histogram) if n}
            expected_counts = {k: v for k, v in details["pixel_counts"].items() if v}
            if actual_counts != expected_counts:
                raise ValueError("The numeric mask counts do not match mask_details.json.")
            labels = np.array(full_mask.crop(box))
            original_crop = np.array(source_image.crop(box))
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit

    with Image.open(image_path) as old_image, Image.open(old_mask_path) as old_mask:
        if (old_image.mode != "RGB" or old_mask.mode not in ("L", "P")
                or old_image.size != size or old_mask.size != size):
            raise ValueError("The old pair does not match the recorded crop size.")
        if not np.array_equal(np.asarray(old_image), original_crop):
            raise ValueError("The image patch differs from the same location in the source image.")
        old_labels = np.array(old_mask)
    if not set(np.unique(labels)).issubset(CODES):
        raise ValueError("Unexpected class number in the updated mask patch.")
    counts = pixel_counts(labels)
    if any(counts[str(code)] == 0 for code in (1, 2, 3)):
        raise ValueError("This practice patch needs examples of all three classes.")
    # This lesson changes only previously unlabelled pixels to Background.
    changed = labels != old_labels
    if np.any(changed & ~((old_labels == 0) & (labels == 3))):
        raise ValueError("Other labels changed in this patch. Review them before comparing training runs.")

    stem = provenance["patch_id"]
    image_name, mask_name = stem + "_image.png", stem + "_mask.png"
    preview_name = stem + "_mask_preview.png"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".background-patch-", dir=output_dir.parent))
    try:
        shutil.copy2(image_path, temporary / image_name)
        Image.fromarray(labels).save(temporary / mask_name)
        palette = np.zeros((256, 3), dtype=np.uint8)
        for code, colour in COLOURS.items():
            palette[code] = colour
        Image.fromarray(palette[labels]).save(temporary / preview_name)
        with Image.open(temporary / mask_name) as saved:
            if saved.mode != "L" or not np.array_equal(np.asarray(saved), labels):
                raise ValueError("Saving changed the numerical mask labels.")
        if sha256(temporary / image_name) != provenance["image_sha256"]:
            raise ValueError("Copying changed the image patch.")
        if any(sha256(path) != digest for path, digest in inputs.items()):
            raise ValueError("An input changed during preparation; no result was published.")
        metadata = {
            "purpose": "One matching pair for three-class practice; no training occurred",
            "patch_id": stem, "source_crop_box": list(box), "size": list(size),
            "image_file": image_name, "mask_file": mask_name,
            "mask_preview_file": preview_name,
            "image_sha256": sha256(temporary / image_name),
            "mask_sha256": sha256(temporary / mask_name),
            "inputs_sha256": inputs,
            "trial_record": str(trial_path), "full_mask": str(full_mask_path),
            "mask_details": str(mask_details_path),
            "script_sha256": sha256(Path(__file__)),
            "pixel_codes": CODES, "pixel_counts": counts,
            "old_mask_pixel_counts": pixel_counts(old_labels),
            "changed_pixels": int(np.count_nonzero(changed)),
            "change_rule": "Only old 0 pixels become explicit Background=3",
            "image_identical_to_previous_training_input": True,
            "trainable_class_codes": [1, 2, 3], "ignore_codes": [0, 255],
            "mask_reading": "Use numeric PNG values, not RGB colour-preview values",
            "model_trained": False,
            "limitation": "Same patch and slide as earlier training; not an independent test"
        }
        (temporary / "patch_details.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (temporary / "READ_ME.txt").write_text(
            "ONE MATCHING PATCH PAIR\n\n"
            f"Image: {image_name}\nNumeric answer key: {mask_name}\n"
            f"Colour preview for viewing: {preview_name}\n\n"
            "The image is unchanged from earlier training. The updated mask is cut\n"
            "from exactly the same location. No resizing or stretching occurred.\n"
            "1 = Periderm; 2 = Endoderm; 3 = Background. Ignore 0 and 255.\n"
            "No model has been trained. The earlier training program uses two\n"
            "classes; we will explain and prepare a three-class model separately.\n")
        if output_dir.exists():
            raise FileExistsError(f"Output appeared during preparation: {output_dir}")
        temporary.rename(output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    print(f"Created ONE matching pair: {size[0]} x {size[1]} pixels each.")
    print("Image pixels are identical to the patch used for the earlier training.")
    print("Only previously unlabelled pixels inside your Background outline changed category.")
    print(f"Image patch: {output_dir / image_name}")
    print(f"Numeric mask patch: {output_dir / mask_name}")
    print(f"Colour preview: {output_dir / preview_name}")
    for code, name in CODES.items():
        print(f"  {code}: {name} = {counts[str(code)]:,} pixels")
    print("Original files and earlier results unchanged. No model has been trained.")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, default=PROJECT / "04_Results/04_First_Training_Trial/run_20261113T125650Z_dqez5187/trial_details.json")
    parser.add_argument("--mask-details", type=Path, default=PROJECT / "04_Results/06_Background_Mask/mask_details.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "04_Results/07_Background_Patch")
    args = parser.parse_args()
    try:
        prepare_pair(args.trial, args.mask_details, args.output)
    except (ValueError, KeyError, TypeError, OSError, Image.DecompressionBombError) as error:
        parser.exit(1, f"Stopped: {error}\n")


if __name__ == "__main__":
    main()
