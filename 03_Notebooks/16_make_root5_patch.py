"""Cut one matching root-5 image/mask pair for Gonzalo's learning exercise.

Learning exercise. Pillow's crop() cuts the identical rectangle from the original
RGB image and the numeric mask. No resizing, prediction or training occurs.
The location comes from the saved root-5 review, not the old root-1 trial.
Run with --check-only to inspect inputs without writing patch files.
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
MASK_DETAILS = PROJECT / "04_Results/21_Root5_Mask/mask_details.json"
PROPOSAL = PROJECT / "05_Notes/Visual_Guides/Root5_Patch_Review/proposal_20261115T105528Z/proposal_record.json"
OUTPUT = PROJECT / "04_Results/22_Root5_Patch"
BOX = [6976, 6016, 8000, 7040]  # left, top, right, bottom; last two excluded
PATCH_ID = "MT25_rep3_root5_x6976_y6016"
CODES = {0: "Unannotated (ignored)", 1: "Periderm", 2: "Endoderm",
         3: "Background", 255: "Conflicting labels (ignored)"}
COLOURS = {0: (48, 53, 59), 1: (230, 60, 60), 2: (40, 190, 80),
           3: (235, 195, 45), 255: (240, 70, 220)}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unchanged(hashes):
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise ValueError(f"An input changed during preparation: {path}")


def load_pair(mask_details):
    """Read the existing mask and image; cut both at the reviewed location."""
    mask_details = Path(mask_details).resolve()
    inputs = {str(path): sha256(path) for path in (mask_details, PROPOSAL)}
    details = json.loads(mask_details.read_text())
    proposal = json.loads(PROPOSAL.read_text())
    image_path = Path(details["image_file"]).resolve()
    annotation = Path(details["annotation_file"]).resolve()
    mask_path = (mask_details.parent / details["mask_file"]).resolve()
    for path in (image_path, annotation, mask_path):
        inputs[str(path)] = sha256(path)
    if (inputs[str(image_path)] != details["image_sha256"]
            or inputs[str(annotation)] != details["annotation_sha256"]):
        raise ValueError("Image or annotations changed after the mask was made.")
    if (proposal["crop_xyxy_exclusive"] != BOX
            or proposal["image_sha256"] != details["image_sha256"]
            or Path(proposal["source_annotations"]).resolve() != annotation):
        raise ValueError("Inputs do not match the reviewed root-5 location.")
    # The proposal's older annotation hash is deliberately not required to match:
    # Gonzalo added Background shapes after that review. Use the NEW mask hash.
    if (details["trainable_class_codes"] != [1, 2, 3]
            or details["exclude_from_future_training"] != [0, 255]):
        raise ValueError("Expected classes 1/2/3 and ignored codes 0/255.")
    old_limit = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = 200_000_000
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            width, height = image.size
            left, top, right, bottom = BOX
            if (image.mode != "RGB" or mask.mode != "L"
                    or image.size != mask.size
                    or [width, height] != [details["width"], details["height"]]
                    or width * height > 200_000_000
                    or not (0 <= left < right <= width and 0 <= top < bottom <= height)):
                raise ValueError("Image, numeric mask or crop dimensions do not match.")
            counts = {str(i): n for i, n in enumerate(mask.histogram()) if n}
            if counts != {k: v for k, v in details["pixel_counts"].items() if v}:
                raise ValueError("Full mask counts disagree with its saved record.")
            rgb = np.array(image.crop(BOX))
            labels = np.array(mask.crop(BOX))
    finally:
        Image.MAX_IMAGE_PIXELS = old_limit
    if (rgb.shape != (1024, 1024, 3) or labels.shape != (1024, 1024)
            or not np.isin(labels, list(CODES)).all()):
        raise ValueError("Unexpected patch dimensions or numeric mask codes.")
    counts = {str(code): int(np.count_nonzero(labels == code)) for code in CODES}
    if not all(counts[str(code)] for code in (1, 2, 3)):
        raise ValueError("The practice pair needs examples of all three categories.")
    metadata = {
        "purpose": "New root-5 practice pair; exact matching crops; no training",
        "patch_id": PATCH_ID, "source_crop_box": BOX, "size": [1024, 1024],
        "crop_convention": "Zero-based left/top included; right/bottom excluded",
        "source_image": str(image_path), "full_mask": str(mask_path),
        "mask_details": str(mask_details), "annotation_file": str(annotation),
        "annotation_sha256": details["annotation_sha256"],
        "crop_review_record": str(PROPOSAL), "inputs_sha256": inputs,
        "pixel_codes": CODES, "pixel_counts": counts,
        "trainable_class_codes": [1, 2, 3], "ignore_codes": [0, 255],
        "labelled_percent": 100 * sum(counts[str(i)] for i in (1, 2, 3)) / labels.size,
        "change_rule": "Exact crops; no resizing, padding or automatic labelling",
        "mask_reading": "Use numeric mask codes, not colour-preview RGB values",
        "model_trained": False,
        "limitation": "Root 5 is from the same slide; not an independent test. Only labelled pixels can be scored.",
        "script_file": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
    }
    unchanged(inputs)
    return rgb, labels, metadata


def prepare_pair(mask_details, output_dir, check_only=False):
    output_dir = Path(output_dir).resolve()
    if not check_only and output_dir.exists():
        raise FileExistsError(f"Output folder already exists; nothing overwritten: {output_dir}")
    rgb, labels, metadata = load_pair(mask_details)
    for code, name in CODES.items():
        print(f"  {code}: {name} = {metadata['pixel_counts'][str(code)]:,} pixels")
    if check_only:
        print("Root 5 pair check passed. No patch files written. No training.")
        return metadata
    print("Cutting the same root 5 square from the image and updated mask...", flush=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".root5-patch-", dir=output_dir.parent))
    try:
        image_name = PATCH_ID + "_image.png"
        mask_name = PATCH_ID + "_mask.png"
        preview_name = PATCH_ID + "_mask_preview.png"
        Image.fromarray(rgb).save(temporary / image_name)
        Image.fromarray(labels).save(temporary / mask_name)
        palette = np.zeros((256, 3), dtype=np.uint8)
        for code, colour in COLOURS.items():
            palette[code] = colour
        Image.fromarray(palette[labels]).save(temporary / preview_name)
        for name, expected, mode in ((image_name, rgb, "RGB"), (mask_name, labels, "L")):
            with Image.open(temporary / name) as saved:
                if saved.mode != mode or not np.array_equal(np.asarray(saved), expected):
                    raise ValueError(f"Saving changed the patch pixels: {name}")
        metadata.update(image_file=image_name, mask_file=mask_name,
                        mask_preview_file=preview_name,
                        image_sha256=sha256(temporary / image_name),
                        mask_sha256=sha256(temporary / mask_name))
        (temporary / "patch_details.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (temporary / "READ_ME.txt").write_text(
            "ROOT 5: ONE MATCHING IMAGE AND MASK PAIR\n\n"
            "Both 1024 x 1024 squares come from the same location on your slide.\n"
            "The image contains microscopy pixels. The mask contains answer codes.\n"
            "1 = Periderm; 2 = Endoderm; 3 = Background; ignore 0 and 255.\n"
            "The colour preview is only for viewing. No resizing or automatic labelling.\n"
            "Unannotated space remains unannotated; the whole patch is not labelled.\n"
            "This is root 5, not the earlier root 1 training patch.\n"
            "Earlier training programs are tied to root 1 and need adapting before use.\n"
            "Same slide, not an independent test. No model was trained.\n")
        unchanged(metadata["inputs_sha256"])
        if output_dir.exists():
            raise FileExistsError(f"Output folder appeared during preparation: {output_dir}")
        temporary.rename(output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print("Created ONE matching root 5 pair: 1024 x 1024 pixels each.")
    print(f"Image patch: {output_dir / image_name}")
    print(f"Numeric mask patch: {output_dir / mask_name}")
    print(f"Colour preview: {output_dir / preview_name}")
    print("Original files and earlier results unchanged. No model has been trained.")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-details", type=Path, default=MASK_DETAILS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        prepare_pair(args.mask_details, args.output, args.check_only)
    except (ValueError, OSError, KeyError, TypeError, Image.DecompressionBombError) as error:
        parser.exit(1, f"Stopped: {error}\n")


if __name__ == "__main__":
    main()
