"""Pair the unchanged practice image with the mask from refined annotations.

Learning exercise. Run with ml-env active.
This copies the existing image patch and crops the same location from the new
numeric mask. It does not resize images, generate predictions or train a model.
Unlike lesson 08, this lesson permits deliberate tissue-boundary corrections.
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
PREVIOUS_PAIR = PROJECT / "04_Results/07_Background_Patch/patch_details.json"
MASK_DETAILS = PROJECT / "04_Results/10_Refined_Mask/mask_details.json"
OUTPUT = PROJECT / "04_Results/11_Refined_Patch"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pixel_counts(labels):
    return {str(code): int(np.count_nonzero(labels == code)) for code in CODES}


def load_pair(previous_pair, mask_details):
    """Read and check the matching image/mask without writing any files."""
    previous_pair, mask_details = Path(previous_pair).resolve(), Path(mask_details).resolve()
    inputs = {str(p): sha256(p) for p in (previous_pair, mask_details)}
    old, details = read_json(previous_pair), read_json(mask_details)
    image_path = (previous_pair.parent / old["image_file"]).resolve()
    old_mask_path = (previous_pair.parent / old["mask_file"]).resolve()
    source_image = Path(details["image_file"]).resolve()
    annotation = Path(details["annotation_file"]).resolve()
    full_mask = (mask_details.parent / details["mask_file"]).resolve()
    trial_path = Path(old["trial_record"]).resolve()
    old_details_path = Path(old["mask_details"]).resolve()
    for path in (image_path, old_mask_path, source_image, annotation,
                 full_mask, trial_path, old_details_path):
        inputs[str(path)] = sha256(path)
    old_inputs = old["inputs_sha256"]
    expected = [(image_path, old["image_sha256"]),
                (old_mask_path, old["mask_sha256"]),
                (source_image, details["image_sha256"]),
                (annotation, details["annotation_sha256"]),
                (trial_path, old_inputs[str(trial_path)]),
                (old_details_path, old_inputs[str(old_details_path)])]
    for path, digest in expected:
        if inputs[str(path)] != digest:
            raise ValueError(f"An input has changed since its saved record: {path}")
    old_details = read_json(old_details_path)
    if old_details["image_sha256"] != details["image_sha256"]:
        raise ValueError("The previous pair and new mask refer to different microscopy images.")
    trial = read_json(trial_path)["provenance"]
    box = old["source_crop_box"]
    if (not isinstance(box, list) or len(box) != 4
            or any(type(v) is not int for v in box)
            or box != trial["source_crop_box"]
            or old["patch_id"] != trial["patch_id"]
            or old["image_sha256"] != trial["image_sha256"]):
        raise ValueError("The pair does not match the earlier training image and location.")
    if (old["trainable_class_codes"] != [1, 2, 3] or old["ignore_codes"] != [0, 255]
            or details["trainable_class_codes"] != [1, 2, 3]
            or details["exclude_from_future_training"] != [0, 255]):
        raise ValueError("Expected three tissue/background classes and ignored codes 0/255.")
    left, top, right, bottom = box
    size = (right - left, bottom - top)
    if list(size) != old["size"]:
        raise ValueError("The crop dimensions do not match the saved pair.")
    previous_limit = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = 200_000_000
        with Image.open(full_mask) as mask, Image.open(source_image) as source:
            width, height = mask.size
            if (mask.mode != "L" or source.mode != "RGB" or source.size != mask.size
                    or [width, height] != [details["width"], details["height"]]
                    or width * height > 200_000_000
                    or not (0 <= left < right <= width and 0 <= top < bottom <= height)):
                raise ValueError("The image, numeric mask and crop dimensions must match.")
            hist = mask.histogram()
            if ({str(i): n for i, n in enumerate(hist) if n}
                    != {k: v for k, v in details["pixel_counts"].items() if v}):
                raise ValueError("Full mask counts disagree with its saved record.")
            labels = np.array(mask.crop(box))
            rgb = np.array(source.crop(box))
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
    with Image.open(image_path) as image, Image.open(old_mask_path) as mask:
        if (image.mode != "RGB" or mask.mode not in ("L", "P")
                or image.size != size or mask.size != size
                or not np.array_equal(np.asarray(image), rgb)):
            raise ValueError("The earlier image patch does not equal this exact source crop.")
        old_labels = np.array(mask)
    if (not np.isin(labels, list(CODES)).all()
            or not np.isin(old_labels, list(CODES)).all()
            or pixel_counts(old_labels) != old["pixel_counts"]):
        raise ValueError("Unexpected mask values or changed previous mask counts.")
    counts = pixel_counts(labels)
    if not all(counts[str(code)] for code in (1, 2, 3)):
        raise ValueError("This practice patch needs labelled examples of all three classes.")
    # Boundary refinements may remove or add tissue labels. Copy all new values
    # exactly, including unlabelled (0) and conflicting (255) pixels.
    transitions, numbers = np.unique(
        np.stack((old_labels.ravel(), labels.ravel()), axis=1), axis=0, return_counts=True)
    changes = [{"old_code": int(a), "new_code": int(b), "pixels": int(n)}
               for (a, b), n in zip(transitions, numbers) if a != b]
    metadata = {
        "purpose": "Same practice image with refined human labels; no training occurred",
        "patch_id": old["patch_id"], "source_crop_box": box, "size": list(size),
        "previous_pair_details": str(previous_pair), "trial_record": str(trial_path),
        "mask_details": str(mask_details), "full_mask": str(full_mask),
        "annotation_file": str(annotation), "annotation_sha256": details["annotation_sha256"],
        "inputs_sha256": inputs, "pixel_codes": CODES, "pixel_counts": counts,
        "old_mask_pixel_counts": pixel_counts(old_labels), "label_changes": changes,
        "changed_pixels": int(np.count_nonzero(labels != old_labels)),
        "change_rule": "Exact crop of refined human mask; no automatic Background filling",
        "image_identical_to_previous_training_input": True,
        "trainable_class_codes": [1, 2, 3], "ignore_codes": [0, 255],
        "mask_reading": "Use numeric PNG values, not colour-preview RGB values",
        "model_trained": False,
        "limitation": "Same patch and slide as earlier training; not an independent test",
        "script_file": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
    }
    if any(sha256(path) != digest for path, digest in inputs.items()):
        raise ValueError("An input changed during checking. Run again after saving it.")
    return image_path, labels, metadata


def prepare_pair(previous_pair, mask_details, output_dir, check_only=False):
    output_dir = Path(output_dir).resolve()
    if not check_only and output_dir.exists():
        raise FileExistsError(f"Output folder already exists; nothing overwritten: {output_dir}")
    image_path, labels, metadata = load_pair(previous_pair, mask_details)
    if check_only:
        print("Refined pair check passed. No patch files were written; no model was trained.")
        return metadata
    stem = metadata["patch_id"]
    image_name, mask_name = stem + "_image.png", stem + "_mask.png"
    preview_name = stem + "_mask_preview.png"
    print("Keeping the same image patch and cutting your refined answer key...", flush=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".refined-patch-", dir=output_dir.parent))
    try:
        shutil.copy2(image_path, temporary / image_name)
        Image.fromarray(labels).save(temporary / mask_name)
        palette = np.zeros((256, 3), dtype=np.uint8)
        for code, colour in COLOURS.items():
            palette[code] = colour
        Image.fromarray(palette[labels]).save(temporary / preview_name)
        with Image.open(temporary / mask_name) as saved:
            if saved.mode != "L" or not np.array_equal(np.asarray(saved), labels):
                raise ValueError("Saving changed the numeric mask values.")
        if sha256(temporary / image_name) != metadata["inputs_sha256"][str(image_path)]:
            raise ValueError("Copying changed the microscopy image.")
        metadata.update(image_file=image_name, mask_file=mask_name, mask_preview_file=preview_name,
                        image_sha256=sha256(temporary / image_name),
                        mask_sha256=sha256(temporary / mask_name))
        (temporary / "patch_details.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (temporary / "READ_ME.txt").write_text(
            "SAME IMAGE, REFINED ANSWER KEY\n\n"
            "The image patch is identical to the one used for earlier training.\n"
            "The numeric mask is the exact matching crop from your refined annotations.\n"
            "No resizing, stretching or automatic labelling of unannotated space.\n"
            "1 = Periderm; 2 = Endoderm; 3 = Background. Ignore 0 and 255.\n"
            "The colour preview is for viewing; train with the numeric mask instead.\n"
            "patch_details.json records the location, inputs and label changes.\n"
            "No training occurred. Earlier files and models are unchanged.\n"
            "This is the same training patch, not an independent test image.\n")
        if any(sha256(path) != digest for path, digest in metadata["inputs_sha256"].items()):
            raise ValueError("An input changed during preparation; no result was published.")
        if output_dir.exists():
            raise FileExistsError(f"Output folder appeared during preparation: {output_dir}")
        temporary.rename(output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"Created ONE matching pair: {labels.shape[1]} x {labels.shape[0]} pixels each.")
    print("Image pixels are identical to the earlier training image.")
    print("The mask now follows your refined outlines and added Background rectangle.")
    print(f"Image patch: {output_dir / image_name}")
    print(f"Numeric mask patch: {output_dir / mask_name}")
    print(f"Colour preview: {output_dir / preview_name}")
    for code, name in CODES.items():
        print(f"  {code}: {name} = {metadata['pixel_counts'][str(code)]:,} pixels")
    print(f"Changed mask pixels compared with the earlier three-category patch: {metadata['changed_pixels']:,}")
    print("Earlier files and models unchanged. No model has been trained.")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-pair", type=Path, default=PREVIOUS_PAIR)
    parser.add_argument("--mask-details", type=Path, default=MASK_DETAILS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check-only", action="store_true", help="Read/check inputs without writing a pair.")
    args = parser.parse_args()
    try:
        prepare_pair(args.previous_pair, args.mask_details, args.output, args.check_only)
    except (ValueError, KeyError, TypeError, OSError, Image.DecompressionBombError) as error:
        parser.exit(1, f"Stopped: {error}\n")


if __name__ == "__main__":
    main()
