"""Prepare one root-4 image and reference mask for a later prediction check.

Learning exercise. This program reuses program
07 to fill the current saved outlines, then crops image and mask identically.
It does not load model weights, predict, or train. --check-only writes nothing.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
ANNOTATIONS = PROJECT / "02_Data/02_My_First_Annotations/MT25_rep3_root5_refined_GV.json"
RUN = PROJECT / "04_Results/27_Root5_Lower_Background_Training/run_20261115T121005Z_d_v61wde"
OUTPUT = PROJECT / "04_Results/30_Root4_Check_Patch"
# Root 4 transition observed in Labelme and saved outlines, around x9439/y5780.
# Zero-based left/top included, right/bottom excluded. No resizing or padding.
BOX = [8928, 5264, 9952, 6288]
PATCH_ID = "MT25_rep3_root4_x8928_y5264"
MASK_PROGRAM = PROJECT / "03_Notebooks/07_make_background_mask.py"


def load_mask_program():
    spec = importlib.util.spec_from_file_location("background_mask", MASK_PROGRAM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(check_only=False):
    if OUTPUT.exists() and not check_only:
        raise FileExistsError(f"Folder already exists; nothing overwritten: {OUTPUT}")
    mask_program = load_mask_program()
    sha = mask_program.sha256
    record_path = RUN / "trial_details.json"
    input_paths = [ANNOTATIONS, record_path, MASK_PROGRAM, Path(__file__).resolve()]
    hashes = {str(path): sha(path) for path in input_paths}
    data = json.loads(ANNOTATIONS.read_text())
    width, height = mask_program.validate(data)
    data.pop("imageData", None)
    image_path = (ANNOTATIONS.parent / data["imagePath"]).resolve()
    hashes[str(image_path)] = sha(image_path)
    training = json.loads(record_path.read_text())
    provenance = training["provenance"]
    if (image_path != Path(provenance["source_image"]).resolve()
            or hashes[str(image_path)] != provenance["source_inputs_sha256"][str(image_path)]):
        raise ValueError("Expected the unchanged microscopy slide used for root 5.")
    train_box = provenance["source_crop_box"]
    left, top, right, bottom = BOX
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Root 4 crop is outside the image.")
    overlap = (max(0, min(right, train_box[2]) - max(left, train_box[0]))
               * max(0, min(bottom, train_box[3]) - max(top, train_box[1])))
    if overlap:
        raise ValueError("The root 4 crop overlaps the root 5 training crop.")
    previous_limit = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = 200_000_000
        with Image.open(image_path) as original:
            if original.mode != "RGB" or original.size != (width, height):
                raise ValueError("Microscopy image and annotation dimensions disagree.")
            rgb = np.array(original.crop(BOX))
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
    # Rasterize at ORIGINAL coordinates to retain exactly program 07's pixel
    # boundary rules. Only the crop is saved; no full-slide mask file is needed.
    full_mask = mask_program.make_mask(data)
    labels = full_mask[top:bottom, left:right].copy()
    del full_mask
    counts = {str(code): int(np.count_nonzero(labels == code))
              for code in mask_program.CODE_NAMES}
    if (rgb.shape != (1024, 1024, 3) or labels.shape != (1024, 1024)
            or sum(counts.values()) != labels.size
            or not all(counts[str(code)] for code in (1, 2))):
        raise ValueError("Expected a matching 1024-square pair with both tissues.")

    def unchanged():
        for name, digest in hashes.items():
            if sha(Path(name)) != digest:
                raise ValueError(f"An input changed during preparation: {name}")

    unchanged()
    print("Root 4: one 1024 x 1024 image and its matching reference mask.")
    print("Verified: no overlap with the root 5 training patch.")
    for code, name in mask_program.CODE_NAMES.items():
        print(f"  {code}: {name} = {counts[str(code)]:,} pixels")
    if not counts["3"]:
        print("No Background reference labels in this patch; empty space stays unlabelled.")
    if check_only:
        print("Preparation check passed. No patch files written. No prediction or training.")
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".root4-check-", dir=OUTPUT.parent))
    try:
        image_name, mask_name = PATCH_ID + "_image.png", PATCH_ID + "_mask.png"
        preview_name = PATCH_ID + "_mask_preview.png"
        Image.fromarray(rgb).save(temporary / image_name)
        Image.fromarray(labels).save(temporary / mask_name)
        palette = np.zeros((256, 3), dtype=np.uint8)
        for code, colour in mask_program.PREVIEW_COLOURS.items():
            palette[code] = colour
        Image.fromarray(palette[labels]).save(temporary / preview_name)
        for name, expected, mode in ((image_name, rgb, "RGB"), (mask_name, labels, "L")):
            with Image.open(temporary / name) as saved:
                if saved.mode != mode or not np.array_equal(np.asarray(saved), expected):
                    raise ValueError(f"Saving changed the pixels: {name}")
        # Preserve the exact reference outlines without duplicating embedded RGB.
        (temporary / "reference_annotations_without_image_data.json").write_text(
            json.dumps(data, indent=2) + "\n")
        programs = temporary / "programs"
        programs.mkdir()
        for path in (Path(__file__).resolve(), MASK_PROGRAM):
            shutil.copy2(path, programs / path.name)
        metadata = {
            "purpose": "Root 4 same-slide check pair; not used to train the selected model",
            "patch_id": PATCH_ID, "source_crop_box": BOX, "size": [1024, 1024],
            "crop_convention": "Zero-based left/top included; right/bottom excluded",
            "source_image": str(image_path), "annotation_file": str(ANNOTATIONS),
            "annotation_sha256": hashes[str(ANNOTATIONS)], "inputs_sha256": hashes,
            "image_file": image_name, "mask_file": mask_name,
            "mask_preview_file": preview_name,
            "image_sha256": sha(temporary / image_name),
            "mask_sha256": sha(temporary / mask_name),
            "pixel_codes": mask_program.CODE_NAMES, "pixel_counts": counts,
            "ignore_codes": [0, 255],
            "labelled_percent": 100 * sum(counts[str(c)] for c in (1, 2, 3)) / labels.size,
            "selected_model_run": str(RUN),
            "selected_model_weights_sha256": training["model_weights_sha256"],
            "training_crop_box": train_box, "overlap_with_training_crop_pixels": overlap,
            "reference_mask_method": "Program 07 full-coordinate rasterization, then exact crop",
            "mask_role": "Comparison reference only; not a model input for prediction",
            "limitations": ["Same slide as training, not independent new-slide evaluation.",
                            "Only explicitly labelled pixels can be scored.",
                            "No automatic Background labels; no claim that every boundary is certain."],
            "model_trained": False, "prediction_performed": False,
        }
        (temporary / "patch_details.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (temporary / "READ_ME.txt").write_text(
            "ROOT 4: PREPARING A NEW EXAMPLE\n\n"
            "The microscopy image will be the model input during prediction.\n"
            "Your mask is kept separately to check the prediction afterwards.\n"
            "The model does not receive the mask when making its prediction.\n"
            "This program fills your current saved outlines using program 07,\n"
            "then cuts the same 1024 x 1024 square from image and mask.\n"
            "1=Periderm, 2=Endoderm, 3=Background, 0=unlabelled, 255=conflict.\n"
            "Grey means unlabelled, not Background. Ignore 0 and 255 when scoring.\n"
            "The colour preview is for viewing; the numeric mask holds the codes.\n"
            "This crop does not overlap the selected model's root 5 training crop.\n"
            "Both crops are from one slide: this is a preliminary check only.\n"
            "No prediction or training has occurred. Original files unchanged.\n")
        unchanged()
        if OUTPUT.exists():
            raise FileExistsError("Output folder appeared during preparation; stopped.")
        temporary.rename(OUTPUT)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"Image patch: {OUTPUT / image_name}")
    print(f"Reference mask: {OUTPUT / mask_name}")
    print(f"Colour preview: {OUTPUT / preview_name}")
    print("Original files and saved model unchanged. No prediction or training occurred.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        prepare(args.check_only)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Stopped: {error}\n")


if __name__ == "__main__":
    main()
