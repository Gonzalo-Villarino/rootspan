"""Use the fixed root-5 model to predict root 4, without further training.

Learning exercise. Only RGB image pixels enter
the model. The root-4 reference mask is used separately to assess predictions.
This is another root from the SAME slide, not independent new-slide evaluation.
--check-only validates inputs and loads saved weights without making predictions
or writing files. The earlier same-training-patch viewer remains unchanged.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


PROJECT = Path(__file__).resolve().parents[1]
PAIR = PROJECT / "04_Results/31_Root4_Background_Check_Patch/patch_details.json"
OUTPUT = PROJECT / "04_Results/32_Root4_Prediction"
HELPER_FILE = PROJECT / "03_Notebooks/11_view_three_category_prediction.py"
LABELS = ["Periderm", "Endoderm", "Background"]
CODES = [1, 2, 3]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_inputs(pair_path, device):
    pair_path = pair_path.expanduser().resolve()
    pair = json.loads(pair_path.read_text())
    run = Path(pair["selected_model_run"]).resolve()
    record_path = run / "trial_details.json"
    record = json.loads(record_path.read_text())
    weights_path = run / "model_weights.pt"
    model_file = run / "programs/09_three_class_unet.py"
    base_file = run / "programs/04_small_unet.py"
    image_path = pair_path.parent / pair["image_file"]
    mask_path = pair_path.parent / pair["mask_file"]
    inputs = [pair_path, record_path, weights_path, model_file, base_file,
              image_path, mask_path, HELPER_FILE, Path(__file__).resolve()]
    hashes = {str(path): sha256(path) for path in inputs}
    checks = [(weights_path, record["model_weights_sha256"]),
              (weights_path, pair["selected_model_weights_sha256"]),
              (model_file, record["model_script_sha256"]),
              (base_file, record["base_model_script_sha256"]),
              (image_path, pair["image_sha256"]), (mask_path, pair["mask_sha256"])]
    if any(hashes[str(path)] != digest for path, digest in checks):
        raise ValueError("A saved input differs from its recorded fingerprint.")
    if (record["output_channels"] != LABELS or record["output_mask_codes"] != CODES
            or record["image_preprocessing"] != "RGB / 255; no resizing or augmentation"):
        raise ValueError("Unexpected model categories or image preprocessing.")
    provenance = record["provenance"]
    source_image = str(Path(pair["source_image"]).resolve())
    if (source_image != str(Path(provenance["source_image"]).resolve())
            or pair["inputs_sha256"][source_image]
            != provenance["source_inputs_sha256"][source_image]):
        raise ValueError("Expected crops from the same unchanged microscopy slide.")
    box, training_box = pair["source_crop_box"], provenance["source_crop_box"]
    overlap = (max(0, min(box[2], training_box[2]) - max(box[0], training_box[0]))
               * max(0, min(box[3], training_box[3]) - max(box[1], training_box[1])))
    if box != [8928, 5264, 9952, 6288] or overlap:
        raise ValueError("Expected the prepared root-4 crop with no training-crop overlap.")
    with Image.open(image_path) as image:
        if image.mode != "RGB" or image.size != (1024, 1024):
            raise ValueError("Expected the unmodified 1024-square RGB image patch.")
        rgb = np.array(image)
    with Image.open(mask_path) as mask:
        if mask.mode != "L" or mask.size != (1024, 1024):
            raise ValueError("Expected the numeric reference mask, not its colour preview.")
        truth = np.array(mask)
    counts = {str(c): int((truth == c).sum()) for c in [0, 1, 2, 3, 255]}
    if (counts != pair["pixel_counts"] or sum(counts.values()) != truth.size
            or not all(counts[str(c)] for c in CODES)):
        raise ValueError("Reference mask counts disagree or a reference class is missing.")
    model_module = load_module("saved_root5_for_root4", model_file)
    model = model_module.ThreeClassUNet()
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not all(bool(torch.isfinite(value).all()) for value in weights.values()):
        raise ValueError("Saved model weights contain non-finite values.")
    model.load_state_dict(weights, strict=True)
    model.requires_grad_(False)
    model.to(device).eval()
    return model, rgb, truth, pair, record, hashes


def unchanged(hashes):
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise ValueError(f"An input changed during this operation: {path}")


def comparison_picture(helper, rgb, truth, prediction, updates):
    # Reference labels choose ONLY the display close-up, after full prediction.
    box = helper.closeup_box(truth)
    panels = [Image.fromarray(rgb), Image.fromarray(helper.PALETTE[truth]),
              Image.fromarray(helper.PALETTE[prediction])]
    width, margin, gap = 512, 28, 24
    canvas = Image.new("RGB", (3 * width + 2 * margin + 2 * gap, 1180), "white")
    draw = ImageDraw.Draw(canvas)
    font = lambda size: ImageFont.load_default(size=size)
    draw.text((margin, 22), "Root 4: prediction from the model trained on root 5",
              font=font(30), fill="#17212B")
    draw.text((margin, 68), f"Saved root 5 model after {updates} updates | No further training | Same slide, different root",
              font=font(21), fill="#374151")
    names = ["1  Root 4 microscopy image", "2  Your reference labels", "3  Model prediction at EVERY pixel"]
    for i, (name, panel) in enumerate(zip(names, panels)):
        x = margin + i * (width + gap)
        draw.text((x, 120), name, font=font(22), fill="#17212B")
        canvas.paste(panel.resize((width, width), Image.Resampling.NEAREST), (x, 160))
        draw.rectangle((x, 160, x + width - 1, 671), outline="#CDD3DA")
        draw.rectangle((x + box[0] // 2, 160 + box[1] // 2,
                        x + box[2] // 2, 160 + box[3] // 2), outline="#00BFE8", width=2)
        crop = panel.crop(box)
        canvas.paste(crop, (x, 728))
        draw.rectangle((x, 728, x + crop.width - 1, 728 + crop.height - 1), outline="#CDD3DA")
    lines = [(693, "Closer view below: the same cyan-box area in all three panels, at original pixel scale"),
             (1008, "Red = Periderm   Green = Endoderm   Yellow = Background"),
             (1044, "Grey = unlabelled in your mask   Pink = conflicting labels (ignored)"),
             (1080, "The model received ONLY the image. Your mask is used separately to check its prediction."),
             (1116, "Only labelled pixels are scored. This is a preliminary check on another root from the same slide.")]
    for y, line in lines:
        draw.text((margin, y), line, font=font(21), fill="#374151")
    return canvas, box


def save_result(prediction, picture, details, helper, parent, hashes):
    parent = parent.expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".root4-building-", dir=parent))
    output = parent / ("view_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                       + "_" + temporary.name.removeprefix(".root4-building-"))
    try:
        Image.fromarray(prediction).save(temporary / "predicted_mask.png")
        Image.fromarray(helper.PALETTE[prediction]).save(temporary / "predicted_mask_preview.png")
        picture.save(temporary / "image_truth_prediction.png")
        with Image.open(temporary / "predicted_mask.png") as saved:
            if saved.mode != "L" or not np.array_equal(np.array(saved), prediction):
                raise ValueError("Saved prediction pixels differ from the model output.")
        details["prediction_sha256"] = sha256(temporary / "predicted_mask.png")
        (temporary / "prediction_details.json").write_text(json.dumps(details, indent=2, allow_nan=False) + "\n")
        programs = temporary / "programs"
        programs.mkdir()
        for path in [Path(__file__).resolve(), HELPER_FILE]:
            shutil.copy2(path, programs / path.name)
        (temporary / "READ_ME.txt").write_text(
            "ROOT 4: PREDICTION USING THE FIXED ROOT 5 MODEL\n\n"
            "The model received only the 1024 x 1024 microscopy image (RGB / 255).\n"
            "It did not receive your reference mask or learn from root 4.\n"
            "Your mask is used separately to check predictions on labelled pixels.\n"
            "The model assigns Periderm, Endoderm or Background at EVERY pixel.\n"
            "No predictions were hidden, replaced or cleaned using your answer key.\n"
            "Mask codes 0 (unlabelled) and 255 (conflict) are excluded from scores.\n"
            "Per-class percentages describe how many labelled pixels of that class matched.\n"
            "They do not assess unlabelled space or provide whole-image accuracy.\n"
            "This crop does not overlap the root 5 training crop, but both share one slide.\n"
            "This is a preliminary check, not independent new-slide evaluation.\n"
            "No further training or changes to the saved model occurred.\n")
        unchanged(hashes)
        if output.exists():
            raise FileExistsError("Output folder appeared; nothing overwritten.")
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-details", type=Path, default=PAIR)
    parser.add_argument("--output-parent", type=Path, default=OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-open", action="store_true", help="Save without opening Preview.")
    args = parser.parse_args()
    try:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print("Checking the root 4 pair and the saved root 5 model...", flush=True)
        model, rgb, truth, pair, record, hashes = load_inputs(args.patch_details, device)
        helper = load_module("root4_prediction_helpers", HELPER_FILE)
        unchanged(hashes)
        print("Verified: another root, with no overlap with the root 5 training crop.")
        print(f"Saved model: {record['completed_updates']} previous training updates. New training updates: 0.")
        if args.check_only:
            print("Input and saved-model check passed. No prediction, training or file writing occurred.")
            return
        print("Predicting from the root 4 IMAGE ONLY. Then comparing with your separate mask...", flush=True)
        prediction = helper.predict(model, rgb, device)
        metrics = helper.compare_labels(truth, prediction)
        # Replace the old viewer's training-specific names: these are root 4 results.
        metrics["agreement_on_labelled_pixels"] = metrics.pop("agreement_on_training_pair")
        metrics["scope"] = "Only labelled root-4 reference pixels; same-slide check, not training-patch accuracy."
        matrix = metrics["confusion_rows_truth_columns_prediction"]
        metrics["per_class"] = {}
        print("Match to your labelled reference pixels on root 4:")
        for i, label in enumerate(LABELS):
            total, correct = sum(matrix[i]), matrix[i][i]
            fraction = correct / total if total else None
            metrics["per_class"][label] = {"reference_pixels": total, "matched_pixels": correct,
                                          "fraction_of_reference_pixels_matched": fraction}
            result = f"{100 * fraction:.2f}%" if fraction is not None else "not assessed"
            print(f"  {label}: {correct:,} / {total:,} matched ({result}).")
        print("These percentages exclude unlabelled space. Both roots are from the same slide.")
        picture, box = comparison_picture(helper, rgb, truth, prediction, record["completed_updates"])
        details = {
            "purpose": "Root 4 prediction with the fixed root 5 model; preliminary same-slide check",
            "created_utc": datetime.now(timezone.utc).isoformat(), "device": str(device),
            "python": sys.version.split()[0], "pytorch": torch.__version__,
            "source_patch_id": pair["patch_id"], "source_crop_box": pair["source_crop_box"],
            "patch_details": str(args.patch_details.expanduser().resolve()),
            "model_run": pair["selected_model_run"], "inputs_sha256": hashes,
            "training_updates": record["completed_updates"], "further_training_updates": 0,
            "model_input": "RGB image only, divided by 255; no resizing or augmentation",
            "reference_mask_used_as_model_input": False,
            "prediction_pixels_hidden_or_replaced_using_truth": False,
            "metrics": metrics, "prediction_codes": dict(zip(map(str, CODES), LABELS)),
            "preview": {"full_panel_scale": 0.5, "closeup_scale": 1.0,
                        "closeup_box_in_patch": box,
                        "selection_note": "Reference labels choose only the display crop, after prediction."},
            "limitations": ["One different root on the same slide; not an independent new-slide test.",
                            "Unlabelled/conflicting reference pixels are excluded from metrics.",
                            "Per-class match fractions are not whole-image accuracy or proof of biological correctness."],
        }
        output = save_result(prediction, picture, details, helper, args.output_parent, hashes)
        print(f"\nComparison saved:\n{output / 'image_truth_prediction.png'}")
        print("Left: image. Middle: your answer key. Right: the model's prediction.")
        print("Saved model, annotations and source patches unchanged. No further training occurred.")
        if not args.no_open and sys.platform == "darwin":
            result = subprocess.run(["open", "-a", "Preview", str(output / "image_truth_prediction.png")], check=False)
            if result.returncode:
                print("Open the saved comparison from Finder; Preview could not open automatically.")
        print("Stop here so we can inspect the prediction together.")
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"Stopped: {error}\n")


if __name__ == "__main__":
    main()
