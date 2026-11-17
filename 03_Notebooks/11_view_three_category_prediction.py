"""View a saved three-category model on the SAME patch used for training.

Learning exercise. This is inference: using learned model
numbers without changing them. Only the RGB image enters the model. The human
mask is used separately for comparison and choosing a display close-up.

Predictions are shown at EVERY pixel without masking or replacing them using
the answer key. Grey in the reference means unlabelled, not Background.
Colours: red Periderm, green Endoderm, yellow Background; reference-only grey
unlabelled and pink conflict. Only reference codes 1/2/3 are scored.
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FIRST_RUN = ROOT / "04_Results/08_Three_Class_Training/run_20261113T155717Z_elwd4ak7"
LABELS = ("Periderm", "Endoderm", "Background")
CLASS_CODES = (1, 2, 3)
PALETTE = np.zeros((256, 3), dtype=np.uint8)
PALETTE[0] = (48, 52, 58)
PALETTE[1] = (218, 54, 65)
PALETTE[2] = (30, 170, 91)
PALETTE[3] = (255, 212, 59)
PALETTE[255] = (209, 77, 199)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_saved_example(run_dir, device):
    record_file = run_dir / "trial_details.json"
    weights_file = run_dir / "model_weights.pt"
    record = json.loads(record_file.read_text())
    if record["output_channels"] != list(LABELS) or record["output_mask_codes"] != list(CLASS_CODES):
        raise ValueError("Expected the three-category teaching model.")
    if sha256(weights_file) != record["model_weights_sha256"]:
        raise ValueError("Saved weights differ from the training record.")
    model_file = run_dir / "programs/09_three_class_unet.py"
    base_file = run_dir / "programs/04_small_unet.py"
    if (sha256(model_file) != record["model_script_sha256"]
            or sha256(base_file) != record["base_model_script_sha256"]):
        raise ValueError("Saved model definitions differ from the training record.")
    spec = importlib.util.spec_from_file_location("saved_three_class_model", model_file)
    unet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unet)
    paths = {k: Path(record["provenance"][k]) for k in ("image", "mask")}
    for name, path in paths.items():
        if sha256(path) != record["provenance"][name + "_sha256"]:
            raise ValueError(f"The saved {name} patch has changed since training.")
    with Image.open(paths["image"]) as im:
        if im.mode != "RGB" or im.size != (1024, 1024):
            raise ValueError("Expected the original 1024-square RGB patch.")
        rgb = np.array(im)
    with Image.open(paths["mask"]) as im:
        if im.mode not in ("P", "L") or im.size != (1024, 1024):
            raise ValueError("Expected the matching numeric mask, not a preview.")
        truth = np.array(im)
    if not np.isin(truth, (0, 1, 2, 3, 255)).all() or not np.isin(truth, (1, 2)).any():
        raise ValueError("Invalid mask codes or no labelled tissue.")
    model = unet.ThreeClassUNet()
    weights = torch.load(weights_file, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.requires_grad_(False)
    model.to(device).eval()
    return model, rgb, truth, record, {
        "run_folder": str(run_dir), "weights_sha256": sha256(weights_file),
        "trial_details_sha256": sha256(record_file),
        "image_sha256": sha256(paths["image"]), "mask_sha256": sha256(paths["mask"]),
        "model_definition": str(model_file), "model_script_sha256": sha256(model_file),
        "base_model_script_sha256": sha256(base_file),
    }


def predict(model, rgb, device):
    """Image only: no reference labels are supplied to this function."""
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.inference_mode():
        scores = model(x.to(device))
        if tuple(scores.shape) != (1, 3, rgb.shape[0], rgb.shape[1]):
            raise RuntimeError("Expected three score maps with the image dimensions.")
        if not bool(torch.isfinite(scores).all().item()):
            raise RuntimeError("Non-finite prediction scores.")
        # Output positions 0/1/2 correspond to saved mask codes 1/2/3.
        labels = (scores.argmax(1)[0] + 1).cpu().numpy().astype(np.uint8)
    return labels


def compare_labels(truth, prediction):
    if (truth.shape != prediction.shape or not np.isin(prediction, CLASS_CODES).all()
            or not np.isin(truth, (0, 1, 2, 3, 255)).all()):
        raise ValueError("Expected matching numeric masks with valid codes.")
    valid = np.isin(truth, CLASS_CODES)
    if not valid.any():
        raise ValueError("No labelled pixels to compare.")
    confusion = [[int(((truth == a) & (prediction == b)).sum()) for b in CLASS_CODES] for a in CLASS_CODES]
    return {
        "labelled_pixels": int(valid.sum()),
        "agreement_on_training_pair": float((truth[valid] == prediction[valid]).mean()),
        "class_order": list(LABELS),
        "confusion_rows_truth_columns_prediction": confusion,
        "predicted_counts_on_labelled_area": {str(k): int(((prediction == k) & valid).sum()) for k in CLASS_CODES},
        "predicted_counts_on_unlabelled_area": {str(k): int(((prediction == k) & (truth == 0)).sum()) for k in CLASS_CODES},
        "predicted_counts_on_whole_patch": {str(k): int((prediction == k).sum()) for k in CLASS_CODES},
        "scope": "Only labelled training-patch pixels are scored; unlabelled predictions have no reference answers.",
    }


def closeup_box(truth):
    """Pick a display window only, after prediction; never change model input."""
    yy, xx = np.nonzero(np.isin(truth, (1, 2)))
    height, width = truth.shape
    w, h = min(512, width), min(256, height)
    left = int(np.clip((int(xx.min()) + int(xx.max())) // 2 - w // 2, 0, width - w))
    top = int(np.clip((int(yy.min()) + int(yy.max())) // 2 - h // 2, 0, height - h))
    return [left, top, left + w, top + h]


def comparison_picture(rgb, truth, prediction, training_updates):
    box = closeup_box(truth)
    panels = [Image.fromarray(rgb), Image.fromarray(PALETTE[truth]), Image.fromarray(PALETTE[prediction])]
    width, margin, gap = 512, 28, 24
    canvas = Image.new("RGB", (3 * width + 2 * margin + 2 * gap, 1180), "white")
    draw = ImageDraw.Draw(canvas)
    font = lambda size: ImageFont.load_default(size=size)
    draw.text((margin, 22), "Three-category prediction: the SAME training patch", font=font(30), fill="#17212B")
    draw.text((margin, 68), f"Saved model after {training_updates} updates | No further training | This is not a test on a new image", font=font(21), fill="#374151")
    names = ["1  Microscopy image", "2  Your ground-truth labels", "3  Model prediction at EVERY pixel"]
    for j, (name, panel) in enumerate(zip(names, panels)):
        x = margin + j * (width + gap)
        draw.text((x, 120), name, font=font(22), fill="#17212B")
        canvas.paste(panel.resize((width, width), Image.Resampling.NEAREST), (x, 160))
        draw.rectangle((x, 160, x + width - 1, 671), outline="#CDD3DA", width=1)
        scaled = [x + box[0] // 2, 160 + box[1] // 2, x + box[2] // 2, 160 + box[3] // 2]
        draw.rectangle(scaled, outline="#00BFE8", width=2)
        crop = panel.crop(box)
        canvas.paste(crop, (x, 728))
        draw.rectangle((x, 728, x + crop.width - 1, 728 + crop.height - 1), outline="#CDD3DA", width=1)
    draw.text((margin, 693), "Closer view below: the same cyan-box area in all three panels, at original pixel scale", font=font(21), fill="#374151")
    draw.text((margin, 1008), "Red = Periderm   Green = Endoderm   Yellow = Background", font=font(21), fill="#17212B")
    draw.text((margin, 1044), "Grey = unlabelled in your mask   Pink = conflicting labels (ignored)", font=font(21), fill="#17212B")
    draw.text((margin, 1080), "Predictions are shown everywhere. Grey areas are NOT copied from your mask.", font=font(21), fill="#17212B")
    draw.text((margin, 1116), "Only pixels labelled in your mask are scored. This is not a test on a new image.", font=font(21), fill="#17212B")
    return canvas, box


def save_view(prediction, picture, details, parent):
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(tempfile.mkdtemp(prefix=f"view_{stamp}_", dir=parent))
    Image.fromarray(prediction).save(output / "predicted_mask.png")
    Image.fromarray(PALETTE[prediction]).save(output / "predicted_mask_preview.png")
    picture.save(output / "image_truth_prediction.png")
    (output / "prediction_details.json").write_text(json.dumps(details, indent=2, allow_nan=False) + "\n")
    (output / "READ_ME.txt").write_text(
        "View from the saved three-category model; no further training.\n"
        "image_truth_prediction.png compares the image, human reference and every-pixel prediction.\n"
        "predicted_mask.png is a 1024 x 1024 numeric mask (1 Periderm, 2 Endoderm, 3 Background).\n"
        "predicted_mask_preview.png shows those same predictions in red, green and yellow.\n"
        "The model received only the microscopy image. Human labels were not used to replace predictions.\n"
        "Same training patch; this does not test performance on a new image.\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=FIRST_RUN,
                        help="Saved training-run folder. Default: the first three-category 200-update run.")
    parser.add_argument("--no-open", action="store_true", help="Save without opening Preview.")
    parser.add_argument("--output-parent", type=Path, default=ROOT / "04_Results/09_Three_Class_Prediction")
    args = parser.parse_args()
    run_dir = args.run.expanduser().resolve()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading saved model from: {run_dir.name}", flush=True)
    model, rgb, truth, record, sources = load_saved_example(run_dir, device)
    print(f"Saved model after {record['completed_updates']} updates.", flush=True)
    print("Predicting from the image. The mask is used only for comparison, not as model input.", flush=True)
    prediction = predict(model, rgb, device)
    metrics = compare_labels(truth, prediction)
    matches_record = metrics["confusion_rows_truth_columns_prediction"] == record["after_training"]["confusion_rows_reference_columns_prediction"]
    picture, box = comparison_picture(rgb, truth, prediction, record["completed_updates"])
    details = {
        "purpose": "Visual comparison on the SAME training pair; not an independent test.",
        "created_utc": datetime.now(timezone.utc).isoformat(), "device": str(device),
        "sources": sources, "source_patch_id": record["provenance"]["patch_id"],
        "script_sha256": sha256(__file__), "model_script_sha256": sources["model_script_sha256"],
        "training_updates": record["completed_updates"], "further_training_updates": 0,
        "metrics": metrics, "labelled_predictions_match_training_record": matches_record,
        "prediction_codes": {"1": "Periderm", "2": "Endoderm", "3": "Background"},
        "prediction_pixels_hidden_or_replaced_using_truth": False,
        "preview": {"full_panel_scale": 0.5, "closeup_box_in_patch": box, "closeup_scale": 1.0,
                    "selection_note": "Ground truth selects the display crop only AFTER full-image prediction."},
        "limitations": ["Model assigns a category to every pixel, including unlabelled areas without reference answers.",
                        "Only reference codes 1, 2 and 3 are scored; 0 and 255 are excluded.",
                        "The visual comparison uses the training example and cannot assess generalization."],
    }
    if sha256(run_dir / "model_weights.pt") != sources["weights_sha256"]:
        raise RuntimeError("Saved weights changed while making this view.")
    if not matches_record:
        print("Note: labelled-pixel predictions differ from the saved training summary; inspect the details.")
    output = save_view(prediction, picture, details, args.output_parent)
    print(f"\nComparison saved:\n{output / 'image_truth_prediction.png'}")
    print("Left: microscopy image. Middle: your answer key. Right: the model's guesses.")
    print("Red = Periderm. Green = Endoderm. Yellow = Background.")
    print("No further training occurred. Saved model, annotations and source patches are unchanged.")
    if not args.no_open and sys.platform == "darwin":
        result = subprocess.run(["open", "-a", "Preview", str(output / "image_truth_prediction.png")], check=False)
        if result.returncode:
            print("Preview could not open automatically; open the saved comparison from Finder.")
    print("Stop here and look at the comparison together.")


if __name__ == "__main__":
    main()
