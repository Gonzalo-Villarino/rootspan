"""Apply the frozen PAT transfer model to one preselected new-image crop.

Image-only inference: no reference masks, annotations, optimizer, training,
or continuity cleanup. The crop was chosen for visible cells before prediction.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import segmentation_models_pytorch as smp
import torch


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "02_Data/05_Training_and_Evaluation/02_Evaluation_Images"
IMAGE = EVALUATION / "02_Images_For_Labelme/20_29_evaluation.png"
RUN = ROOT / "04_Results/39_PAT_Transfer_Learning/run_20261115T140159Z"
PARENT = ROOT / "04_Results/41_New_Image_20_29_Prediction"
IMAGE_SHA = "df2ccc184dcbd52cdc35b8c6b8b92f32a863e678ca2a24d25aa2ba41ec11abd0"
WEIGHTS_SHA = "234b7de1969d80dd1523ce52afc89d5952b5e9ba4a63cdd7b3c8bff5b1c13633"
BOX = (7988, 5618, 9012, 6642)  # x0, y0, x1, y1; right/bottom exclusive.
LABELS = {1: "Periderm", 2: "Endoderm", 3: "Background"}
PALETTE = np.array([[48, 52, 59], [219, 48, 63], [26, 166, 94], [255, 220, 58]], dtype=np.uint8)
Image.MAX_IMAGE_PIXELS = 200_000_000


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def font(size):
    path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


def comparison(rgb, prediction, path):
    """Display the full crop plus the SAME central detail from each panel."""
    canvas = Image.new("RGB", (1100, 1160), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), "20_29: prediction on a new image crop", font=font(30), fill="#17212b")
    draw.text((24, 60), "Saved model | Image alone | No new training | Direct output, without cleanup", font=font(21), fill="#374151")
    panels = [Image.fromarray(rgb), Image.fromarray(PALETTE[prediction])]
    detail = (256, 320, 768, 680)
    for i, (panel, title) in enumerate(zip(panels, ["1  Microscopy image", "2  Model prediction at EVERY pixel"])):
        x = 24 + i * 540
        draw.text((x, 105), title, font=font(23), fill="#17212b")
        resample = Image.Resampling.LANCZOS if i == 0 else Image.Resampling.NEAREST
        canvas.paste(panel.resize((512, 512), resample), (x, 142))
        x0, y0, x1, y1 = detail
        draw.rectangle((x + x0 // 2, 142 + y0 // 2, x + x1 // 2, 142 + y1 // 2), outline="#00bac7", width=2)
        draw.text((x, 673), "Same boxed area, enlarged for viewing", font=font(20), fill="#374151")
        canvas.paste(panel.crop(detail), (x, 706))
    draw.text((24, 1089), "Red = Periderm     Green = Endoderm     Yellow = Background", font=font(23), fill="#17212b")
    draw.text((24, 1127), "No human reference supplied: accuracy has not been measured on this image.", font=font(21), fill="#374151")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verify inputs and model loading; do not predict or save outputs.")
    parser.add_argument("--no-open", action="store_true", help="Save the result without opening Preview.")
    args = parser.parse_args()
    weights = RUN / "model_weights.pt"
    if sha(IMAGE) != IMAGE_SHA or sha(weights) != WEIGHTS_SHA:
        raise ValueError("The selected image or frozen model changed. Stop and check the files before proceeding.")
    plan = json.loads((RUN / "experiment_plan.json").read_text())
    if plan["class_order"] != list(LABELS.values()):
        raise ValueError("Unexpected model class order.")
    with Image.open(IMAGE) as image:
        if image.mode != "RGB" or image.size != (14077, 8532):
            raise ValueError("Unexpected microscopy image dimensions or mode.")
        rgb = np.array(image.crop(BOX))
        full_dimensions = image.size
    if rgb.shape != (1024, 1024, 3):
        raise ValueError("The crop must contain exactly 1024 x 1024 unchanged RGB pixels.")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3, activation=None)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval().requires_grad_(False)
    if args.check_only:
        print("Verified: local 20_29 image, 1024 x 1024 crop, frozen weights and three-category model.")
        print("Model loaded successfully. No prediction, training or result-file writing occurred.")
        return
    print("20_29: using ONE 1024 x 1024 image crop at its original pixel scale.", flush=True)
    print("The saved model receives the image only. No human mask or annotations are read.", flush=True)
    print("Making a prediction on " + ("Apple GPU (MPS)." if device == "mps" else "CPU."), flush=True)
    model.to(device)
    x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None].to(device) / 255
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    with torch.inference_mode():
        probabilities = model((x - mean) / std).softmax(1)[0].cpu().numpy()
    if probabilities.shape != (3, 1024, 1024) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid prediction shape or scores.")
    prediction = (probabilities.argmax(0) + 1).astype(np.uint8)
    if sha(IMAGE) != IMAGE_SHA or sha(weights) != WEIGHTS_SHA:
        raise ValueError("Input files changed during prediction; no results saved.")
    PARENT.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(tempfile.mkdtemp(prefix=f"view_{stamp}_", dir=PARENT))
    image_path = out / "20_29_x7988_y5618_image.png"
    prediction_path = out / "predicted_mask.png"
    display = out / "OPEN_ME_image_and_prediction.png"
    Image.fromarray(rgb).save(image_path)
    Image.fromarray(prediction).save(prediction_path)
    Image.fromarray(PALETTE[prediction]).save(out / "predicted_mask_colours.png")
    np.savez_compressed(out / "model_probabilities.npz", probabilities=probabilities)
    comparison(rgb, prediction, display)
    shutil.copy2(__file__, out / Path(__file__).name)
    record = {
        "source_image": str(IMAGE), "source_image_sha256": IMAGE_SHA,
        "source_dimensions": full_dimensions, "crop_box_xyxy_exclusive": BOX,
        "crop_pixels_resized_or_adjusted": False,
        "crop_choice": "Fixed before predictions, using microscopy visibility only; no human tissue identity assumed.",
        "model_run": str(RUN), "model_weights_sha256": WEIGHTS_SHA,
        "architecture": "PAT transfer model: UNet++ / ResNet34, three categories",
        "model_checkpoint_update": 200, "device": device,
        "class_codes": LABELS, "normalization_mean": [.485, .456, .406],
        "normalization_std": [.229, .224, .225],
        "training_updates_this_run": 0, "reference_mask_read": False,
        "postprocessing": "None; argmax of model category scores only.",
        "prediction_sha256": sha(prediction_path), "crop_sha256": sha(image_path),
        "script_sha256": sha(__file__),
        "accuracy_measured": False, "included_in_our_recent_training": False,
        "original_PAT_training_exposure": "Not established; not a proven independent test.",
        "scope": "One crop, not the full slide. Lateral roots have no separate class in this model.",
        "inputs_unchanged": True,
    }
    (out / "prediction_details.json").write_text(json.dumps(record, indent=2) + "\n")
    (out / "READ_ME.txt").write_text(
        "Open OPEN_ME_image_and_prediction.png. Left: microscopy. Right: direct model prediction.\n"
        "Red=Periderm; green=Endoderm; yellow=Background. Cyan boxes mark matching enlarged views.\n"
        "No human mask was read; no learning or cleanup occurred. Accuracy is not measured.\n"
        "The model has no separate Lateral root class. This crop alone does not assess the whole slide.\n"
        "The image was not in our recent training; original public PAT exposure is not established.\n"
    )
    print("Prediction saved. No further training or cleanup occurred.", flush=True)
    print("Open the side-by-side result:\n" + str(display), flush=True)
    print("We have not measured accuracy: this image has no human reference labels.", flush=True)
    print("Original image and saved model unchanged.", flush=True)
    if not args.no_open:
        opened = subprocess.run(["/usr/bin/open", "-a", "Preview", str(display)], check=False)
        if opened.returncode:
            print("Preview did not open automatically. Open the result path above manually.")


if __name__ == "__main__":
    main()
