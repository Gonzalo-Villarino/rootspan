"""Train a fresh three-category U-Net on ONE existing image/mask pair.

Learning exercise. The same microscopy patch
used in the two-class lesson is paired with the updated mask from program 08.
Only explicit Periderm, Endoderm and Background labels supply training answers.
Unlabelled pixels are never automatically treated as Background.

The model sees the RGB image, not the mask. The loss compares its raw scores
with mask-derived targets: mask 1 -> target 0, 2 -> 1, 3 -> 2. Mask 0 and 255
both map to ignored target 255. Class scores and mask codes are different.

Settings match the earlier 200-update lesson where applicable: fresh seed 34,
Adam, learning rate 0.001, gradient clipping 5, RGB / 255, no resizing or
augmentation. The third output and additional labelled pixels change the task;
the numerical loss is not directly comparable with the old two-class loss.
Measurements use the training pair only, not independent validation/test data.
"""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = Path(__file__).with_name("09_three_class_unet.py")
_spec = importlib.util.spec_from_file_location("lesson_three_class_unet", MODEL_FILE)
unet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(unet)
PATCH_DETAILS = ROOT / "04_Results/07_Background_Patch/patch_details.json"
OUTPUT_PARENT = ROOT / "04_Results/08_Three_Class_Training"
PATCH_ID = "MT25_rep3_practice_r000_c007"
IGNORE = 255
SEED = 34
LEARNING_RATE = 0.001
CLIP_NORM = 5.0


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def mask_to_target(mask):
    """Convert numeric mask codes to score positions without changing the mask."""
    if mask.ndim != 2 or mask.dtype.kind not in "ui" or not np.isin(mask, [0, 1, 2, 3, 255]).all():
        raise ValueError("Expected a 2D integer mask containing only 0, 1, 2, 3, 255.")
    target = np.full(mask.shape, IGNORE, dtype=np.int64)
    for channel, code in enumerate(unet.OUTPUT_MASK_CODES):
        target[mask == code] = channel
    if not (target != IGNORE).any():
        raise ValueError("No labelled pixels to learn from.")
    return torch.from_numpy(target).unsqueeze(0)


def load_practice_pair(details_path=PATCH_DETAILS):
    details_path = Path(details_path).resolve()
    details_bytes = details_path.read_bytes()
    details = json.loads(details_bytes)
    if (details["patch_id"] != PATCH_ID or details["size"] != [1024, 1024]
            or details["source_crop_box"] != [7168, 0, 8192, 1024]
            or details["trainable_class_codes"] != list(unet.OUTPUT_MASK_CODES)
            or details["ignore_codes"] != list(unet.IGNORE_MASK_CODES)):
        raise ValueError("Patch record does not match this three-category lesson.")
    image_path = details_path.parent / details["image_file"]
    mask_path = details_path.parent / details["mask_file"]
    image_bytes, mask_bytes = image_path.read_bytes(), mask_path.read_bytes()
    if (sha256_bytes(image_bytes) != details["image_sha256"]
            or sha256_bytes(mask_bytes) != details["mask_sha256"]):
        raise ValueError("Image or mask has changed since the patch record was saved.")
    with Image.open(io.BytesIO(image_bytes)) as im:
        if im.mode != "RGB" or im.size != (1024, 1024):
            raise ValueError("Expected the existing 1024 x 1024 RGB image patch.")
        rgb = np.array(im)
    with Image.open(io.BytesIO(mask_bytes)) as im:
        if im.mode not in ("L", "P") or im.size != (1024, 1024):
            raise ValueError("Use the 1024 x 1024 numeric mask, not the colour preview.")
        mask = np.array(im)
    target = mask_to_target(mask)
    counts = {str(k): int((mask == k).sum()) for k in (0, 1, 2, 3, 255)}
    if counts != details["pixel_counts"] or not all(counts[str(k)] for k in (1, 2, 3)):
        raise ValueError("Mask counts differ from the record or a category is missing.")
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    provenance = {
        "patch_id": PATCH_ID, "image": str(image_path), "mask": str(mask_path),
        "image_sha256": sha256_bytes(image_bytes), "mask_sha256": sha256_bytes(mask_bytes),
        "patch_details": str(details_path), "patch_details_sha256": sha256_bytes(details_bytes),
        "source_crop_box": details["source_crop_box"], "mask_pixel_counts": counts,
        "earlier_two_class_trial_record": details["trial_record"],
    }
    return image, target, provenance


def statistics_from_scores(scores, target):
    """Assess only labelled pixels; rows = reference, columns = prediction."""
    if (scores.ndim != 4 or scores.shape[1] != 3
            or scores.shape[0] != target.shape[0] or scores.shape[2:] != target.shape[1:]):
        raise ValueError("Three score maps and the target must have matching dimensions.")
    if not bool(torch.isfinite(scores).all().item()):
        raise RuntimeError("Non-finite model scores.")
    valid = target != IGNORE
    if not bool(valid.any().item()):
        raise ValueError("There are no labelled pixels to assess.")
    loss = torch.nn.functional.cross_entropy(scores, target, ignore_index=IGNORE)
    truth = target[valid].detach().cpu().numpy()
    predicted = scores.argmax(1)[valid].detach().cpu().numpy()
    confusion = np.bincount(3 * truth + predicted, minlength=9).reshape(3, 3)
    per_class = {}
    for i, label in enumerate(unet.OUTPUT_LABELS):
        total = int(confusion[i].sum())
        correct = int(confusion[i, i])
        per_class[label] = {
            "reference_pixels": total, "correct_pixels": correct,
            "fraction_of_reference_pixels_matched": correct / total if total else None,
        }
    loss_value = float(loss.item())
    if not np.isfinite(loss_value):
        raise RuntimeError("Non-finite loss.")
    return {
        "loss": loss_value,
        "labelled_pixel_agreement": float(np.trace(confusion) / confusion.sum()),
        "labelled_pixels_assessed": int(confusion.sum()),
        "confusion_rows_reference_columns_prediction": confusion.tolist(),
        "class_order": list(unet.OUTPUT_LABELS), "per_class": per_class,
        "scope": "Only labelled pixels of the SAME training patch; not whole-image or new-image accuracy.",
    }


def measure(model, image, target):
    model.eval()
    with torch.inference_mode():
        return statistics_from_scores(model(image), target)


def train_trial(image, target, device, steps=200, max_seconds=180, progress=None):
    if not 1 <= steps <= 200 or not 1 <= max_seconds <= 600:
        raise ValueError("Use 1-200 updates and 1-600 seconds.")
    if tuple(sorted(torch.unique(target).tolist())) != (0, 1, 2, IGNORE):
        raise ValueError("This practice pair must contain all three targets and ignored pixels.")
    torch.manual_seed(SEED)
    model = unet.ThreeClassUNet().to(device)
    image, target = image.to(device), target.to(device)
    before = measure(model, image, target)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    history = []
    started = time.perf_counter()
    stopped = "requested_updates_completed"
    for update in range(1, steps + 1):
        if history and time.perf_counter() - started >= max_seconds:
            stopped = "time_limit_reached_between_updates"
            break
        optimizer.zero_grad(set_to_none=True)
        scores = model(image)
        loss = torch.nn.functional.cross_entropy(scores, target, ignore_index=IGNORE)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("Non-finite loss; stopping the trial.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM, error_if_nonfinite=True)
        optimizer.step()
        value = float(loss.item())
        history.append({"update": update, "loss_before_update": value})
        if progress and (update == 1 or update % 5 == 0 or update == steps):
            progress(update, value)
    after = measure(model, image, target)
    return model, before, after, history, stopped, time.perf_counter() - started


def save_trial(model, record, history, program_sources, parent):
    """Save a separate run; retain code copies needed to recreate the model."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(tempfile.mkdtemp(prefix=f"run_{stamp}_", dir=parent))
    weights_path = output / "model_weights.pt"
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, weights_path)
    with (output / "training_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["update", "loss_before_update"])
        writer.writeheader()
        writer.writerows(history)
    programs = output / "programs"
    programs.mkdir()
    for name, data in program_sources.items():
        (programs / name).write_bytes(data)
    record = dict(record, model_weights_sha256=sha256_bytes(weights_path.read_bytes()),
                  saved_programs_sha256={name: sha256_bytes(data) for name, data in program_sources.items()})
    (output / "trial_details.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    (output / "READ_ME.txt").write_text(
        "Three-category learning trial on one microscopy patch.\n"
        "model_weights.pt contains learned model numbers. programs contains the model recipe.\n"
        "trial_details.json records inputs, labels, settings and results; training_history.csv records updates.\n"
        "All measurements are restricted to labelled training pixels. Unlabelled/conflicting pixels are ignored.\n"
        "No independent validation or testing. Weights are for prediction; exact training resumption is not included.\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--max-seconds", type=int, default=180)
    parser.add_argument("--patch-details", type=Path, default=PATCH_DETAILS)
    parser.add_argument("--output-parent", type=Path, default=OUTPUT_PARENT)
    parser.add_argument("--check-only", action="store_true", help="Read/check the pair without training or saving.")
    args = parser.parse_args()
    if not 1 <= args.steps <= 200 or not 1 <= args.max_seconds <= 600:
        parser.error("Use 1-200 updates and 1-600 seconds.")
    image, target, provenance = load_practice_pair(args.patch_details)
    print(f"Practice pair: {PATCH_ID} (1024 x 1024 pixels).", flush=True)
    print("Same image patch; updated mask with Periderm, Endoderm and Background.", flush=True)
    print("Image = input. Your mask = answer key. Unlabelled/conflicting pixels ignored.", flush=True)
    for code, label in zip(unet.OUTPUT_MASK_CODES, unet.OUTPUT_LABELS):
        print(f"  {label}: {provenance['mask_pixel_counts'][str(code)]:,} labelled pixels", flush=True)
    if args.check_only:
        print("Pair check passed. No model created, training performed or files written.")
        return

    program_sources = {p.name: p.read_bytes() for p in (Path(__file__), MODEL_FILE, unet.BASE_MODEL_FILE)}
    device = unet.choose_device(args.device)
    print(f"Fresh three-category model on {'Apple GPU (MPS)' if device.type == 'mps' else 'CPU'}.", flush=True)
    print(f"Up to {args.steps} updates of ONE pair; training-loop limit about {args.max_seconds} seconds.", flush=True)
    print("The time limit is checked between updates; setup and saving take extra time.", flush=True)
    print("Loss measures mismatch with your updated mask; lower is better for this example.", flush=True)
    model, before, after, history, stopped, elapsed = train_trial(
        image, target, device, args.steps, args.max_seconds,
        progress=lambda step, loss: print(f"Update {step}/{args.steps} | loss before update: {loss:.4f}", flush=True),
    )
    record = {
        "purpose": "Three-class one-pair training practice only; no independent validation/test.",
        "created_utc": datetime.now(timezone.utc).isoformat(), "provenance": provenance,
        "training_script_sha256": sha256_bytes(program_sources[Path(__file__).name]),
        "model_script_sha256": sha256_bytes(program_sources[MODEL_FILE.name]),
        "base_model_script_sha256": sha256_bytes(program_sources[unet.BASE_MODEL_FILE.name]),
        "model_class": "ThreeClassUNet", "parameter_count": sum(p.numel() for p in model.parameters()),
        "python": platform.python_version(), "pytorch": str(torch.__version__), "device": str(device),
        "seed": SEED, "optimizer": "Adam", "learning_rate": LEARNING_RATE, "gradient_clip_norm": CLIP_NORM,
        "requested_updates": args.steps, "completed_updates": len(history), "max_loop_seconds": args.max_seconds,
        "loop_and_final_measure_seconds": elapsed, "stop_reason": stopped,
        "image_preprocessing": "RGB / 255; no resizing or augmentation",
        "mask_to_target": {"0": "ignore", "1": 0, "2": 1, "3": 2, "255": "ignore"},
        "output_channels": list(unet.OUTPUT_LABELS), "output_mask_codes": list(unet.OUTPUT_MASK_CODES),
        "ignore_target": IGNORE, "before_training": before, "after_training": after,
        "limitations": [
            "All measurements use labelled pixels of the training pair, not a new image or the whole patch.",
            "Only explicit Background outlines provide background supervision. Other empty space is ignored.",
            "One patch from one slide cannot establish generalization to new roots or slides.",
            "Loss values from the two-class and three-class tasks are not directly comparable.",
            "Training starts fresh; weights support prediction, not exact optimizer resumption.",
            "A seed improves repeatability but does not guarantee identical results across devices/versions.",
        ],
    }
    output = save_trial(model, record, history, program_sources, args.output_parent)
    print(f"\nFinished {len(history)} updates. Training loss: {before['loss']:.6f} -> {after['loss']:.6f}")
    if after["loss"] >= before["loss"]:
        print("Loss did not improve in this short trial; we will inspect it together.")
    print("Match to your labelled pixels on this SAME training patch:")
    for label in unet.OUTPUT_LABELS:
        result = after["per_class"][label]
        print(f"  {label}: {result['correct_pixels']:,} / {result['reference_pixels']:,} matched "
              f"({100 * result['fraction_of_reference_pixels_matched']:.2f}%).")
    print("These percentages do not assess unlabelled space or performance on new images.")
    print(f"Learned model numbers and training record saved in:\n{output}")
    print("Original images, annotations, masks, patches and earlier models are unchanged.")


if __name__ == "__main__":
    main()
