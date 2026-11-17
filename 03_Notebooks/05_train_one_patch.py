"""First learning trial: fit our small U-Net to ONE existing image-mask pair.

Learning exercise. This is training practice, not an
independent test of biological accuracy. Nothing is downloaded or overwritten.
The model sees only the RGB image; the mask supplies the answers for the loss.
Mask 1 -> target 0 (Periderm), mask 2 -> target 1 (Endoderm).
Mask 0 (unannotated) and 255 (uncertain overlap) are ignored, NOT background.

PyTorch references:
https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html
https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = Path(__file__).with_name("04_small_unet.py")
spec = importlib.util.spec_from_file_location("lesson_unet", MODEL_FILE)
unet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unet)
PATCH_ROOT = ROOT / "04_Results/03_Practice_Patches"
PATCH_ID = "MT25_rep3_practice_r000_c007"
IGNORE = 255


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mask_to_target(mask):
    """Read numeric mask codes, never RGB preview colours."""
    if mask.ndim != 2 or not np.isin(mask, [0, 1, 2, 255]).all():
        raise ValueError("Expected a 2D numeric mask containing only 0, 1, 2, 255.")
    target = np.full(mask.shape, IGNORE, dtype=np.int64)
    target[mask == 1] = 0
    target[mask == 2] = 1
    if not (target != IGNORE).any():
        raise ValueError("No labelled tissue pixels to learn from.")
    return torch.from_numpy(target).unsqueeze(0)


def load_practice_pair():
    manifest_path = PATCH_ROOT / "patch_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    matches = [p for p in manifest["patches"] if p["id"] == PATCH_ID]
    if len(matches) != 1:
        raise ValueError("The expected practice pair is missing or duplicated.")
    entry = matches[0]
    image_path = PATCH_ROOT / entry["image"]
    mask_path = PATCH_ROOT / entry["mask"]
    with Image.open(image_path) as im:
        if im.mode != "RGB":
            raise ValueError("Expected the existing RGB image patch.")
        rgb = np.array(im)
    with Image.open(mask_path) as im:
        if im.mode not in ("P", "L"):
            raise ValueError("Use the numeric mask, not the coloured preview.")
        mask = np.array(im)
    if rgb.shape != (1024, 1024, 3) or mask.shape != (1024, 1024):
        raise ValueError("Image and mask must both be 1024 x 1024. No resizing is done.")
    counts = {str(k): int((mask == k).sum()) for k in (0, 1, 2, 255)}
    expected = dict(entry["source_pixel_counts_excluding_padding"])
    expected["0"] += entry["padding_pixels"]
    if counts != expected or not counts["1"] or not counts["2"]:
        raise ValueError("Mask counts differ from the manifest or a tissue is missing.")
    target = mask_to_target(mask)
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    provenance = {
        "patch_id": PATCH_ID,
        "selection_reason": "Existing grid pair with the most pixels in its less abundant tissue class.",
        "image": str(image_path), "mask": str(mask_path),
        "image_sha256": sha256(image_path), "mask_sha256": sha256(mask_path),
        "manifest_sha256": sha256(manifest_path),
        "source_crop_box": entry["source_crop_box"], "mask_pixel_counts": counts,
    }
    return image, target, provenance


def measure(model, image, target):
    """Loss and pixel agreement on the SAME pair used for training."""
    model.eval()
    with torch.inference_mode():
        logits = model(image)
        loss = torch.nn.functional.cross_entropy(logits, target, ignore_index=IGNORE)
        valid = target != IGNORE
        correct = (logits.argmax(1) == target) & valid
        result = {"loss": float(loss.item()),
                  "labelled_pixel_agreement": float(correct.sum().item() / valid.sum().item())}
    if not np.isfinite(list(result.values())).all():
        raise RuntimeError("Non-finite result; training cannot be assessed.")
    return result


def train_trial(image, target, device, steps=30, max_seconds=180, progress=None):
    """Start fresh. Repeat one pair; cap updates and elapsed training-loop time."""
    torch.manual_seed(34)
    model = unet.SmallUNet().to(device)
    image, target = image.to(device), target.to(device)
    before = measure(model, image, target)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    history = []
    started = time.perf_counter()
    stopped = "requested_updates_completed"
    for update in range(1, steps + 1):
        if history and time.perf_counter() - started >= max_seconds:
            stopped = "time_limit_reached_between_updates"
            break
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = torch.nn.functional.cross_entropy(logits, target, ignore_index=IGNORE)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("Non-finite loss; stopping the trial.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        # The loss describes the guesses BEFORE this update, not after it.
        value = float(loss.item())
        history.append({"update": update, "loss_before_update": value})
        if progress and (update == 1 or update % 5 == 0 or update == steps):
            progress(update, value)
    after = measure(model, image, target)
    elapsed = time.perf_counter() - started
    return model, before, after, history, stopped, elapsed


def save_trial(model, record, history, parent):
    """Make a new run folder each time; never replace an earlier lesson."""
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(tempfile.mkdtemp(prefix=f"run_{stamp}_", dir=parent))
    # Portable learned numbers; the U-Net structure remains in program 04.
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output / "model_weights.pt")
    with (output / "training_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["update", "loss_before_update"])
        writer.writeheader()
        writer.writerows(history)
    (output / "trial_details.json").write_text(json.dumps(record, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--max-seconds", type=int, default=180)
    parser.add_argument("--check-only", action="store_true", help="Read/check the pair; no model or training.")
    args = parser.parse_args()
    if not 1 <= args.steps <= 200 or not 1 <= args.max_seconds <= 600:
        parser.error("Use 1–200 steps and 1–600 seconds.")
    image, target, provenance = load_practice_pair()
    print(f"Practice pair: {PATCH_ID} (1024 x 1024 pixels).", flush=True)
    print("Image = input. Your mask = answer key. Unlabelled/uncertain pixels ignored.", flush=True)
    if args.check_only:
        print("Pair check passed. No model created, training performed or files written.")
        return
    device = unet.choose_device(args.device)
    print(f"First training trial on {'Apple GPU (MPS)' if device.type == 'mps' else 'CPU'}.", flush=True)
    print(f"Up to {args.steps} updates of ONE pair; training-loop limit about {args.max_seconds} seconds.", flush=True)
    print("The time limit is checked between updates; setup and saving take extra time.", flush=True)
    print("Loss measures mismatch with your mask; lower is better for this example.", flush=True)
    model, before, after, history, stopped, elapsed = train_trial(
        image, target, device, args.steps, args.max_seconds,
        progress=lambda step, loss: print(f"Update {step}/{args.steps} | loss before update: {loss:.4f}", flush=True),
    )
    record = {
        "purpose": "One-pair training practice only. No independent validation/test.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "training_script_sha256": sha256(__file__), "model_script_sha256": sha256(MODEL_FILE),
        "python": platform.python_version(), "pytorch": str(torch.__version__), "device": str(device),
        "seed": 34, "optimizer": "Adam", "learning_rate": 0.001, "gradient_clip_norm": 5.0,
        "requested_updates": args.steps, "completed_updates": len(history),
        "max_loop_seconds": args.max_seconds, "loop_and_final_measure_seconds": elapsed,
        "stop_reason": stopped, "image_preprocessing": "RGB / 255; no resizing or augmentation",
        "mask_to_target": {"0": "ignore", "1": 0, "2": 1, "255": "ignore"},
        "output_channels": ["Periderm", "Endoderm"], "ignore_target": IGNORE,
        "before_training": before, "after_training": after,
        "limitations": ["All measurements use the training pair; they do not measure new-image performance.",
                        "No background class is learned. Predictions outside annotations are not assessed.",
                        "This run starts fresh; saved weights support later prediction, not exact optimizer resumption.",
                        "A seed improves repeatability but does not guarantee identical results across devices/versions."],
    }
    output = save_trial(model, record, history, ROOT / "04_Results/04_First_Training_Trial")
    print(f"\nFinished {len(history)} updates. Training loss: {before['loss']:.4f} -> {after['loss']:.4f}")
    if after["loss"] >= before["loss"]:
        print("The loss did not improve in this short trial; we will inspect it together.")
    print(f"Learned model numbers and training record saved in:\n{output}")
    print("This used one practice pair. Performance on new images is still unknown.")
    print("Your original image, annotations, masks and patches are unchanged.")


if __name__ == "__main__":
    main()
