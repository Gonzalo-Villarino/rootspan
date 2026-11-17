"""Prepare a small U-Net with three possible answers for each pixel.

Learning exercise. This reuses the small U-Net
structure in 04_small_unet.py and replaces its final two-score layer with a
three-score layer. The earlier program and saved two-class models are unchanged.

Running this file creates a fresh, UNTRAINED model in memory and checks it with
an invented image. It does not read microscopy images, masks or saved weights.
It does not train, download anything or save a model. The reusable model recipe
is this Python file; learned weights will be saved in a later training lesson.

Score order: Periderm, Endoderm, Background. Their saved mask codes are 1, 2, 3.
The score positions inside PyTorch are 0, 1, 2; these are not mask codes.
Mask codes 0 (unlabelled) and 255 (conflict) must stay ignored during later
training and assessment. No training or mask conversion is implemented here.
"""

import argparse
import importlib.util
import platform
import sys
from pathlib import Path

# This setup check does not need to create Python bytecode cache files.
sys.dont_write_bytecode = True

import torch
from torch import nn


BASE_MODEL_FILE = Path(__file__).with_name("04_small_unet.py")
_spec = importlib.util.spec_from_file_location("lesson_two_class_structure", BASE_MODEL_FILE)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

OUTPUT_LABELS = ("Periderm", "Endoderm", "Background")
OUTPUT_MASK_CODES = (1, 2, 3)
IGNORE_MASK_CODES = (0, 255)
choose_device = _base.choose_device


class ThreeClassUNet(_base.SmallUNet):
    """Same small U-Net body, with three raw scores per pixel.

    Input: RGB floats shaped [batch, 3, height, width].
    Output: scores shaped [batch, 3, height, width], in OUTPUT_LABELS order.
    Input's three channels are colours; output's three channels are categories.
    Scores are not percentages. Height and width must be multiples of 8.
    """

    def __init__(self):
        super().__init__()
        # The body supplies 16 features at each pixel. This final layer turns
        # those features into three category scores instead of two.
        self.output = nn.Conv2d(self.output.in_channels, len(OUTPUT_LABELS), kernel_size=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--size", type=int, default=64,
                        help="Invented square size; default 64. This does not resize your patches.")
    args = parser.parse_args()
    if not 8 <= args.size <= 1024 or args.size % 8:
        parser.error("--size must be a multiple of 8 between 8 and 1024.")

    torch.manual_seed(34)
    device = choose_device(args.device)
    model = ThreeClassUNet().to(device).eval()
    invented_image = torch.zeros((1, 3, args.size, args.size),
                                dtype=torch.float32, device=device)
    with torch.inference_mode():
        scores = model(invented_image)
        if tuple(scores.shape) != (1, 3, args.size, args.size):
            raise RuntimeError("The model did not produce three scores at every input position.")
        if not bool(torch.isfinite(scores).all().item()):
            raise RuntimeError("The model produced an invalid score.")

    print("Three-category U-Net setup check passed.")
    print(f"Python: {platform.python_version()} | PyTorch: {torch.__version__}")
    print(f"Computer used: {'Apple GPU (MPS)' if device.type == 'mps' else 'CPU'}")
    print(f"Adjustable model numbers: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Invented input: {args.size} x {args.size} pixels with 3 colour channels (RGB).")
    print(f"Output: {args.size} x {args.size} positions with 3 raw category scores each.")
    print("One score for Periderm, one for Endoderm and one for Background.")
    print("Mask codes remain: 1 = Periderm; 2 = Endoderm; 3 = Background.")
    print("Mask codes 0 (unlabelled) and 255 (conflict) will remain ignored in training.")
    print("These untrained scores are not percentages and have no biological meaning yet.")
    print("Your real image patches remain 1024 x 1024; this small input is only a setup check.")
    print("No microscopy images, masks or saved model weights were read. No training occurred.")
    print("No model was saved and no files were written by this check.")


if __name__ == "__main__":
    main()
