"""A small U-Net for Gonzalo's root-image learning exercise.

U-Net is a neural-network design. PyTorch supplies its building blocks.
This compact teaching implementation is inspired by Ronneberger et al. (2015):
https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
It is not the original authors' code or Ling's trained PAT model.

Running this file creates a FRESH, UNTRAINED model and checks its output size
using an invented array. It does not open root images/masks, download weights,
change model weights by learning, or save a trained model.

For later lessons: output channel 0 represents Periderm (saved mask code 1);
output channel 1 represents Endoderm (saved mask code 2). Saved mask codes
0 (unannotated/padding) and 255 (uncertain overlap) must be ignored in training
loss and metrics. They are NOT additional learned classes. Read the mask's
palette indices, not its display colours. No training is implemented here.
"""

import argparse
import platform

import torch
from torch import nn


OUTPUT_LABELS = ("Periderm", "Endoderm")
OUTPUT_MASK_CODES = (1, 2)
IGNORE_MASK_CODES = (0, 255)


class TwoConvolutions(nn.Sequential):
    """Two layers that learn to recognize local image patterns."""

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )


class SmallUNet(nn.Module):
    """Three down/up levels, with 16, 32, 64 and 128 internal channels.

    Input: floating-point RGB images shaped [batch, 3, height, width].
    Output: two raw scores per pixel, shaped [batch, 2, height, width].
    The scores are not probabilities. Height and width must be multiples of 8.
    """

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = TwoConvolutions(3, 16)
        self.enc2 = TwoConvolutions(16, 32)
        self.enc3 = TwoConvolutions(32, 64)
        self.bottom = TwoConvolutions(64, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = TwoConvolutions(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = TwoConvolutions(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = TwoConvolutions(32, 16)
        self.output = nn.Conv2d(16, len(OUTPUT_LABELS), kernel_size=1)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected [batch, 3, height, width] RGB input.")
        if any(size < 8 or size % 8 for size in image.shape[-2:]):
            raise ValueError("Height and width must be positive multiples of 8.")
        if not image.is_floating_point():
            raise ValueError("Convert image pixels to floating-point numbers first.")

        # Smaller views gather context from a wider area of the image.
        e1 = self.enc1(image)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        middle = self.bottom(self.pool(e3))

        # Bring back detail from the corresponding earlier layers (skip links).
        d3 = self.dec3(torch.cat((self.up3(middle), e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        return self.output(d1)


def choose_device(requested):
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple GPU support is unavailable in this installation.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--size", type=int, default=64, help="Invented square size; default 64.")
    args = parser.parse_args()
    if not 8 <= args.size <= 1024 or args.size % 8:
        parser.error("--size must be a multiple of 8 between 8 and 1024.")

    torch.manual_seed(34)
    device = choose_device(args.device)
    model = SmallUNet().to(device).eval()
    # This is invented data, not one of Gonzalo's microscopy patches.
    example = torch.zeros((1, 3, args.size, args.size), dtype=torch.float32, device=device)
    with torch.inference_mode():
        scores = model(example)
        assert tuple(scores.shape) == (1, 2, args.size, args.size)
        assert bool(torch.isfinite(scores).all().item()), "Non-finite model output."

    print("Small U-Net setup check passed.")
    print(f"Python: {platform.python_version()} | PyTorch: {torch.__version__}")
    print(f"Computer used: {'Apple GPU (MPS)' if device.type == 'mps' else 'CPU'}")
    print(f"Adjustable model numbers: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Invented input: {args.size} x {args.size} pixels with 3 colour channels.")
    print(f"Output: {args.size} x {args.size} positions with 2 raw tissue scores each.")
    print("Later labels: Periderm and Endoderm.")
    print("The scores are untrained and have no biological meaning yet.")
    print("No root images or masks were read. No training occurred. No files were written.")


if __name__ == "__main__":
    main()
