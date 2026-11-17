# Third-party notices

## Original PAT

This project builds on the original PAT project by the Salk Harnessing Plants Initiative, including Ling Zhang's work.

- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10981931/
- Source: https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm
- Pinned source commit: `51cdf8966f6da17b711aeb5625205a8a2ca39467`
- Copyright (c) 2023 Salk Harnessing Plants Initiative.
- Source code licence: MIT, reproduced in `00_Software/PAT_Pinned_20261115/LICENSE` and `00_Software/PAT_Reference/LICENSE`.

The original source files are preserved. RootSpan adapts inference tiling and combines original cleanup stages with its own root paths and reporting. It does not claim to reproduce the original FilFinder measurement pipeline.

The original PAT checkpoint is not bundled; obtain it from the upstream download linked in README.md. The project licence does not grant rights in third-party model weights or source microscopy datasets. The small newly trained transition head is included with its provenance.

## Software libraries

PyTorch, torchvision, segmentation-models-pytorch, NumPy, SciPy, Pillow, scikit-image, OpenCV, pandas and other dependencies retain their own licences. They are installed from package distributions rather than copied into this repository. See `requirements/` for the versions used. Labelme and Fiji were used during annotation and manual-measurement exercises; they are not bundled.

The U-Net architecture and original PAT work remain credited in the learning records. A different project name does not transfer authorship of these methods to RootSpan.
