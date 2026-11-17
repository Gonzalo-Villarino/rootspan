# RootSpan

**Periderm length estimates from root microscopy images.**

A project by **Gonzalo Villarino**. RootSpan combines a local upload interface, root tracing, tissue segmentation and downloadable measurements. It also preserves the step-by-step learning work that led to the app: annotations, pixel masks, patches, small U-Net exercises and later experiments using the published PAT model.

**Status: research and learning prototype, version 0.6.0-rootspan.1.** The underlying local release was approved by Gonzalo Villarino for a practical test on 17 November 2026. This is not a claim of validated biological accuracy. Review the predicted masks and root paths before using a number in research.

## Start here

1. [What RootSpan measures](docs/METHODS_AND_LIMITS.md)
2. [Install and open the app](#installation-macos)
3. [The learning journey and Word manuals](docs/LEARNING_GUIDE.md)
4. [Example measurements](examples/README.md)
5. [Data, model weights and reproducibility](docs/REPRODUCIBILITY.md)

## What you can do

- Upload a TIFF or other supported microscopy image through a browser on your own Mac.
- Detect and trace separated, approximately horizontal main roots.
- View tissue predictions and per-root results.
- Export periderm estimates in micrometres using your image calibration.
- Inspect quality flags and missing results rather than treating a failed measurement as zero.

The browser interface starts empty. Uploaded images and results stay on the computer running the app. This is a local application, not a hosted image-analysis service.

## What the number means

The main result is the **length of predicted periderm portions along a detected root path**, after tissue-mask cleanup. It can include several disconnected portions. It is not automatically the anatomical distance from a verified transition to a verified root–hypocotyl junction. An incorrectly placed path or endpoint can affect the number, including by extending into the hypocotyl.

A saved expert correction for one particular root in slide `7_18` is separately identified. It applies only when its image and result identifiers match. A new uploaded image does not receive that correction simply because it has a similar filename.

## Installation (macOS)

The recorded environment is Python **3.13.2 on Apple silicon**. Package versions are pinned in `requirements/`. Windows and Linux have not been verified. This first distribution is a source release; a clean installation on another computer still needs verification.

1. Install Python 3.13 and Git, then obtain this repository:

   ```bash
   git clone https://github.com/Gonzalo-Villarino/rootspan.git
   cd rootspan
   ```

2. Download the original PAT model from the [upstream model download](https://drive.google.com/file/d/1E5f0Heav8L_k9WxvvEQoBxC_J_zLH-J4/view), linked by the [original PAT project](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm). Its filename is `best_model_unet_plusplus_resnet34_periderm4classes_Lin_0301_1024_6.pth`.

3. Run the installer, replacing the example path with your downloaded file:

   ```bash
   python3.13 setup_rootspan.py --pat-checkpoint "/full/path/to/best_model_unet_plusplus_resnet34_periderm4classes_Lin_0301_1024_6.pth"
   ```

   This creates two project-local Python environments, checks the exact original model checksum, converts it to tensor-only weights, and verifies the runtime. It does not train a model. Setup can require several gigabytes for dependencies.

4. Double-click **RootSpan.command** in the project folder, or run:

   ```bash
   ./RootSpan.command
   ```

   Open **http://127.0.0.1:8766/** in Chrome (the browser used for verification). Keep the launcher open while analysing. Port 8766 lets RootSpan coexist with the earlier local app.

5. Upload a microscopy image with separated, approximately horizontal roots and the **hypocotyl side on the right**. Enter the correct **pixels per micrometre** for that original image. The example slides used `0.5299`, so `length in µm = length in pixels / 0.5299`. Do not reuse that calibration for a different acquisition without checking it.

If a root is dark, blurry, connected to another root or incompletely detected, inspect its quality status and overlay. A blank result means unavailable, not zero periderm.

## Project map

| Folder | Contents |
|---|---|
| `03_Notebooks/` | Numbered Python learning programs, plus an R plotting script |
| `05_Notes/` | Word learning manuals and progress notes |
| `02_Data/` | Compact annotations, review records and example crops |
| `04_Results/` | Retained compact results from the learning experiments |
| `06_Models/` | Transition checkpoint code, small learned head and provenance |
| `07_Periderm_Length_Proof/`, `08_Model_Refinement_20261116/` | Measurement and refinement experiments |
| `09_Periderm_GUI/` | Current RootSpan app and software tests |
| `10_…` through `17_…` | Historical investigations and release checks; not all are current methods |
| `examples/` | Two saved outputs from the approved reporting mode |
| `docs/` | Current method, learning guide, scope and reproducibility notes |

Some labels are hand-drawn; others are programmatically prepared or rule-derived development labels. Their provenance is described in [the data guide](02_Data/README.md); do not treat all annotations as independently verified human ground truth.

The numbered folders preserve the learning sequence. Earlier documents describe earlier methods and may show the former app name or local paths. They are historical records; use this README for current setup.

## Tests

After setup, run the software regression suite:

```bash
00_Software/ml-env/bin/python -m unittest discover -s 09_Periderm_GUI -p 'test_*.py' -v
```

These tests cover software behaviour, not biological accuracy on a new dataset. [Verification details](docs/REPRODUCIBILITY.md) distinguish local checks from a fresh installation.

## Credits and licence

RootSpan builds on [PAT: Pipeline for Analysis of Periderm](https://pmc.ncbi.nlm.nih.gov/articles/PMC10981931/) and the [original Salk HPI PAT repository](https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm), including Ling Zhang's work. The original source licence and attribution are retained. RootSpan is a separate project, not an official PAT release.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for reused code, model sources and licence scope. Original PAT model weights are obtained from the authors rather than redistributed here. See [CITATION.cff](CITATION.cff) for this project.
