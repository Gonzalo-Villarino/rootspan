# What RootSpan measures

## Processing

1. Read the uploaded RGB microscopy image and its calibration.
2. Detect separated main-root tracks, ordered from top to bottom. The current orientation assumes the hypocotyl side is on the right.
3. Apply the original PAT four-class U-Net++ with a ResNet34 encoder. It predicts Background, Periderm, Endodermis and Lateral root. RootSpan's earlier teaching model was a smaller U-Net; it is not the model used for this final mask-based report.
4. Use class-safe tiling (1024-pixel tiles, 64-pixel halo, 896-pixel core) and original PAT 2D cleanup stages. This tiling differs from the original PAT full-image workflow.
5. Sample the cleaned periderm mask along the detected main-root path. Add the lengths of the periderm-labelled portions, then divide by pixels per micrometre.
6. Save the automatic value, any explicitly identified expert correction, overlays and CSV results.

The app also retains the earlier V1 transition detector and V2 path/transition diagnostics. The main numeric report uses cleaned periderm-mask path lengths; it does not require an accepted transition prediction. It does not use the original PAT FilFinder measurement stage. These distinctions matter when comparing numbers with PAT or Fiji.

## What remains uncertain

- Some root paths and right endpoints need visual review. Hypocotyl tissue can be included accidentally.
- A sum of periderm-labelled path portions is not necessarily the distance between the biological transition and root–hypocotyl junction.
- Small predicted fragments can produce a small nonzero number. An automatic zero only means no sampled periderm was detected; it does not establish biological absence.
- The special expert-zero record for `7_18` root 1 is a human observation, separately labelled and tied to the exact source/result. It is not learned model performance.
- Dark or blurry roots, touching roots, incomplete roots and different acquisition conditions can cause errors or missing measurements. Quality flags are heuristic, not a validated diagnostic.
- Training fit, repeatedly reviewed development images, human-selected regions and expert-corrected results are not independent test accuracy. The original PAT model's exposure to the archived slides is unknown.
- The transition detector was developed using reviewed fields and feedback. Neither a small set of approved transitions nor the locked version establishes performance on hundreds of new slides.

## What improved during the project

The early one-patch trials demonstrated the training loop. Adding a background category and improving labels helped fit those labelled examples. Later work moved to original PAT weights, context and path-based measurements. The final reporting change made an available per-root mask estimate visible even when the transition or anatomical endpoint was uncertain. This improved access to results; it did not itself prove that the biological measurement became more accurate.

The next scientific validation should use slides not used for training, model selection or repeated visual tuning; a fixed annotation/measurement definition; blinded manual measurements; and per-root error and failure-rate reporting. Keep the present version fixed during that comparison.
