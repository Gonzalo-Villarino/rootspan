FIXED-DETECTOR CHECK ON 20_29 ROOT 4
16 November 2026

What we did
We kept the saved two-appearance detector unchanged. It had learned from five windows on COL0. We traced the main root from the 20_29 microscopy image using fluorescence geometry, checked the path visually, and predicted along it. Your reference mask was opened for comparison only after the prediction had been saved. No training occurred.

Result in simple words
The detector did not locate a supported T. It called the left-hand Endoderm mixed appearance. It found some Periderm-only windows toward the right, but no clear-Endoderm run before them. Its fixed rule therefore returned no T.

What the numbers mean
Ten overlapping windows were evaluated: seven mixed, three Periderm-only, zero Endoderm-only. These are not ten independent biological samples and not a percentage accuracy. Both appearance scores are predictions, not calibrated certainty.

The first input window contains your Endoderm reference labels and no Periderm reference labels. Unlabelled pixels remain unverified. This supports the finding that the early mixed calls are a problem; it does not prove which visual feature caused them. We have not established poor image quality as the cause.

Reference boundary
The gap between the last Endoderm-labelled path point and first Periderm-labelled path point spans path distances455–466pixels, with midpoint460.5. This is a working reference inferred from your existing polygons, not an exact manually marked transition point. Since the model returned no T, no model-versus-reference distance or automatic periderm length is reported.

What we learned
The earlier correction fitted the reviewed teaching root, but it did not transfer reliably to this other image. This is the distinction between fitting teaching examples and generalisation: using what was learned on another image. The next improvement should use a broader, consistently reviewed appearance teaching set across several roots and images. Repeatedly adjusting one line cannot establish that ability.

Next step
Keep this failed check and the frozen model as a baseline. Select a small batch of existing examples showing clear Endoderm, mixed appearance and clear Periderm from more than one source image. Review their biological identities before fitting again. Keep a separate image for a later check. Do not train on this crop and then present success on it as independent validation.

Limits
20_29 has been used for earlier development, so this is not a fresh blind test. Membership in the original published PAT training set is unknown. The route is image-derived and follows the main root, but rectangular classifier windows may still contain surrounding branch pixels. The classifier predicts appearances in windows; it does not identify or count individual cells. This crop is a local transition view, not a complete main root for length measurement.

Files
RESULT_FOR_LEARNING.png: simple source/reference/prediction comparison.
PREPARED_MAIN_ROOT_PATH.png: path used for sampling.
prediction_before_reference_comparison.json: original fixed prediction.
comparison_result.json: evaluation against the existing reference.
reference_support_after_prediction.json: reference-label support in each image window.
20_29_PRESENCE_SCORES.png: raw appearance scores.
verification.json: preservation and consistency checks.
windows/: the exact processed images shown to the detector.

Reproduce into a new folder:

"${ROOTSPAN_PROJECT}/00_Software/ml-env/bin/python" "${ROOTSPAN_PROJECT}/08_Model_Refinement_20261116/check_frozen_20_29.py" --phase all --output "${ROOTSPAN_PROJECT}/04_Results/66_Fixed_20_29_Check_repeat"

The command refuses to overwrite an existing output folder. The exact executed program and helper copies are saved here. Original images, annotations and all model weights are unchanged. The Word manual was not edited; these notes are saved for its next update.

Reading the coloured bars: the human Periderm region includes transition tissue under our measurement convention. A purple model window within that red region is not automatically an error. The concerning finding here is mixed appearance predicted in the left-hand region labelled Endoderm.

Independent external reviewer review could not run because subscription login is required. No review was sent or approval claimed. See review/review_status.json.
