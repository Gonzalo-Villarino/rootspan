START HERE — IMPROVED ROOT4 LEARNING RESULT

Open OPEN_ME_root4_before_after.png first.
The last panel is the improved model PLUS a separate continuity rule.
raw_vs_continuity_full.png shows what the model itself predicted and what
post-processing changed. Neither used your answer key to draw predictions.

WHAT CHANGED
We adapted the public PAT UNet++/ResNet34 model rather than continuing to train
only the tiny teaching U-Net. We prepared613 overlapping fields from16 roots
across3 slides, using your training annotations and archived human annotations.
Only23 fields show both tissues. The successful run prioritised these difficult
transition examples. Root4 and root6 were excluded from this new training.
Training ran800 updates. The best root4 validation checkpoint was update200;
later updates did not improve this check. The selected weights were reloaded
and their predictions reproduced exactly.

WHAT POST-PROCESSING DOES HERE
For this single primary root oriented Endoderm-left/Periderm-right, a rule uses
the model's scores to choose one transition and make each tissue region
continuous. Background pixels stay unchanged. This is an experimental rule
inspired by PAT's use of root continuity, not Ling's exact postprocessor.
The rule can be wrong when its orientation/one-transition assumption is wrong.
A clean-looking region alone is not evidence of biological accuracy.

MEASURED RESULTS (unchanged root4 reference, labelled pixels only)
IoU / overlap: Periderm33.2% ->65.8% raw model ->81.1% after rule.
IoU / overlap: Endoderm15.1% ->62.3% raw model ->72.6% after rule.
IoU / overlap: Background99.76% ->99.85% raw/after rule.
IoU penalises both missing and extra predicted pixels. It is not a confidence.
Recall after cleanup: Periderm100%, Endoderm75.17%, Background99.85%.
100% Periderm recall does NOT mean perfect segmentation:6158 labelled Endoderm
pixels are still called Periderm. The predicted transition is at x9307 on the
full slide and starts Periderm too early relative to your reference. Some
non-tissue artefact pixels are still called tissue. Do not conceal these errors.

STATUS
Substantial improvement, useful for continuing the learning exercise; not a
fully solved biological segmentation task or a research-ready validation.
Root4 has repeatedly guided model/settings selection and is development
validation. Original PAT exposure to this slide is unknown. An independently
acquired new slide is needed to assess generalisation. Original source images,
JSON annotations, reference mask and earlier models are unchanged.

FILES
selected_prediction.png — numeric final mask (1Periderm,2Endoderm,3Background).
raw_model_prediction.png — numeric model output before cleanup.
raw_model_probabilities.npy — model's scores after softmax, not human labels.
comparison_results.json — full paths, metrics, selection criteria and hashes.
metrics.csv — simple before/raw/after score table.
programs/ — reproducible comparison and helper code.
The weights and training record are in the selected_run path in the JSON.

NEXT LEARNING STEP
First look at the before/after comparison and the remaining boundary error.
Freeze this version. Then choose a genuinely separate slide for a new check,
with biological labels confirmed before seeing model output. Do not keep
changing root4 until it looks perfect.

Learning notes:05_Notes/04_PAT_and_Root4_Repair_20261115.txt.
The existing Word manual was not edited; these records are for later integration.
