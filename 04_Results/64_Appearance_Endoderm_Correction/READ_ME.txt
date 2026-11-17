TEACHING AN ERROR CORRECTION — 16 November 2026

Gonzalo identified the first predicted T region as clear Endoderm: long rectangular cells, no short irregular Periderm cells there. We saved this as appearance example D. His100percent describes biological confidence, not measured model accuracy.

Simple explanation
The model had mistaken this kind of Endoderm appearance for mixed tissue. We supplied a correctly labelled example and fitted the small appearance classifier again. We did not tell it to move a line by a chosen distance. Its changed image predictions moved the line automatically through the unchanged boundary rule.

Result
At the original mistaken scan field, the predicted appearance changes from mixed to Endoderm. T moves right by128px, approximately242um. It is now approximately873um left of Gonzalo's reviewed T, compared with1114um before. It remains outside the reviewed interval. This is a partial improvement, not a solved automatic measurement model.

Controlled comparison
Same frozen published PAT encoder, head architecture,seed1617,500updates,AdamWsettings,preprocessing,scanning geometry,stride64px and ordered E/M/P rule. Equal total loss weight for each appearance class. Two Endoderm windows now share the Endoderm weight. A control rerun using the original three windows reproduced all295scan scores and T exactly after the code was generalized to accept four windows.

Important detail
The new teaching PNG is an axis-aligned crop. The scan uses a locally rotated crop along the root. Before this correction, the teaching PNG itself already had an Endoderm highest score, while the path-aligned scan field was called mixed. Endoderm scores increased for both; the original mistaken scan field now becomes Endoderm. This shows sensitivity to pose/context. It is not evidence of explicit cell-shape measurement. Nearby lateral branches remain a possible context confound.

Learning and evaluation
There are four reviewed windows from two roots on ONE slide. Root3 now contributes training information. Improvement on root3 is a teaching/correction check, not independent testing. More iterations alone are not what changed this comparison: the teaching examples changed. Original images,annotations,U-Net weights and earlier results are retained. The main-root route is still supplied; no new automatic periderm length is reported.

Files
BEFORE_AFTER_CORRECTION.png — before/after/reference T on the source image.
NEW_ENDODERM_EXAMPLE.png — new teaching example from the wrong region.
comparison.json — exact method, results and limitations.
added_endoderm_01/ — learned small appearance-head weights,processed inputs,scores,code and training history.
control_verification.json — original experiment reproduced exactly.

Repeat into a new folder:
"${ROOTSPAN_PROJECT}/00_Software/ml-env/bin/python" "${ROOTSPAN_PROJECT}/08_Model_Refinement_20261116/appearance_experiment.py" --labels "${ROOTSPAN_PROJECT}/02_Data/10_Cell_Appearance_Review/confirmed_appearance_labels_02.json" --run-name added_endoderm_repeat --output-parent "${ROOTSPAN_PROJECT}/04_Results/64_Appearance_Endoderm_Correction"

Next scientific step
Check the remaining early-mixed area in the source and add any further examples only with biological justification. Keep lateral-root context separate from main-root cell identity. Do not simply move T to the reference coordinate. Validate any final method on separate images after development.

Word manual was not edited. This record is saved for its next update.

Independent review: external reviewer review could not run because subscription login is required. No review was sent or received; no approval claimed. See review/review_status.json.
