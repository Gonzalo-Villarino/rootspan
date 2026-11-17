LOCKED TRANSITION CHECKPOINT v1 — 16 November 2026

This is the version Gonzalo approved: the solid blue T on 2_12 root 1.

WHAT IS LOCKED?
The learned image-recognition weights, image preparation, T-selection rule,
settings, source code and example inputs are preserved together. The checkpoint
files are read-only. File fingerprints detect accidental changes. Any later
improvements belong in a NEW version folder.

WHAT WAS VERIFIED?
The saved copy was run from outside the project folder. It reproduced every
appearance score exactly and placed T at exactly the approved position.
The rule also reproduced all four saved profiles: the accepted COL0 and A04
positions stayed unchanged; 20_29 still returned no supported T.
These checks establish computational reproduction, not accuracy on new roots.

WHAT DOES IT DO?
Given a microscopy crop and a checked path along the main root, it looks for
Endoderm-like and Periderm-like appearances. A mixed call requires both.
It selects a supported T or returns no T. The continuity rule rejects an early
candidate when a long stretch of clear Endoderm contradicts that candidate.
The blue T represents the left edge of the mixed-cell zone. For the agreed
regional measurements, the mixed zone belongs with Periderm.

This checkpoint combines an image-recognition model with a T-selection rule.
The latest improvement changed the rule; it did not retrain the weights.
It is not yet a validated program for unattended whole-slide length measurements.
It does not directly outline or count individual cells.

WHAT HAPPENS NEXT?
Keep this version fixed and check it on additional roots. Record its prediction
before comparing with Gonzalo's biological judgement. Only after checking T and
the measurement endpoints should we assess lengths in micrometres. 2_12 helped
develop this rule, so it is not an independent final test of this version.

SAVED EVIDENCE
ACCEPTANCE.json records Gonzalo's approval and exact coordinates.
evidence/approved_result.png is the original candidate picture he approved;
its historical captions predate approval. ACCEPTANCE.json is the latest status.
evidence/self_check.json records the successful computational replay.
Independent external reviewer review was unavailable because subscription login was absent;
no review was sent. Local checks and Gonzalo's biological review are recorded.

TECHNICAL COMMANDS — FOR LATER REFERENCE
No command is needed now. The following first command only checks file integrity:

"${ROOTSPAN_PROJECT}/00_Software/ml-env/bin/python" "${ROOTSPAN_PROJECT}/06_Models/Locked_Transition_v1_20261116/verify_checkpoint.py" --require-read-only

To reproduce the approved example, choose a NEW results folder outside this
checkpoint and run:

"${ROOTSPAN_PROJECT}/00_Software/ml-env/bin/python" "${ROOTSPAN_PROJECT}/06_Models/Locked_Transition_v1_20261116/run_checkpoint.py" --self-check --output "/absolute/path/to/a/new/results_folder"

For another root, use the same Python and runner with:
--image "/absolute/path/to/crop.png" --route "/absolute/path/to/route.json" --output "/absolute/path/to/a/new/results_folder"

Route keys: path_xy_crop_pixels and distance_along_path_pixels. Optional
source_crop_xyxy supplies the full-slide offset. The checked path and image must
match, with Endoderm toward the left and hypocotyl toward the right. Whole-slide
root finding and path preparation remain separate checked steps.

SOFTWARE AND PRESERVATION
The existing project ml-env environment runs this copy. environment.json and
requirements.freeze.txt record its software versions. No training or downloads
occur during inference. Other hardware or software may change numerical results;
rerun the self-check after such changes. Previous models and source images remain
unchanged. The earlier segmentation U-Net is a separate saved model.
