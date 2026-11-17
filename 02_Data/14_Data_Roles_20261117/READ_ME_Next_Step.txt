NEXT EXPERIMENT: MANUAL MEASUREMENT VERSUS LOCKED V1
17 November 2026

We will compare six roots while keeping v1 unchanged. Start with one root in Fiji.
This is a small assisted measurement pilot, not a claim that the model is ready for hundreds of slides.

WHAT YOU WILL DO
1. Open the original 1_7.tif in Fiji.
2. Set the image scale: 0.5299 pixels = 1.00 micrometre, aspect ratio1.0. Global unchecked for this image.
   This is equivalent to0.5299 pixels per micrometre, confirmed by you for1_7.
3. Zoom in and choose your T: left edge of the mixed-cell zone on the main root.
4. Choose the root-hypocotyl junction J on the right, using the same definition as before.
5. With Segmented Line, click along the centre of the curved main root from J to T, following the published PAT tracing direction.
   Double-click to finish. This is a length trace, not a closed polygon or tissue-area annotation.
6. Add the trace to ROI Manager, give it a root-specific name, measure it, and save both
   the ROI and Results table in the manual-measurement folder. Keep the original TIFF unchanged.
We will perform these steps together, one at a time. None of these manual measurements has occurred yet.

WHAT CODEX WILL DO AFTER YOUR MANUAL REFERENCES ARE SAVED
Check the main-root geometry and shared right endpoint. Run lockedv1 on the image and route
without giving it your T or periderm length. Compare the model length and your length.
Show all six outcomes, including no-T or failed cases. Never move the model T to make it agree.
A shared expert endpoint makes this assisted measurement, not fully automatic endpoint finding.

DATA ROLES
Teaching candidates: COL0_Slide1, MT25_rep3,20#MT28_gfp,25717_gfp.
Familiar development checks: A04_Slide2,20_29,2_12.
Current v1 pilot candidate:1_7. Six exact root identities still need source-only selection;
the overview contains overlaps, so no claim of six suitable complete roots yet.
Reserved local evaluation:15_27,22_22.22_22 has had an overview-only suitability inspection.
These are not claimed unseen by the original published PAT encoder; its training membership is unresolved.
All roots and patches of one slide stay in the same data role.

SAVED THIS SESSION
Three original TIFF copies verified against SALK using SHA256:1_7,15_27,22_22.
Total1,304,661,261 bytes (about1.3GB). SALK safely ejected successfully.
Lockedv1:40 read-only files verified unchanged. No prediction, training, or v2 created.
Word manual has not been changed. New record: data_roles_and_pilot_plan.json.

Fiji documentation:
https://imagej.net/software/fiji/downloads
https://imagej.net/ij/docs/tools.html
https://imagej.net/ij/docs/menus/analyze.html

Published Fiji method verified: see PAT_Fiji_Method_Verified.txt. PAT used Global; our single-image exercise leaves it unchecked.
