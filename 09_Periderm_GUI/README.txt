APP UPDATE 0.6.0 — Per-root mask estimates
Open http://127.0.0.1:8765/ in Chrome. The default V2 + PAT mode now reports cleaned-mask periderm estimates in the main table and periderm_estimates.csv without requiring confirmed transition/junction landmarks. Explicit expert corrections are labelled. This is a reporting/measurement change, not model retraining. Dark/blurred or untraceable roots can still be withheld with reasons. Fresh7_18 upload verified seven values; Col_rep1 six. See16_Per_Root_Estimates_20261117/FINAL_STATUS.txt and browser_run_verification.json.

APP UPDATE 17 November 2026: version 0.5.0
V2 measurements are unchanged. New V2 analyses can include a separate PAT cleanup comparison with per-root label statistics, overlays and CSV. These are not anatomical periderm measurements. The original cleanup was tested against the three saved Fiji ROIs and did not consistently improve accuracy. No corrected V3 model is released. See ../15_All_Root_Data_20261117/RESULTS_AND_NEXT_STEP.txt.

SUPERSEDING STATUS — 17 November2026: V2.1 was withdrawn after review identified unreliable endpoint estimates. Seven filled rows did NOT establish seven correct periderm measurements. The new V3 candidate failed its improvement check and was not released. V1/V2 remain unchanged. See12_V3_Development_20261117/REVIEW_AND_RESULTS.txt. Historical notes follow.

17 November 2026 — CURRENT LOCAL GUI: V2.1
Open empty upload page: http://127.0.0.1:8765/
Saved 7_18 result: http://127.0.0.1:8765/?run=20261117T125305Z_8bbe6d9a
The app now returns seven provisional periderm estimates for seven 7_18 tracks.
Root 4 carries a brief-transition flag. Source images and locked V1 unchanged;
V2.0 preserved. No neural training. Five-slide execution and 14 tests passed.
Browser displayed all seven rows, CSV matched standalone, Start a new image
returned to an empty upload page. V2.1 default; V2.0 and V1 remain selectable.

Do not equate completion with biological accuracy: 1_7 lengths still differ
from the three manual references by about -30%, +14%, +37%. The new endpoint
heuristic is closer to their stopping points but does not fix the model's T.
On 7_18 root2 the user corrected 'cut end' to hypocotyl, then confirmed D.
D is saved as a separate expert reference, not supplied to automatic inference.
The automatic root2 endpoint remains about 460 um left of D.

Detailed work and next scientific step:
11_V21_Development_20261117/READ_ME_V21.txt
11_V21_Development_20261117/verification.json
11_V21_Development_20261117/root2_automatic_vs_D.png
Model/pipeline snapshot: 06_Models/Periderm_Pipeline_V21_20261117/
The Word manual was not edited. Next: validate T and junction independently on
several roots; address observed disagreement before treating estimates as
biological results. No further annotation work has been assigned to the user.

Earlier version notes (historical):

17 November 2026 — V2 PIPELINE SAVED AND GUI VERIFIED
User requested fixing missing 7_18 estimates and explicitly requested a new V2 for changes.
V1 locked weights/rule and original pipeline.py remain unchanged. V2 is a separate inference
pipeline update, NOT a newly trained neural network.

Fixed: the inspection route was truncated at the estimated measuring endpoint, hiding
right-hand Periderm context. V2 inspects the full visible track and measures separately
to the original endpoint. Same preprocessing, scores, rule, root detection and quality guard.
If T is beyond the estimated endpoint, report a conflict and withhold length.

7_18: all seven candidates processed. Roots6/7 provisional lengths 8463/3162 um.
Roots2/3 remain T-to-visible-end segments only (27715/3849 um), not periderm lengths.
Roots1/5: T/endpoint conflicts; root4: no supported T. Do not invent zeros/numbers.
V1 -> V2 provisional counts: 1_7 3->4;22_22 5->5;7_18 1->2;Col_rep1 6->6;WOX4 3->5.
All previously returned T positions and measuring endpoints unchanged. Coverage improvement
is not biological accuracy validation; prior manual discrepancies remain relevant.

GUI http://127.0.0.1:8765/ starts empty; selector defaults V2 and offers original V1.
Latest V2 result: http://127.0.0.1:8765/?run=20261117T121029Z_4ddd8069
Version files: 06_Models/Periderm_Pipeline_V2_20261117.
Evidence: 10_V2_Development_20261117 (comparison.json, verification.json, READ_ME_V2.txt,
GUI screenshots, diagnostic PAT comparison, regression runs, review attempt).
9 unit tests passed; GUI API run exactly reproduced standalone V2; 7-row CSV download
matched saved bytes; V1/V2 selector and Start a new image tested; locked40files verified.
No new training; reserved15_27 not used. Original PAT decoder checked diagnostically only.
external reviewer review unavailable: subscription login required; no review sent.
Word manual NOT edited. Learning notes saved here and READ_ME_V2.txt.
SALK files already local; no open SALK files when checked. User told to eject then unplug.

NEXT: review recovered root7 blueT and orange endpoint, then diagnose root1/5 endpoint
conflicts against anatomy. Do not call V2 a validated autonomous model or force all7values.
Do not tune on reserved15_27. No need to redo all annotations to understand this change.

EARLIER METHOD DETAILS (V1 retained):

PERIDERM LAB — LOCAL GUI 0.2.1
17 November 2026

OPEN AND USE
1. Double-click Periderm Lab.command. Keep its launcher window open while analysing.
2. At http://127.0.0.1:8765, click Choose File and select your microscopy image.
3. Check the pixels-per-micrometre scale; the copied slides checked in this session
   were confirmed by Gonzalo to use 0.5299 pixels per micrometre.
4. Click Analyse image. The program finds root candidates and attempts each separately.
5. Inspect the source image, blue T candidates and orange measuring endpoints.
6. Export CSV to save the measurements AND flags. Blank measurements are not zero.
7. Start a new image clears the screen without deleting saved results.

The normal opening page is empty. An explicit saved-result link with ?run= opens
only that saved run. Reloading such a link keeps that result; Start a new image
returns to the clean upload page. No project images or history are preloaded.

EXECUTING, NOT TRAINING
This uses the same locked Transition V1. Its weights and transition rule are unchanged.
The GUI's root-separation and image-quality checks are separate processing code.
V1 is the saved PAT image encoder plus presence head and a fixed transition rule;
it is not the original small U-Net practice model.
No human polygons, root paths, manual T or Fiji measurements are required to run.
This launcher depends on this project's ml-env; it is not a standalone installer.

WHAT CHANGED
Large black spaces enclosed between touching roots are no longer filled as root tissue.
Persistent parallel roots are separated using image-derived background boundaries.
Roots are listed in automatic top-to-bottom order. No expected count is hard-coded.
A short root-like fragment touching the image edge is retained with a partial-root flag.
A preliminary low-detail check can withhold a measurement on a possible blurry image.

RESULT FLAGS
Provisional: T and an automatic widening-based right endpoint candidate were found.
This is not proof that the anatomical root-hypocotyl junction or T is correct.
No supported transition: V1 could not place T under its fixed rule. No length.
Endpoint uncertain: only the T-to-visible-end segment is given; periderm length is blank.
Possible blur / low detail: the preliminary focus/contrast check failed. No length or
segment measurement is reported. A plotted blue line is only the model's candidate T.
Partial root / cut off: an edge fragment lacks full context. No length.
Connected roots / Could not analyse: unresolved geometry or error. No length.
A row stays visible even if it has no numerical measurement. A candidate count is
not a verified biological root count. Some faint, crossed or fragmented roots can
still be missed, and more complex layouts may need improvements.

BLUR CHECK — LIMITS
This is not a trained blur classifier. At native resolution, 320x96-pixel windows
along the root compare fine detail after smoothing at sigma 1 versus sigma 3,
normalised by signal variance. A detail ratio below 0.001 or 95th-minus-5th-percentile
contrast below 4/255 flags a window. Measurements are withheld when at least 60%
of the root windows, or at least 60% of three or more windows near T, are flagged.
The threshold is a conservative severe-detail-loss screen. Controlled artificial
blur was detected; expert-labelled real blurry/clear images have not validated it.
Passing this check does not prove the image or model output is good. Noise, staining,
cell size and magnification can affect it. Blurry and no supported T are distinct.
The archived PAT code inspected here used a human quality-selection GUI; this new
automatic screen is not claimed to reproduce an original trained PAT blur detector.

INPUTS AND MEASUREMENTS
Single-plane RGB/RGBA fluorescence TIFF/PNG/JPEG, up to 200 million pixels and 1 GB.
Large roughly horizontal roots with hypocotyl to the right, at V1's native imaging scale.
Length is the curved path from model T to the estimated right endpoint, divided by
pixels per micrometre. Calibration does not correct a change in microscope resolution.
The root-hypocotyl endpoint is still an anatomical estimate from widening, not a
validated biological detector. This is a learning prototype, not a validated PAT release.

WHERE THINGS ARE SAVED
runs/: each execution, its images, per-root predictions, routes, quality records,
measurements.csv and original-image/model fingerprints. Earlier runs remain unchanged.
uploads/: local image copies. No cloud upload. Originals are preserved.
checks/multiroot_20261117/: repeated execution checks, method/test records and review.
../02_Data/16_New_SALK_GUI_Checks_20261117/: verified SALK copies of 7_18, Col_rep1,
WOX4_rep1, plus previews, source paths and checksums. About 1.3 GB copied.

These are execution/coverage checks, not an accuracy validation. The previous 1_7
manual comparison's disagreements remain on record. Historical exposure of these
archive images to the original PAT encoder is unknown. 15_27 remains reserved.
