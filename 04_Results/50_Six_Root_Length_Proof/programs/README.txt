PERIDERM LENGTH PROOF OF CONCEPT — 15 NOVEMBER 2026

Start with the six-root review:
../04_Results/50_Six_Root_Length_Proof/OPEN_ME_REVIEW_AND_MEASURE.html
Word walkthrough:
../05_Notes/06_Periderm_Length_Proof_Walkthrough.docx

STATUS
The measurement and review workflow works. A repeat run reproduced the same raw
prediction mask, scores and six lengths exactly. This does NOT establish biological
accuracy. The model remains exploratory and needs transition AND basal-end review.
No new model was trained for this measurement proof; no replacement model was accepted.

WHAT EACH RESULT MEANS
1. Automatic: preserved three-category UNet++/ResNet34 transfer model predicts from
   RGB image only. No Labelme file is read by predict_slide.py or measure_roots.py.
2. Annotation-assisted: build_proof.py calculates a centreline inside each of the six
   Periderm polygons that Gonzalo already drew. These are NOT model predictions.
3. Independent manual: currently blank. Gonzalo can trace in the review page or Fiji.
   Browser traces remain in memory until individually downloaded as CSV and JSON.

DEMO INPUTS AND MODEL
Slide: MT25_rep3_practice.png,15423x9543 original pixels.
Annotations: MT25_rep3_root5_refined_GV.json contains all six roots, including revised
roots4 and5; other roots retain earlier practice outlines. A compact snapshot with
imageData removed is saved in folder50; it points to the unchanged original PNG.
Working model:04_Results/39_PAT_Transfer_Learning/run_20261115T140159Z/model_weights.pt
Model SHA256:234b7de1969d80dd1523ce52afc89d5952b5e9ba4a63cdd7b3c8bff5b1c13633
This slide has been used for development. No independent performance claim is valid.

METHOD
Full slide: native1024 tiles,64-pixel context on each side,896-pixel cores. No resize.
Root geometry: classesPeriderm/Endoderm,quarter-resolution topology,4-iteration
opening to separate thin connections,2-iteration closing; long components sorted
from top to bottom. Skeleton path with8-neighbour Euclidean weights; endpoints
extended to component edge;51-point quadratic smoothing; restored native coordinates.
Automatic tissue boundary: mean core P/(P+E) scores sampled every8pixels, smoothed
24pixels; close only enclosed gaps<=150pixels; keep basalmost Periderm interval if
it comes within250pixels of the automatic right endpoint. This endpoint is a
candidate, not confirmed hypocotyl anatomy. Branches/artifacts can affect it.
The rule is inspired by PAT, not a byte-identical reproduction of PAT.
Lengths add Euclidean distances along the saved path, not a straight chord.

CALIBRATION
20_29 and2_12: user confirmed0.5299pixels/um on15Nov2026. Length_um=length_px/.5299.
MT25: not separately confirmed; final six-root table remains in pixels.
Cropping and display zoom preserve calibration. Image resizing changes it.
TIFF72DPI is display metadata and is NOT microscopy calibration.

REPRODUCE THE SAVED DEMO
From the project folder, run:
00_Software/ml-env/bin/python 07_Periderm_Length_Proof/reproduce_saved_demo.py
A new folder is created under04_Results/51_Reproduction_Check. Previous files stay.
The annotation snapshot and image hashes are checked. Model weights are frozen.

NEW SLIDE EXPLORATORY USE
00_Software/ml-env/bin/python 07_Periderm_Length_Proof/run_new_slide.py --image "/absolute/path/image.png" --output "/absolute/path/new_result_folder" --expected-roots 6 --basal-side right
Add --pixels-per-um 0.5299 only when that calibration is confirmed for the specific
original image. Use a space between --pixels-per-um and the number.
The command assumes horizontal primary roots, basal end on the right. It produces
provisional output even if root count or anatomy is wrong: inspect QC flags.
Do not use predictions as experimental measurements without manual comparison.

VALIDATION
07_Periderm_Length_Proof/test_geometry.py checks known straight/curved geometry,
diagonal distance, unit conversion, invalid calibration, gap handling, six-root
records and saved path/length agreement. Browser tests exercised tracing, undo,
calibration and both downloads; synthetic test exports are labelled and confined
to05_Notes/Reviews/20261115_PAT_Length_Proof, not the user comparison table.
Original source images, annotation files and model weights were preserved.

EXPLORATORY COMPARISONS
Folder49 retains the original and revised measurement attempts, the released
four-class PAT model comparison, and20_29 image-only inference.20_29 has seven
visible primary roots. Its saved numbers are provisional, not validated results.
A one-change E-left/P-right rule did not solve the transition problem and was not
selected for the final comparison. The published4-class model also had tracing/
transition failures. Rejected attempts were not hidden or substituted for evidence.

REFERENCES AND PROVENANCE
PAT paper:https://pmc.ncbi.nlm.nih.gov/articles/PMC10981931/
Official code:https://github.com/Salk-Harnessing-Plants-Initiative/PAT-Pipeline-for-Analysis-of-Periderm
Pinned clone:00_Software/PAT_Pinned_20261115,commit51cdf8966f6da17b711aeb5625205a8a2ca39467.
Exact inputs, paths, hashes, native-coordinate polylines and annotations used are in
folder50/proof_record.json. Runtime checks and reproduction verification are in
05_Notes/Reviews/20261115_PAT_Length_Proof.

AUTHORSHIP
The new measurement and review scripts were written in this project with project tooling, using Python, NumPy, SciPy, scikit-image, Pillow and PyTorch/SMP. This is a documented adaptation, not code written by Ling or a new validated PAT release. The original PAT paper and repository are credited above. Independent external reviewer review could not run because subscription login was required; no review was sent. Local checks are recorded separately.
