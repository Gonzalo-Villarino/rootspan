CELL APPEARANCE EXPERIMENT — 16 November 2026

What changed
A small three-appearance classifier was fitted using your reviewed clear-Endoderm (A), mixed-cell (B), and clear-Periderm (C) windows from COL0 root2. It uses image features from the frozen published PAT encoder. The existing working U-Net was not changed.

What it did
It scanned 295 fields along the previously reviewed main-root path of COL0 root3. Appearance scores were saved first. A separate ordered rule allows Endoderm, then mixed appearance, then Periderm. Missing states are allowed; an interior T is not forced. Once T is selected at the start of mixed appearance, the regional measurement convention calls the left main-root region Endoderm and the right region Periderm, including mixed. Background and lateral roots must remain separate.

Result
A candidate T was found, but it was 590.43 native pixels (about 1114 micrometres) left of your working reference, outside the approximate interval you reviewed. This would overestimate Periderm length by that amount if the route and right endpoint stayed fixed. The candidate is not approved for automatic measurements.

What this does NOT establish
Only three teaching fields from one root were used. Augmentation produced no independent biological examples. The comparison root is on the same familiar slide; pretraining exposure of the published PAT encoder is unknown. The path was human-assisted. The model may respond to thickness, brightness, branch context or cell appearance; this experiment does not prove explicit cell-shape recognition. Nearby lateral roots can appear in sampling windows even though their centers follow the main root. Do not describe this as a new blind test or a finished measurement model.

What happens next
Add reviewed examples of clear Endoderm near a transition, alongside mixed examples from additional roots. These should help distinguish a thick Endoderm region from genuinely mixed cells. Freeze teaching labels before fitting again, and evaluate on a different acquisition once available. Do not move the human reference simply to make a model pass.

Files
RESULT_TO_REVIEW.png: simple comparison and regional rule illustration.
run_01/ROOT3_APPEARANCE_PROFILE.png: complete raw scores and ordered interpretation.
run_01/TEACHING_INPUTS.png: exact processed teaching inputs.
run_01/appearance_head.pt: learned small classifier weights; requires the unchanged published encoder and this script's preprocessing.
run_01/result.json: settings, limitations and numeric outcome.
verification.json: checks performed.
review/review_status.json: external reviewer review unavailable (login required); no review sent.

Reproduce in a NEW output folder, without overwriting the original run:
"${ROOTSPAN_PROJECT}/00_Software/ml-env/bin/python" "${ROOTSPAN_PROJECT}/08_Model_Refinement_20261116/appearance_experiment.py" --run-name run_02

The script contains all preprocessing, training and scoring steps. Seed is recorded. Small floating-point differences can occur across devices/software. Original run_01 includes the exact executed script; the current script only adds a run-name option for safe reproduction.

Learning note
Finding the mixed zone and enforcing a clean left/right rule are different operations. A neat regional output can still be wrong if T is misplaced. Judge the boundary against biological review, not just the smoothness of the coloured output.

Word manual was not edited in this step. This text and all experiment files are saved for its next update.
