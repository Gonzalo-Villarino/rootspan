EXPERT ACCEPTANCE UPDATE — 16 November 2026
Gonzalo explicitly confirms the vertical blue model T as a reasonable left edge of the mixed-cell zone. This supersedes the earlier project tooling interpretation that it is too far left or unaccepted. This is post-prediction acceptance on one root, not an independent numerical accuracy test. Earlier A/B/C appearance labels remain recorded; no annotations, model or rule changed. See BIOLOGICAL_REVIEW_ACCEPTED_20261116.json.

A04 ROOT 3: UNCHANGED APPEARANCE DETECTOR

A and B endoderm C Transition zone. Very bluerry image tho

No new teaching, weights, threshold or decision-rule changes. Predictions saved before comparison.
Only the nearest scan window to each reviewed field is compared; geometric framing differs slightly.
A: expert Endoderm; model nearby Endoderm.
B: expert Endoderm; model nearby Endoderm.
C: expert Mixed / transition; model nearby Endoderm.

Blue = model T candidate. No exact reference T was given, so its location still needs biological review.
Blur is recorded, not proven to cause any discrepancy. No exact T or new length accuracy is established.
This slide has been inspected in earlier development; it is not a fresh blind test. Original PAT exposure unknown.
All original images, labels and model weights unchanged. Word manual not edited.

Interpretation: the blue candidate T is before B, a field the expert calls Endoderm. Do not accept this T for measurement. The rule allows later returns to Endoderm. Changing that rule alone would not establish correct mixed-cell recognition near C. Blur is not a proven cause.

Repeat with the saved original program path and a NEW output folder: run --phase prepare, inspect the route, then --phase predict and --phase compare, passing the same --output NEW_DIRECTORY each time. Code: ${ROOTSPAN_PROJECT}/08_Model_Refinement_20261116/check_frozen_a04_root3.py

CONTEXT UPDATE: A is consistently Endoderm across sampled centres. B is mostly Endoderm with two mixed calls near its right edge. C has three Endoderm-only and three mixed calls. Mixed appearance in a larger expert field does not require every smaller field to contain both cell types. The original single-centre comparison is limited; it does not show that the detector entirely missed mixed cells in C.
Blue model T lies before B, which expert identifies as Endoderm. This appears inconsistent with the agreed one-boundary regional convention; do not accept it for measuring lengths. Exact T reference remains unspecified.
See A04_ROOT3_FIELD_CONTEXT.png and region_context_diagnostic.json. Do not interpret the initial nearest-centre comparison as absence of all mixed predictions in C.

Independent external reviewer review unavailable: subscription login required. No review sent or approval claimed. Final context diagnostic also unreviewed.
