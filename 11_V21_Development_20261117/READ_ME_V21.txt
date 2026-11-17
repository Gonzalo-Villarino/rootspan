SUPERSEDING STATUS — 17 November2026: V2.1 was withdrawn after review identified unreliable endpoint estimates. Seven filled rows did NOT establish seven correct periderm measurements. The new V3 candidate failed its improvement check and was not released. V1/V2 remain unchanged. See12_V3_Development_20261117/REVIEW_AND_RESULTS.txt. Historical notes follow.

17 November 2026 — V2.1: tracing and endpoint candidates

WHAT CHANGED
1. Touching roots: the program follows their identities and avoids crossing bright
   tissue when separating neighbouring root tracks. No expected root count is supplied.
2. Right stopping point: a sustained decrease in fluorescence proposes an endpoint.
   This is a simple image rule, not a trained root–hypocotyl junction classifier.
   If unsupported, the earlier widening estimate or uncertainty flag remains.
3. Short transition: only when the locked V1 rule finds no T, two mixed fields
   immediately between at least three Endoderm-only and three Periderm-only fields
   can supply a provisional T. These overlapping fields are not independent samples.
   Such a result carries an additional review flag. No mixture is invented.
4. The saved neural weights are unchanged. No training occurred. Blue vertical lines
   still mark T; orange dots mark automatic endpoint candidates.

WHAT THIS FIXES — AND WHAT IT DOES NOT
On 7_18, V2.1 produces seven provisional lengths, compared with two in V2.0.
Root 4 has only brief mixed-appearance support and needs particular review.
This fixes processing coverage on this slide, not biological accuracy.
V1 and V2.0 remain available and unchanged. Results from earlier runs are preserved.

Five development slides were checked with identical code and 0.5299 pixels/um:
  slide       tracks     V2.0 lengths     V2.1 lengths
  7_18          7             2                7
  1_7           7             4                5
  22_22         7             5                7
  Col_rep1      6             6                6
  WOX4_rep1     6             5                5
No supported T / endpoint conflict still prevents two lengths on 1_7.
The cut-off WOX4 fragment still has no length. Reserved slide 15_27 was not used.
These five slides have been inspected during development: they are not a blind test.
Historical exposure of the PAT encoder to these archived slides is unknown.

YOUR ROOT 2 CORRECTION
You corrected the earlier 'cut end' answer: the right structure is hypocotyl.
You confirmed D as the root 2 stopping point on 7_18. D is saved as an expert
reference, not inserted into automatic inference or used as a hardcoded endpoint.
The V2.1 automatic endpoint is approximately 460 um left of D (point-to-point
separation). See root2_automatic_vs_D.png. This discrepancy remains unresolved.

CHECK AGAINST YOUR FIJI REFERENCES ON 1_7
Point-to-point endpoint differences from your manual ROI, in um:
             V2.0       V2.1
  root 1      231         216
  root 2      595         140
  root 6      693          33
These endpoint changes are closer. Full periderm lengths are still substantially
and inconsistently different because T and the curved route also determine length:
             manual um    V2.1 um     difference
  root 1       8002          5624       -29.7%
  root 2      13711         15588       +13.7%
  root 6      17936         24590       +37.1%
In particular, closer endpoints did NOT improve every total-length comparison.
Do not use these experimental estimates as validated biological measurements.
There are only three manual references, not a sufficient accuracy study.

HOW TO USE
1. Open Periderm Lab.command in 09_Periderm_GUI, or http://127.0.0.1:8765/.
2. The ordinary page starts empty. Select V2.1, choose a local image and confirm scale.
3. Click Analyse image. It runs automatically without Labelme annotations.
4. Inspect the blue T, orange stopping point, main-root route and any flags.
5. Export CSV. Blank values mean no supported measurement, not zero periderm.

WHAT WE LEARNED
A tissue detector and a measuring pipeline are different pieces. Fixing root
tracking or choosing an endpoint can make execution more complete without making
the tissue detector more accurate. A numerical result is not an accuracy score.
The next scientific improvement requires independent T and junction references
on several roots. Those references should diagnose failures before any new model
training, and separate untouched images should later measure final accuracy.

REPRODUCTION AND RECORDS
verification.json: five-slide comparison, exact coordinates, manual errors, checks.
root2_junction_confirmed_D.json: your latest anatomical correction and D reference.
trial_v21_7_18/: automatic seven-root result, numeric CSV, routes, scores and audits.
regression/: other four slides, each with its own full record.
14 unit checks passed, including original V1/V2 checks, abstention, no invented
endpoint on a uniform field or image edge, touching-root separation and replay
of approved locked profiles. These are computational checks, not accuracy proof.
Locked V1: 40 files verified unchanged and read-only. V2.0 fingerprints match.
Word manual has not been edited during this update. This note records the work.
All inputs remain local; the SALK disk is not needed for these copied images.
