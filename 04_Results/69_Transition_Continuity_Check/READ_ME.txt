TRANSITION CONTINUITY CHECK — 16 November 2026

Why the earlier line was too early
The frozen appearance model predicted both appearances in a short three-window burst, then38 consecutive Endoderm-only windows, then a later sustained mixed/Periderm sequence. The old decoder selected the first eligible burst because clear Periderm existed somewhere later. It did not treat the long intervening Endoderm-only stretch as contrary evidence. The raw classifier may still make appearance mistakes; this experiment does not establish their cause or correct the learned features.

What changed
The learned weights, raw image scores,0.5threshold, input size and original mixed-candidate/support requirements stayed fixed. A new decoder guard rejects an early candidate if an intervening Endoderm-only run has at least3 centres separated by320native path pixels, one input-window width, before its supporting Periderm run. This preserves the biological requirement for both predicted appearances; it never calls pure Endoderm mixed to force T. It is a testable heuristic, not a universal biological law.

Result
2_12candidate moved from source[5686.130303161157, 272.61427150132204] to[8414.342292421163, 411.8250414267261] without manually moving a line or fitting the expert coordinate. The original burst was contradicted by8spaced Endoderm fields; the later candidate had no such contradiction. The new line is farther right than B and awaits biological acceptance. No exact expert T was supplied, so no numerical T accuracy or root-length result is claimed.

Checks
Synthetic cases confirm that no T is forced without both appearances and required E/P support, long E returns reject early candidates, and short E returns may occur within mixed tissue. Original decoder replay matched all4saved profiles exactly. Accepted COL0root3 and A04root3 T coordinates were unchanged.20_29 still returned no T. Weights, source predictions and scores were preserved.

Evaluation status
2_12informed this rule change after a failed check; it now contributes to method development, despite no neural training on it. It cannot serve as independent validation of this updated rule. An unused slide is needed for a stronger later test. This version is a candidate; the previous detector/default U-Net is not overwritten.

Saved
experiment_plan.json; transition_continuity_rule.py; check_transition_continuity.py; comparison_results.json; four per-case decoder records; verification.json;212_CONTINUITY_CANDIDATE.png. Run the program from08_Model_Refinement_20261116using ml-env. To reproduce preserve this directory and give a new experiment plan/output location in a copy; the program refuses to overwrite its result.

Word manual not edited. Notes retained for its next update.

Independent external reviewer review unavailable because subscription login is required. No review sent and no external reviewer approval claimed. Local checks are complete; biological review of the new2_12T remains pending.
