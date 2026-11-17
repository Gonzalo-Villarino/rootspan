2_12 ROOT1: FIXED APPEARANCE DETECTOR

Expert reference before inference: the square i see in B seems more endoderm but if u move a bit to the right of B i see transtion zone

Blue is the model T candidate (left of B rectangle). Its biological position needs your review.
B window counts: {"Endoderm_only": 6, "both_mixed": 0, "Periderm_only": 0, "neither": 0}. These windows overlap; they are not accuracy percentages. No A/C truth labels or exact T coordinates were invented.

The upper-root route was derived from source fluorescence with the same geometry settings and inspected before inference. The exact frozen A04 inference routine was reused, with the same head/encoder, input window size, normalization, thresholds, stride and run rule. Predictions saved before comparison. No training, tuning, normalization fitting or manual displacement.

2_12 remains outside training/tuning. Earlier U-Net predictions on this slide were viewed, so this is not a pristine blind test; original published PAT image exposure unknown. No complete root or length validation.

Reproduce into a NEW --output directory using this program: --phase prepare; inspect route and record passed route_visual_check.json; --phase predict; --phase compare.

Word manual not edited. All original sources and models unchanged.

Exact source close-up for expert review: 212_MODEL_T_CELL_REVIEW.png. Both panels show1000x420native source pixels at identical display scale. Model T is left of B; sampled windows in B are all Endoderm-only. No A/C labels inferred. Independent external reviewer review unavailable due login; no review sent.
