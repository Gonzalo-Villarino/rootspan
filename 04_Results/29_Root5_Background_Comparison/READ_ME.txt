ROOT 5: ADDING BACKGROUND BELOW THE ROOT

Completed 15 November 2026 at Gonzalo's explicit request to run the full sequence.

CHANGE
project tooling drew one additional Background polygon in Labelme below root 5 and saved it. It follows clear space below the root, leaving a small uncertain edge margin. All prior 15 shapes, image and other annotation metadata were unchanged. A backup is in 05_Notes/Checks/Root5_Lower_Background_Demo_20261115T120351Z. No overlap with existing tissue labels was introduced.
The same 1024 x 1024 image patch was used. Only 193,215 mask pixels changed: unlabelled (0) to Background (3). Periderm, Endoderm and ignored conflicts were unchanged.

TRAINING
Same three-category U-Net, unchanged program 17 and model definitions, fresh seed 34, Adam learning rate 0.001, gradient clipping 5, RGB/255, no resizing/augmentation, 600 requested/completed updates. Both trials started fresh. The old trained model was retained.
The new run used the updated pair through --patch-details and separate --output-parent. Program 17's generic root-5-versus-root-1 experiment note describes the earlier lesson; THIS comparison_plan.json/comparison_results.json records the actual baseline and single label change for the present experiment.

RESULT
Green pixels below the Periderm reference outline: 11,454 before; 2,285 after (80.05% fewer). All remaining 2,285 lie in unannotated pixels. Of the original green-band locations, the new model predicts 10,253 Background, 830 Endoderm and 371 Periderm. There are also 1,455 newly green pixels elsewhere in the same lower region; therefore the band shrank and shifted, rather than simply losing a fixed subset of pixels.
On newly labelled Background pixels, the old model predicted 182,102 Background, 10,403 Endoderm and 710 Periderm. The new model predicts all 193,215 as Background.
Using the SAME updated reference for both models: labelled Background agreement 97.9661% before -> 100% after. Tissue matching remained very high but did not improve: Periderm 34,949/34,950 before -> 34,947/34,950 after; Endoderm 14,130/14,130 before -> 14,129/14,130 after. The new model has four mismatches among 595,461 labelled pixels.

LIMITS
This is fitting the same training example, not independent evaluation. Only 56.79% of the patch has confirmed category labels. Unlabelled margins remain ignored, so colour changes there cannot all be scored as biologically right or wrong. The lower green band improved but was not fully eliminated. One fresh run per label set with the same seed/settings is not replicated evidence of broader performance. No extra correction or training was done after inspecting this result.

FIGURES
root5_before_after_closeup.png: microscopy, old prediction, new prediction in the same close-up.
Full comparison: ${ROOTSPAN_PROJECT}/04_Results/28_Root5_Lower_Background_Prediction/view_20261115T121105Z_xwp5xe8f/image_truth_prediction.png

PROGRAMS AND RECORDS
comparison_plan.json: pre-training controlled-change check.
comparison_results.json: numeric comparison and exact input/output locations.
analyze_comparison.py and draw_comparison.R: reproducible analysis and scientific figure.
New mask: 04_Results/25_Root5_Lower_Background_Mask
New patch: 04_Results/26_Root5_Lower_Background_Patch
New trained model: ${ROOTSPAN_PROJECT}/04_Results/27_Root5_Lower_Background_Training/run_20261115T121005Z_d_v61wde
New prediction: ${ROOTSPAN_PROJECT}/04_Results/28_Root5_Lower_Background_Prediction/view_20261115T121105Z_xwp5xe8f

Word manual has not been edited in this step; detailed learning notes and figures are saved for integration.
