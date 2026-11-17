REQUIRING BOTH CELL APPEARANCES
16 November 2026

The idea in simple words
A long, orderly rectangular cell appearance by itself should remain Endoderm. A mixed-zone call needs evidence of shorter, irregular Periderm-like cells as well. We tested this with two separate predictions for each small main-root image window: Endoderm-like appearance present, and Periderm-like appearance present.

Step 1 — Check the error
The old appearance classifier itself called the new blue-line area mixed. The later boundary rule was not creating that local error. Removing off-root upper pixels, including a bright fragment, did not fix it. This does not prove what visual feature caused the error.

Step 2 — Save the biological correction
Gonzalo identified this main-root area as Endoderm. We saved the actual path-aligned scan window as teaching example E. We did not automatically label the whole region up to the reference T. There are now five teaching windows: three clear Endoderm, one mixed, and one clear Periderm. The appearance categories keep equal total training weight.

Step 3 — Compare two changes separately
We kept the published image-feature extractor fixed and compared the previous three-way classifier with two separate presence predictions. The latter has teaching answers E=[1,0], mixed=[1,1], and P=[0,1]. A mixed call requires both predicted scores to reach 0.5 for at least three consecutive overlapping windows, with a clear-Endoderm run before and a clear-Periderm run after. These settings were recorded before seeing the result. Missing evidence does not force an internal T.

Step 4 — Compare with the existing reference
Previous three-way classifier, four windows: T about 873 micrometres left of the working reference.
Two presence predictions, four windows: the same T.
Previous three-way classifier, five windows: T about 148 micrometres left of the reference.
Two presence predictions, five windows: the same improved T.

What this means
The extra biological teaching example made the difference in this comparison. Changing the output design alone did not improve T. Both versions with the new example place T inside the approximate interval Gonzalo previously reviewed. The former error window now supports Endoderm and no longer passes the Periderm-presence threshold. We did not move the line by a chosen distance or move the human reference to make the result look better.

What the model has NOT established
These are appearance predictions for whole windows. The model does not outline, count or explicitly measure individual cells. A score above 0.5 is a decision rule, not proof that a particular cell exists. Only five teaching windows from two roots on one slide were used; augmented copies are not independent biological examples. Root 3 now contributes teaching data, and its main-root path was supplied. This is a correction check, not an independent test or a finished automatic length method. The 320-pixel-wide windows and 64-pixel spacing also limit boundary precision.

Next step
Review the new blue line biologically, then check the unchanged method on separate images. Mixed tissue remains included in regional Periderm length. The main-root region left of T is Endoderm. Lateral roots and background must remain separate.

What is saved
NEW_MIXED_ZONE_RESULT.png — the simple before/new/reference comparison.
CURRENT_WINDOW_VERSUS_MIXED_EXAMPLE.png — wrong window beside confirmed mixed example B.
diagnosis.json and context_probe.json — checks of the earlier error.
presence_experiment_plan.json — design recorded before fitting.
comparison_results.json and result_summary.json — numeric results and limits.
three_way_five_windows/ — previous classifier design with added example E.
two_presence_four_windows/ — output-design change without example E.
two_presence_five_windows/ — both-output version with example E, weights, code and predictions.
verification.json — rule tests, source hashes and comparison checks.

Reproduction
The exact executed scripts are copied into each run. The current script adds output-path arguments for safe repetition; it does not change the learning or prediction calculations. The following command reuses the saved five-window image features and writes a new repeat folder. It refuses to replace existing run folders:

"${ROOTSPAN_PROJECT}/00_Software/ml-env/bin/python" "${ROOTSPAN_PROJECT}/08_Model_Refinement_20261116/test_both_appearances.py" --output-parent "${ROOTSPAN_PROJECT}/04_Results/65_Mixed_Appearance_Evidence_Check/repeat_01"

Original images, annotations, existing U-Net and earlier models are retained. The Word manual was not edited; this explanation is saved for its next update.

Independent review could not run because external reviewer subscription login is required. No review was sent or approval claimed. See review/review_status.json.

Biological review completed: Gonzalo answered Yes when asked whether the new blue line is a reasonable left edge of the mixed zone. Acceptance applies to this displayed teaching root only. The original reference remains unchanged. A frozen research copy is saved in06_Models/20261116_Two_Appearance_Candidate; it is not the default model.
