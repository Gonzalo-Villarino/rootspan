from pathlib import Path
import json,hashlib,os
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
audit=json.loads((OUT/'inventory_and_annotation_audit.json').read_text());by={x['slide']:x for x in audit['source_images']}
roles=[]
for name in ['20#MT28_gfp','25717_gfp','MT25_rep3']:
 r=by[name];gv=name=='MT25_rep3';roles.append({'slide_id':name,'role':'teaching','source_image':r['image'],'image_sha256':r['image_sha256'],'decoded_pixels_sha256':r['decoded_pixels_sha256'],'annotation_source':audit['GV_current_annotation_summary']['path'] if gv else r['annotations'],'annotation_sha256':audit['GV_current_annotation_summary']['sha256'] if gv else r['annotation_sha256'],'included_root_ids':[4,5] if gv else [1,2,3,4,5,6],'root_numbering':'Primary roots ordered from top to bottom in source image; pairing confirmed by their vertical locations.','pending_root_ids':[1,2,3,6] if gv else [],'readiness':'candidate teaching labels; targeted biological review and background/LR policy required before mask generation','why':('Roots4 and5 were deliberately refined by Gonzalo. Hold other practice-root labels pending agreement with archived boundaries.' if gv else 'Six archived Periderm/Endoderm pairs and lateral-root polygons; reusable annotated image with additional cell appearance/brightness. Already used in local training, so not an independent evaluation source.'),'historical_exposure':'Used in local training/development; public PAT pretraining image membership unresolved.'})
roles.append({'slide_id':'20_29','role':'development_check','source_image':by['20_29']['image'],'image_sha256':by['20_29']['image_sha256'],'decoded_pixels_sha256':by['20_29']['decoded_pixels_sha256'],'reference_status':'No full-slide human Labelme JSON found. Two assistant sparse training masks are not certified evaluation ground truth.','readiness':'needs reviewed reference boundaries and basal endpoints; first review region is root4 box[5440,4256,6464,5280]','why':'Different slide with variable clarity, plus a user-identified clearer transition. Already used for diagnosis and rejected adaptation training; suitable as a familiar development challenge, never a pristine final test.','historical_exposure':'Gradient updates on assistant20_29 masks occurred in rejected runs45/46. Baseline39 did not use20_29 in local gradient training.','next_run_rule':'No20_29 pixels or labels in next training loss; do not initialize from45/46 or their descendants if reporting this as a development holdout. Existing39 baseline or an audited pre-20_29 checkpoint only.','first_reference_review_box':[5440,4256,6464,5280]})
roles.append({'slide_id':'2_12','role':'reserved_test_candidate','source_image':by['2_12']['image'],'image_sha256':by['2_12']['image_sha256'],'decoded_pixels_sha256':by['2_12']['decoded_pixels_sha256'],'reference_status':'No human tissue reference or manual lengths found.','readiness':'not yet scoreable; requires independently prepared human reference and original PAT provenance check','why':'No use in our local gradient training was found. Different slide with visible cells and realistic branches/quality variation. Reserve entire slide; do not cherry-pick fields based on model success.','historical_exposure':'Six image-only predictions were generated after checkpoint selection on15Nov; records state no tuning from them. Source image and predictions have been viewed. Original released PAT model membership unknown.','blind_final_test':False,'next_run_rule':'No gradients, tuning, normalization fitting, threshold selection, model choice or rule selection from this slide. Fix evaluation protocol and freeze selected model before scoring. Report as provisional same-workspace holdout, not a new blind independent test.','visible_quality_note':'Overview includes a partial primary-root fragment at bottom and some branch overlap. Inventory root completeness before looking at future predictions; report exclusions with reasons.','stronger_future_test':'A separate previously unused acquisition/slide set is still needed for a strong independent final performance claim.'})
plan={'created_local_date':'2026-11-16','timezone':'Europe/Zurich','status':'proposed_sets_selected; annotations/reference preparation pending; no_training_authorized_by_this_file','training_run':False,'inference_run':False,'source_files_changed':False,'grouping_unit':'Original biological slide/source image, not PNG versus TIFF, root crops, patches or augmentations. If shared specimen/acquisition identities are later found, group those too.','unique_slides':5,'candidate_teaching_roots':14,'candidate_teaching_slides':3,'roles':roles,'label_policy':{'do_not_merge_disagreeing_polygons':True,'MT25_dispute_roots':[1,2,3,6],'MT25_position_differences_file':'inventory_and_annotation_audit.json','archived_annotator':'Not established; do not claim Ling drew all archived polygons.','unlabelled':'Unknown/ignored, not automatically background.','background':'Archived sources have no explicit Background label. User practice JSON has6 Background shapes; these do not establish full-slide background truth. Review or deliberately define background supervision before training.','lateral_roots':'Archived LR polygons retained as evidence. Do not silently relabel LR as Background or primary Periderm/Endoderm. Decide explicit handling before training; current3-class model has no LR output.','transition_definition':'Agree on a consistent biological start of Periderm and basal/hypocotyl endpoint before length training/evaluation; do not infer definitive class from brightness alone.'},'pre_training_gates':['Agree on target anatomical boundary and endpoint convention.','Biologically review reusable teaching examples; resolve only necessary disagreements instead of redrawing everything.','Create reviewed20_29 development reference; assistant sparse training masks are not an independent reference.','Prepare2_12 test references separately and freeze them before scoring; record all exclusions and prior exposure.','Validate model lineage and source-level split before generating new patches; no overlapping or augmented aliases across roles.','Set length-error evaluation and failure-reporting rules before final test; pixel overlap alone is insufficient.'],'initialization_constraint':'If20_29 is development-only for next experiment, do not continue20_29-adapted candidates45/46. Public PAT ancestry remains a limitation until verified.','sources':['https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data','https://pmc.ncbi.nlm.nih.gov/articles/PMC10981931/'],'local_evidence':['02_Data/01_PAT_Examples/SOURCE_MANIFEST.json','04_Results/35_Multiple_Slides_Data/manifest.json','04_Results/39_PAT_Transfer_Learning/run_20261115T140159Z/experiment_plan.json','04_Results/45_Adaptation_With_Replay/run_20261115T161633Z/experiment_plan.json','04_Results/46_Adaptation_With_Retention/run_20261115T161951Z/experiment_plan.json','05_Notes/Reviews/20261115_Model_Repair/untuned_check_plan.json','03_Notebooks/34_compare_adaptation.py']}
# An explicit whole-source group check prevents TIFF/PNG/crop duplicates in different sets.
seen={}
for r in roles:
 k=r['decoded_pixels_sha256'];assert k not in seen;seen[k]=r['role']
assert len(seen)==5 and sum(len(r['included_root_ids']) for r in roles if r['role']=='teaching')==14
(OUT/'split_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
for dirname,role in [('01_Teaching_References','teaching'),('02_Development_Reference','development_check'),('03_Reserved_Test_Reference','reserved_test_candidate')]:
 folder=OUT/dirname;folder.mkdir(exist_ok=True)
 for r in roles:
  if r['role']!=role:continue
  link=folder/(r['slide_id']+'.png');link.symlink_to(os.path.relpath(r['source_image'],folder))
  (folder/(r['slide_id']+'_selection.json')).write_text(json.dumps(r,indent=2)+'\n')
 (folder/'READ_ME.txt').write_text('These are links to unchanged source images, not copies or new training patches. Read split_plan.json. No training has occurred. Full source annotations are NOT automatically approved for every root.\n')
text='''TEACHING AND TESTING CHOICES — 16 NOVEMBER2026

I checked five distinct slides. TIFFs, PNG copies and patches are not separate specimens.
No model has been trained or run during this selection. Source images and labels are unchanged.

1. TEACHING — examples used to learn
20#MT28_gfp: six roots with archived Periderm,Endoderm and lateral-root annotations.
25717_gfp: six roots with the same three saved annotation categories.
MT25_rep3: initially roots4 and5 using Gonzalo's refined Periderm/Endoderm outlines.
This gives14 candidate roots across3 slides, not14 independent slides.
These images have already been used in local training, so they are not final-test images.
The two archived slides add different root/cell appearance and brightness. Their image
pixels match the images embedded in their annotations, and basic coordinate checks pass.
Their annotation author is not established; biological review is still required.

2. DEVELOPMENT CHECK — a practice exam while improving the model
20_29: keep it separate from the NEXT training set. Use it to check whether changes help
on another slide. Begin with its previously identified clearer root4 transition.
It has already influenced experiments, and rejected models45/46 trained on two areas.
Therefore it is NOT a fresh independent final test. Do not initialize the next model from
those adapted candidates if using20_29 as development-only. The preserved baseline39 did
not use20_29 in its local training. Its original public PAT ancestry is still unaudited.
No completed human reference JSON was found. Existing assistant-drawn sparse masks were
used for training and must not be called independently verified ground truth.

3. RESERVED TEST CANDIDATE — keep aside while improving the model
2_12: no use in our local gradient training was found. Keep the WHOLE slide out of
training and model selection. No new model predictions were generated today.
It has visible cells but also branches and a partial root near the bottom. Do not select
only successful-looking cases. Record whether each root is measurable before scoring.
We still need human reference boundaries/manual lengths for this small evaluation set.
Previous predictions have already been viewed, and original PAT training membership is
unknown: it is a provisional test candidate, NOT an untouched blind final test.
For a strong final scientific claim, reserve additional previously unused slides later.

WHAT NEEDS ATTENTION BEFORE TRAINING
Some MT25 practice outlines and archived outlines disagree about where Periderm starts.
Hold roots1,2,3 and6 aside for label reconciliation. This does not mean either annotator
is automatically wrong. Root6's two starting x-coordinates differ by about3764pixels;
this is a coordinate difference, not an established biological length error.
The earlier dataset ignored disagreement pixels, including transition areas; it did not
resolve the biological disagreement. This may limit the teaching signal, but the audit
does not prove it is the sole cause of prediction failures.
Archived labels have no explicit Background polygons. Unlabelled space is not automatically
background. Keep lateral-root annotations and decide their handling explicitly.
The immediate next step is a brief review of the target transition/end convention and the
existing candidate labels. We do not need to redraw every contour or hundreds of images.

SIMPLE DISTINCTION
Teaching = learn from the examples.
Development check = compare changes while we are still improving the method.
Reserved test = check the chosen method after decisions are frozen.
Keep all roots, patches, image conversions and augmentations from one slide in its assigned
group. This follows grouped evaluation guidance:
https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data

FILES
split_plan.json: exact proposed roles, roots, paths, hashes and readiness conditions.
inventory_and_annotation_audit.json: checks, label counts and outline disagreements.
visual_checks: source-only and saved-outline review figures; no model predictions.
Numbered reference folders contain shortcuts, saving disk space.
'''
(OUT/'READ_FIRST.txt').write_text(text)
# Retain a compact, machine-checkable selection audit separate from the user's data.
checks={'source_groups_disjoint':True,'unique_slides':5,'teaching_root_count':14,'archived_files_hashes_match':all(r.get('expected_file_hashes_match',True) for r in audit['source_images']),'archived_embedded_pixels_match':all(r.get('embedded_image_matches',True) for r in audit['source_images']),'tiff_png_pairs_match':all(r.get('tiff_png_pixel_identical_verified_today',True) for r in audit['source_images']),'no_basic_annotation_structural_errors':all(not r.get('structural_warnings',[]) for r in audit['source_images']) and not audit['GV_current_annotation_summary']['structural_warnings'],'no_training_or_new_inference':True,'biological_labels_certified':False,'pristine_final_test_available':False}
assert all(checks[k] for k in ['source_groups_disjoint','archived_files_hashes_match','archived_embedded_pixels_match','tiff_png_pairs_match','no_basic_annotation_structural_errors'])
(OUT/'selection_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
