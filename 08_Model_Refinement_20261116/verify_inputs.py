from pathlib import Path
import hashlib,json,importlib.util
import numpy as np
from PIL import Image
from check_reference import raster,metrics
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
r=ROOT/'04_Results/52_Reference_Check_20261116';d=json.loads((r/'annotations_snapshot.json').read_text());actual=np.array(Image.open(r/'reference_mask.png'))
spec=importlib.util.spec_from_file_location('old_mask',ROOT/'03_Notebooks/07_make_background_mask.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert np.array_equal(m.make_mask(d),actual)
assert np.array_equal(raster(d),actual)
assert actual.shape==(1024,1024) and set(np.unique(actual))=={0,1,2,3}
assert metrics(np.where(actual==0,3,actual),actual)['tissue_mean_iou']==1
bad=np.where(actual==1,2,np.where(actual==2,1,3)).astype('uint8');assert metrics(bad,actual)['tissue_mean_iou']==0
manifest=json.loads((ROOT/'04_Results/53_Selected_Teaching_Data/manifest.json').read_text());sources=manifest['sources'];assert len(sources)==3 and sum(len(s['root_ids']) for s in sources)==14
for s in sources:
 assert sha(s['image'])==s['image_sha256'] and sha(s['annotation'])==s['annotation_sha256']
assert sha(ROOT/'06_Models/working_model.pt')=='234b7de1969d80dd1523ce52afc89d5952b5e9ba4a63cdd7b3c8bff5b1c13633'
results={'reference_mask_matches_existing_converter':True,'metric_identity_and_swapped_labels_checks':True,'training_slides':3,'primary_roots':14,'training_source_files_unchanged':True,'previous_working_weights_unchanged':True,'new_reference_labels_unchanged':sha(ROOT/'02_Data/05_Training_and_Evaluation/02_Evaluation_Images/03_Your_Annotations/20_29_root4_reference_GV.json')==sha(r/'annotations_snapshot.json'),'reserved2_12_used_this_run':False}
(ROOT/'04_Results/54_Refinement_Experiments/input_verification.json').write_text(json.dumps(results,indent=2)+'\n');print(results)
