"""Meaningful invariants: anatomical rule, lateral-root exclusion, source roles and derived labels."""
from pathlib import Path
import json
import numpy as np
from main_root_rule import interpret
from prepare_training import sha
ROOT=Path(__file__).resolve().parents[1];tests={}
p=np.full((4,256,1024),.001,dtype='float32');p[2]=.997;p[:,100:141,:]=.001;p[1,100:141,:512]=.997;p[0,100:141,512:]=.997;p[:,141:230,720:736]=.001;p[3,141:230,720:736]=.997
out,meta=interpret(p);assert meta['applied'] and abs(meta['boundary_x']-512)<=2;assert out[120,200]==2 and out[120,800]==1 and out[50,500]==3 and out[180,725]==4 and out[120,512]==255;tests['synthetic_direction_root_background_and_lateral_branch']='PASS'
p=np.zeros((4,256,1024),dtype='float32');p[2]=1;_,m=interpret(p);assert not m['applied'];tests['no_root_does_not_invent_transition']='PASS'
folder=ROOT/'04_Results/59_Rule_Derived_Teaching_Data';man=json.loads((folder/'manifest.json').read_text());assert {s['slide'] for s in man['sources']}=={'MT25_rep3','20#MT28_gfp','25717_gfp','COL0_Slide1'};tests['20_29_A04_2_12_excluded_from_gradient_data']='PASS'
for s in man['sources']:
 assert sha(s['image'])==s['image_sha256'] and sha(s['annotation'])==s['annotation_sha256']
for q in man['patches']:
 assert sha(folder/q['image'])==q['image_sha256'] and sha(folder/q['mask'])==q['mask_sha256']
 if q.get('rule_derived_codes'):
  assert q['slide']=='COL0_Slide1' and q['rule_derived_codes']==[2] and q['loss_weight']==.5 and q['counts']['255']==0
assert sha(man['rule_regions'])==man['rule_regions_sha256'];tests['teaching_data_hashes_and_rule_provenance']='PASS'
assert sha(ROOT/'06_Models/working_model.pt')=='234b7de1969d80dd1523ce52afc89d5952b5e9ba4a63cdd7b3c8bff5b1c13633';tests['preserved_working_model']='PASS'
ref=ROOT/'02_Data/05_Training_and_Evaluation/02_Evaluation_Images/03_Your_Annotations/20_29_root4_reference_GV.json';assert sha(ref)=='49c15efb8b6f7f9284e812c93c22923c5ba2ba496c49ef6d9fb22ff4d1d23265';tests['user_reference_unchanged']='PASS'
(ROOT/'04_Results/54_Refinement_Experiments/rule_and_data_verification.json').write_text(json.dumps(tests,indent=2)+'\n');print(json.dumps(tests,indent=2))
