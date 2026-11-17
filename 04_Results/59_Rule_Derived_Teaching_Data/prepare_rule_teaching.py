"""Rule-derived Endoderm examples on COL0 only, following user's confirmed main-root orientation.
These are NOT hand-drawn human ground truth. Existing human-reference20_29 and reserved A04 are never training inputs.
"""
from pathlib import Path
import json,shutil
import numpy as np
from PIL import Image,ImageDraw
from prepare_training import sha,poly,bounds,touch
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'04_Results/56_Expanded_Teaching_Data';OUT=ROOT/'04_Results/59_Rule_Derived_Teaching_Data';SRC=ROOT/'02_Data/07_Additional_Archived_Annotations_20261116';DRAFT=ROOT/'02_Data/08_Endoderm_Drafts_NEED_REVIEW';Image.MAX_IMAGE_PIXELS=300_000_000
OUT.mkdir(exist_ok=False);(OUT/'images').mkdir();(OUT/'masks').mkdir();man=json.loads((OLD/'manifest.json').read_text());d=json.loads((SRC/'COL0_Slide1.json').read_text());drafts=json.loads((DRAFT/'draft_manifest.json').read_text());im=Image.open(SRC/'COL0_Slide1.png');rules=[]
for r in drafts['drafts']:
 s=json.loads(Path(r['draft_json']).read_text())['shapes'][0];pts=np.array(s['points']);top=pts[:len(pts)//2];bottom=pts[len(pts)//2:][::-1];assert np.array_equal(top[:,0],bottom[:,0]);x0,y0,_,_=r['source_box_xyxy'];xs=top[:,0];keep=np.ones(len(xs),bool)
 # Explicit source-only visual review identified branch origins inside drafts4and5.
 # Exclude them conservatively rather than assigning lateral-root cells to Endoderm.
 if r['root']==4:keep=xs>=336
 if r['root']==5:keep=xs<=320
 t=top[keep];b=bottom[keep];assert len(t)>=6
 points=np.vstack([t,b[::-1]])+[x0,y0];rules.append({'label':'Endoderm','points':points.tolist(),'shape_type':'polygon','flags':{'rule_derived':True},'root_id':r['root'],'provenance':'User-confirmed E-left/P-right main-root orientation plus archive P boundary and image-only interior tracing. Class identity of individual regions not independently certified. More than192px left of P start; uncertain/branch areas excluded.'})
(OUT/'COL0_rule_derived_Endoderm_regions.json').write_text(json.dumps({'source_image':str(SRC/'COL0_Slide1.png'),'shapes':rules,'status':'Rule-derived teaching experiment; not independent human ground truth.','approved_biological_rule':'User states hypocotyl is always on right, Periderm proximal/right, Endoderm distal/left along primary root. Does not apply to lateral roots or background.','do_not_use_as_test_reference':True},indent=2)+'\n')
old_col=[p for p in man['patches'] if p['slide']=='COL0_Slide1'];boxes={(p['box'][0],p['box'][1]):p.get('root') for p in old_col};human=sorted(d['shapes'],key=lambda s:np.mean(np.array(s['points'])[:,1]))
for r,s in enumerate(human,1):
 a=np.array(s['points']);tx=a[:,0].min();cy=np.mean(a[a[:,0]<tx+150,1])
 for off in [-128,0,128]:boxes[(int(np.clip(tx+off-384,0,im.width-768)),int(np.clip(cy-384,0,im.height-768)))]=r
man['patches']=[p for p in man['patches'] if p['slide']!='COL0_Slide1'];weak_count=0
for (x,y),rid in boxes.items():
 box=[x,y,x+768,y+768];bits=np.zeros((768,768),dtype='uint8')
 for s in human:
  if touch(box,bounds(s)):bits[poly(s,box)]|=1
 for s in rules:
  if touch(box,bounds(s)):bits[poly(s,box)]|=2
 mask=np.zeros_like(bits);mask[bits==1]=1;mask[bits==2]=2;mask[bits==3]=255;counts={str(c):int((mask==c).sum()) for c in [0,1,2,3,4,255]};assert counts['255']==0
 name=f'COL0_Slide1_x{x}_y{y}';pi=OUT/'images'/f'{name}.png';pm=OUT/'masks'/f'{name}.png';im.crop(box).save(pi);Image.fromarray(mask).save(pm);weak_count+=counts['2']>0;man['patches'].append({'id':name,'slide':'COL0_Slide1','root':rid,'sample_type':'rule_transition' if counts['2'] else 'Periderm','box':box,'image':str(pi),'mask':str(pm),'counts':counts,'image_sha256':sha(pi),'mask_sha256':sha(pm),'rule_derived_codes':[2] if counts['2'] else [],'loss_weight':.5 if counts['2'] else 1.})
man['policy']='20roots4slides:14oldrootswithsavedP/E;6COL0rootswithsavedP and experimental rule-derivedE. Primary-rootinterioronly;branches/transitiongap excluded.20_29developmentonly,A04nottraining,2_12unused. Rule-labelled fields have half loss weight.';man['rule_regions']=str(OUT/'COL0_rule_derived_Endoderm_regions.json');man['rule_regions_sha256']=sha(OUT/'COL0_rule_derived_Endoderm_regions.json');(OUT/'manifest.json').write_text(json.dumps(man,indent=2)+'\n');shutil.copy2(__file__,OUT/Path(__file__).name);print(len(man['patches']),'fields;',weak_count,'containrule-derivedEndoderm;zeroP/Econflicts',flush=True)
