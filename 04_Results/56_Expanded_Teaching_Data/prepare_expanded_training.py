"""Add archived COL0 Periderm-only labels. A04 remains excluded from teaching."""
from pathlib import Path
import json,shutil
import numpy as np
from PIL import Image
from prepare_training import sha,poly,bounds,touch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'04_Results/56_Expanded_Teaching_Data';OLD=ROOT/'04_Results/53_Selected_Teaching_Data';Image.MAX_IMAGE_PIXELS=300_000_000
OUT.mkdir(exist_ok=False);(OUT/'images').mkdir();(OUT/'masks').mkdir();man=json.loads((OLD/'manifest.json').read_text())
for p in man['patches']:
 for k in ['image','mask']:p[k]=str((OLD/p[k]).resolve())
imports=json.loads((ROOT/'02_Data/07_Additional_Archived_Annotations_20261116/import_manifest.json').read_text());source=next(s for s in imports['records'] if s['slide']=='COL0_Slide1');ap=Path(source['annotation']);ip=Path(source['image']);assert sha(ap)==source['annotation_sha256'] and sha(ip)==source['image_sha256'];d=json.loads(ap.read_text());im=Image.open(ip);shapes=sorted(d['shapes'],key=lambda s:np.mean(np.array(s['points'])[:,1]));boxes={}
for r,s in enumerate(shapes,1):
 b=bounds(s)
 for frac in [0,.04,.1,.2,.35,.5,.7,.9]:
  xx=int(np.clip(b[0]+frac*(b[2]-b[0])-384,0,im.width-768));strip=poly(s,[xx,0,xx+768,im.height]);yy=np.where(strip[:,320:448])[0]
  if len(yy):boxes[(xx,int(np.clip(np.median(yy)-384,0,im.height-768)))]=r
for (x,y),r in boxes.items():
 box=[x,y,x+768,y+768];mask=np.zeros((768,768),dtype='uint8')
 for s in shapes:
  if touch(box,bounds(s)):mask[poly(s,box)]=1
 name=f'COL0_Slide1_x{x}_y{y}';pi=OUT/'images'/f'{name}.png';pm=OUT/'masks'/f'{name}.png';im.crop(box).save(pi);Image.fromarray(mask).save(pm);counts={str(c):int((mask==c).sum()) for c in [0,1,2,3,4,255]};man['patches'].append({'id':name,'slide':'COL0_Slide1','root':r,'sample_type':'Periderm','box':box,'image':str(pi),'mask':str(pm),'counts':counts,'image_sha256':sha(pi),'mask_sha256':sha(pm)})
man['sources'].append({'slide':'COL0_Slide1','image':str(ip),'image_sha256':sha(ip),'annotation':str(ap),'annotation_sha256':sha(ap),'root_ids':list(range(1,7))});man['policy']='Original14roots on3slides plus6Periderm-only annotatedCOL0roots. 20roots4slides. Other COL0 pixels remain unlabelled. A04 and20_29 excluded from gradient training;2_12 reserved untouched. No labels invented.';man['class_support_note']='Only original14roots have both P/E labels; added6roots supply Periderm alone.';(OUT/'manifest.json').write_text(json.dumps(man,indent=2)+'\n');shutil.copy2(__file__,OUT/Path(__file__).name);print(len(boxes),'additional fields;',len(man['patches']),'total;4teaching slides;A04excluded',flush=True)
