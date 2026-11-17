"""Conservative training-only label review against archived annotations.

Gonzalo's rough early training polygons and archived polygons disagree in some
places. Ignore those tissue pixels for training rather than choosing one answer
or inventing a new boundary. Root4 reference and root6 remain untouched.
Run after22, before20. This modifies only the new35 derived dataset, once.
"""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'04_Results/35_Multiple_Slides_Data'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 mp=DATA/'manifest.json';d=json.loads(mp.read_text());assert 'conservative_label_review' not in d
 ap=ROOT/'02_Data/01_PAT_Examples/02_Annotations/MT25_rep3.json';a=json.loads(ap.read_text());w,h=a['imageWidth'],a['imageHeight'];flags=np.zeros((h,w),dtype='uint8')
 # Only shapes belonging to training roots1,2,3,5; never draw root4/6.
 tissue=[s for s in a['shapes'] if s['label'] in ['Periderm','Endoderm']]
 assert len(tissue)==12
 shapes=[tissue[i] for i in [0,1,2,3,4,5,8,9]]
 assert [s['label'] for s in shapes]==['Periderm','Endoderm']*4
 for label,value in [('Periderm',1),('Endoderm',2)]:
  canvas=Image.new('1',(w,h));dr=ImageDraw.Draw(canvas)
  for s in shapes:
   if s['label']==label:dr.polygon([tuple(p) for p in s['points']],fill=1)
  flags[np.asarray(canvas)] |= value
 lookup=np.array([0,1,2,255],dtype='uint8');mask=lookup[flags];changes=[]
 shutil.copy2(mp,DATA/'manifest_before_training_label_review.json')
 for p in d['patches']:
  if 'slide' in p:continue
  pp=DATA/p['mask'];m=np.array(Image.open(pp));b=p['box'];ref=mask[b[1]:b[3],b[0]:b[2]];ignore=np.isin(m,[1,2])&(m!=ref)
  n=int(ignore.sum())
  if n:
   old=p['mask_sha256'];m[ignore]=0;Image.fromarray(m).save(pp);p['mask_sha256']=sha(pp);p['counts']={str(c):int((m==c).sum()) for c in [0,1,2,3,255]};changes.append({'patch':p['id'],'ignored_tissue_pixels':n,'previous_mask_sha256':old})
 d['sources_sha256'][str(ap)]=sha(ap);d['conservative_label_review']={'rule':'For training crops from MT25 only, keep manual tissue labels only where the archived training-root annotations agree. Disagreements become ignored0, never an alternative tissue or background. Background labels unchanged. Root4/6 shapes not rasterized.','source':str(ap),'changed_patches':changes,'note':'Counts include overlapping crops, not unique slide pixels.'}
 mp.write_text(json.dumps(d,indent=2)+'\n');shutil.copy2(__file__,DATA/'programs'/Path(__file__).name)
 print(f'Reviewed training labels: {len(changes)} fields have uncertain tissue pixels excluded. Original annotations and validation unchanged.',flush=True)
if __name__=='__main__':main()
