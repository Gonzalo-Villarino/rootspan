"""Add tissue examples from two archived PAT slides already copied from SALK.

Existing annotations are reused, not newly drawn by AI. Outside their outlines
stays unlabelled; lateral-root labels and conflicts are ignored in this
three-category exercise. Background still comes from Gonzalo's explicit labels.
No root4 or root6 image pixels/labels are added from the practice slide.
"""
import hashlib,importlib.util,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'04_Results/35_Multiple_Slides_Data'
BASE=ROOT/'04_Results/33_Multiple_Roots_Data'
ARCHIVE=ROOT/'02_Data/01_PAT_Examples'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 manifest=json.loads((BASE/'manifest.json').read_text());shutil.copytree(BASE,OUT)
 Image.MAX_IMAGE_PIXELS=300000000
 record=json.loads((ARCHIVE/'SOURCE_MANIFEST.json').read_text())
 manifest['additional_archived_slides']=[]
 for stem in ['20#MT28_gfp','25717_gfp']:
  example=next(e for e in record['examples'] if e['example']==stem)
  for f in example['files']:assert sha(ARCHIVE/f['relative_destination'])==f['sha256']
  ap=ARCHIVE/'02_Annotations'/f'{stem}.json';ip=ARCHIVE/'01_Images'/f'{stem}.png';d=json.loads(ap.read_text());im=Image.open(ip).convert('RGB');w,h=im.size
  assert im.size==(d['imageWidth'],d['imageHeight'])
  flags=np.zeros((h,w),dtype='uint8')
  for label,flag in [('Periderm',1),('Endoderm',2),('LR',4)]:
   canvas=Image.new('1',(w,h));draw=ImageDraw.Draw(canvas)
   for s in d['shapes']:
    if s['label']==label:draw.polygon([tuple(p) for p in s['points']],fill=1)
   flags[np.asarray(canvas)] |= flag
  mask=np.array([0,1,2,255,255,255,255,255],dtype='uint8')[flags];del flags
  candidates=set()
  for s in d['shapes']:
   if s['label'] not in ['Periderm','Endoderm']:continue
   pts=np.array(s['points']);x0,y0=np.floor(pts.min(0)).astype(int);x1,y1=np.ceil(pts.max(0)).astype(int)
   canvas=Image.new('1',(w,h));ImageDraw.Draw(canvas).polygon([tuple(p) for p in s['points']],fill=1);region=np.asarray(canvas)
   for x in range(max(0,x0-64),min(w-512,x1-256)+1,384):
    yy=np.where(region[:,x+192:x+320])[0]
    if len(yy):candidates.add((x,int(np.clip(np.median(yy)-256,0,h-512))))
  added=0
  for x,y in sorted(candidates):
   m=mask[y:y+512,x:x+512].copy();counts={str(c):int((m==c).sum()) for c in [0,1,2,3,255]}
   if counts['1']+counts['2']<512:continue
   name=f'{stem.replace("#","_")}_x{x}_y{y}';pi=OUT/'images'/f'{name}.png';pm=OUT/'masks'/f'{name}.png';im.crop((x,y,x+512,y+512)).save(pi);Image.fromarray(m).save(pm)
   manifest['patches'].append({'id':name,'root':'archived','slide':stem,'box':[x,y,x+512,y+512],'image':str(pi.relative_to(OUT)),'mask':str(pm.relative_to(OUT)),'image_sha256':sha(pi),'mask_sha256':sha(pm),'counts':counts});added+=1
  manifest['sources_sha256'].update({str(ap):sha(ap),str(ip):sha(ip)})
  manifest['additional_archived_slides'].append({'slide':stem,'roots':6,'fields':added,'image':str(ip),'annotations':str(ap),'provenance':'Archived human annotations from SALK backup; individual annotator not established. Only saved Periderm/Endoderm used; LR/conflicts ignored; unlabelled remains unlabelled.'})
  d.pop('imageData',None);(OUT/f'{stem.replace("#","_")}_annotations_without_image_data.json').write_text(json.dumps(d))
  print(f'Added {added} labelled fields from {stem}. No automatic background.',flush=True)
 manifest['base_manifest_sha256']=sha(BASE/'manifest.json');manifest['training_slide_count']=3;manifest['training_root_count']=16
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');shutil.copy2(__file__,OUT/'programs'/Path(__file__).name)
 print(f'Total {len(manifest["patches"])} fields; original sources unchanged. No training.',flush=True)
if __name__=='__main__':main()
