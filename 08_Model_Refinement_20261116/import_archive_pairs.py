"""Copy two additional archived Labelme pairs; preserve source files, verify embedded image against TIFF."""
from pathlib import Path
import json,hashlib,base64,io,shutil
from PIL import Image,ImageDraw,ImageFont
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'02_Data/07_Additional_Archived_Annotations_20261116';SOURCE=Path('/Volumes/SALK_DATA/GV_Laptop_2019-2024_SALK/FY-GWAS_FY_Candidate_Genes/GWAS-MTs-FY/FY47_WRKYs/OUTPUT/Col_MT');Image.MAX_IMAGE_PIXELS=300_000_000

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pixhash(im):
 h=hashlib.sha256()
 for y in range(0,im.height,256):h.update(im.crop((0,y,im.width,min(y+256,im.height))).convert('RGB').tobytes())
 return h.hexdigest()
OUT.mkdir(exist_ok=False);records=[]
for name,role in [('COL0_Slide1','teaching_Periderm_only'),('A04_Slide2','reserved_Periderm_check_only')]:
 source=SOURCE/f'{name}.json';target=OUT/f'{name}.json';shutil.copy2(source,target);assert sha(source)==sha(target);d=json.loads(target.read_text());im=Image.open(io.BytesIO(base64.b64decode(d['imageData'])));tif=Image.open(SOURCE/d['imagePath']);assert im.size==tif.size==(d['imageWidth'],d['imageHeight']);h1=pixhash(im);h2=pixhash(tif);assert h1==h2,'Embedded image differs from TIFF; do not proceed';tif.close();ip=OUT/f'{name}.png';im.save(ip)
 for s in d['shapes']:
  assert s['label']=='Periderm' and s['shape_type']=='polygon';a=np.array(s['points']);assert a.shape[1]==2 and np.isfinite(a).all() and a.min()>=0 and (a[:,0]<im.width).all() and (a[:,1]<im.height).all()
 record={'slide':name,'role':role,'image':str(ip),'image_sha256':sha(ip),'annotation':str(target),'annotation_sha256':sha(target),'source_annotation':str(source),'source_tiff':str(SOURCE/d['imagePath']),'size_wh':list(im.size),'RGB_pixel_sha256':h1,'embedded_matches_TIFF_pixels':True,'polygons':len(d['shapes']),'labels':['Periderm'],'annotation_author':'Unconfirmed archive annotation; not automatically attributed to Ling or Gonzalo.','outside_polygons':'Unlabelled, never inferred as Endoderm or Background.'};records.append(record)
 # Source-only contact sheet: no prediction, no new annotation.
 canvas=Image.new('RGB',(1400,900),'white');dr=ImageDraw.Draw(canvas);font=lambda n:ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',n);dr.text((15,10),name+' | Saved Periderm outlines; source review only',font=font(23),fill='black')
 for i,s in enumerate(sorted(d['shapes'],key=lambda s:np.mean(np.array(s['points'])[:,1]))):
  a=np.array(s['points']);tx=a[:,0].min();near=a[a[:,0]<tx+256];cy=np.mean(near[:,1]);x=int(np.clip(tx-512,0,im.width-1024));y=int(np.clip(cy-384,0,im.height-768));crop=im.crop((x,y,x+1024,y+768)).convert('RGB');draw=ImageDraw.Draw(crop);draw.line([(px-x,py-y) for px,py in s['points']]+[(s['points'][0][0]-x,s['points'][0][1]-y)],fill=(220,45,55),width=3);crop=crop.resize((452,339));xx=15+(i%3)*463;yy=65+(i//3)*405;dr.text((xx,yy),f'Root {i+1}',font=font(21),fill='black');canvas.paste(crop,(xx,yy+30))
 canvas.save(OUT/f'{name}_source_review.png');im.close();print(name,record['size_wh'],'verified',flush=True)
(OUT/'import_manifest.json').write_text(json.dumps({'records':records,'policy':'No external originals modified. Only Periderm annotations present. Roles chosen before new-model predictions. A04 is a partial class check, not complete segmentation or length validation.'},indent=2)+'\n')
print('Saved bytes',sum(p.stat().st_size for p in OUT.iterdir()),flush=True)
