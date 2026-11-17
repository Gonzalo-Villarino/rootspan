"""New, source-grouped teaching fields. No development image enters training.
Reuses the selected saved human polygons, including explicit lateral-root labels.
Only saved Background polygons supervise Background. Other pixels are ignored.
"""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'04_Results/53_Selected_Teaching_Data'
Image.MAX_IMAGE_PIXELS=300_000_000
CODES={'Periderm':1,'Endoderm':2,'Background':3,'LR':4}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def poly(s,box):
 x,y,x1,y1=box;im=Image.new('1',(x1-x,y1-y));pts=[(a-x,b-y) for a,b in s['points']]
 if s.get('shape_type','polygon')=='rectangle':
  (a,b),(c,d)=pts;ImageDraw.Draw(im).rectangle((min(a,c),min(b,d),max(a,c),max(b,d)),fill=1)
 else:ImageDraw.Draw(im).polygon(pts,fill=1)
 return np.array(im)
def bounds(s):
 a=np.array(s['points']);return [*a.min(0),*a.max(0)]
def touch(a,b):return a[0]<b[2] and a[2]>b[0] and a[1]<b[3] and a[3]>b[1]
def main():
 OUT.mkdir(exist_ok=False);(OUT/'images').mkdir();(OUT/'masks').mkdir();(OUT/'source_snapshots').mkdir()
 plan=json.loads((ROOT/'02_Data/06_Dataset_Selection_20261116/split_plan.json').read_text());manifest={'patches':[],'sources':[],'policy':'Selected14 roots on3 teaching slides. Saved P/E human outlines only, named LR separate class, explicit manual Background only. Cross-class overlap ignored255; unknown0. MT25 disputed roots excluded. No development images, gradients, pseudo-labels, or predictions used.','role':'teaching','classes':CODES,'shape_order':'roots top-to-bottom by mean vertex y; verified during source audit.'}
 for role in plan['roles'][:3]:
  ip=Path(role['source_image']);ap=Path(role['annotation_source']);assert sha(ip)==role['image_sha256'];assert sha(ap)==role['annotation_sha256'];d=json.loads(ap.read_text());im=Image.open(ip);w,h=im.size;assert (w,h)==(d['imageWidth'],d['imageHeight'])
  by={label:sorted([s for s in d['shapes'] if s['label']==label],key=lambda s:np.mean(np.array(s['points'])[:,1])) for label in CODES}
  assert len(by['Periderm'])==len(by['Endoderm'])==6
  snapshots={k:v for k,v in d.items() if k!='imageData'};(OUT/'source_snapshots'/f"{role['slide_id']}.json").write_text(json.dumps(snapshots,indent=2)+'\n')
  roots=role['included_root_ids'];source={'slide':role['slide_id'],'image':str(ip),'image_sha256':sha(ip),'annotation':str(ap),'annotation_sha256':sha(ap),'root_ids':roots};manifest['sources'].append(source)
  # All labels from excluded MT25 roots are absent, including when another crop meets them.
  selected=[]
  for r in roots:selected += [by['Periderm'][r-1],by['Endoderm'][r-1]]
  selected+=by['Background']+by['LR'];sb=[bounds(s) for s in selected];boxes={}
  for r in roots:
   ps=by['Periderm'][r-1];es=by['Endoderm'][r-1];pp=np.array(ps['points']);ep=np.array(es['points']);tx=(pp[:,0].min()+ep[:,0].max())/2
   # Get root height near transition from a raster, not an assumed straight root.
   for offset in [-320,-160,0,160,320]:
    cx=tx+offset;xx=int(np.clip(cx-384,0,w-768));b=[xx,0,xx+768,h];band=poly(ps,b)|poly(es,b);yy=np.where(band[:,320:448])[0];cy=float(np.median(yy)) if len(yy) else float(np.mean(pp[:,1]));y=int(np.clip(cy-384,0,h-768));boxes[(xx,y)]=(r,'transition')
   for label,s in [('Periderm',ps),('Endoderm',es)]:
    bs=bounds(s)
    for frac in [.05,.2,.4,.6,.8,.95]:
     cx=bs[0]+frac*(bs[2]-bs[0]);xx=int(np.clip(cx-384,0,w-768));band=poly(s,[xx,0,xx+768,h]);yy=np.where(band[:,320:448])[0]
     if len(yy):boxes[(xx,int(np.clip(np.median(yy)-384,0,h-768)))]=(r,label)
  # LR examples on the two archived teaching slides; no LR labels inferred for MT25.
  for s in by['LR'][::3]:
   b=bounds(s);xx=int(np.clip((b[0]+b[2])/2-384,0,w-768));yy=int(np.clip((b[1]+b[3])/2-384,0,h-768));boxes[(xx,yy)]=(None,'LR')
  for s in by['Background']:
   b=bounds(s);xx=int(np.clip((b[0]+b[2])/2-384,0,w-768));yy=int(np.clip((b[1]+b[3])/2-384,0,h-768));boxes[(xx,yy)]=(None,'Background')
  for (x,y),(r,kind) in sorted(boxes.items()):
   box=[x,y,x+768,y+768];bits=np.zeros((768,768),dtype='uint8')
   for s,b in zip(selected,sb):
    if touch(box,b):bits[poly(s,box)]|=1<<(CODES[s['label']]-1)
   mask=np.full(bits.shape,255,dtype='uint8');mask[bits==0]=0
   for c in range(1,5):mask[bits==1<<(c-1)]=c
   counts={str(c):int((mask==c).sum()) for c in [0,1,2,3,4,255]}
   if sum(counts[str(c)] for c in [1,2,3,4])<1000:continue
   name=f"{role['slide_id'].replace('#','_')}_x{x}_y{y}";pi=OUT/'images'/f'{name}.png';pm=OUT/'masks'/f'{name}.png';im.crop(box).save(pi);Image.fromarray(mask).save(pm)
   manifest['patches'].append({'id':name,'slide':role['slide_id'],'root':r,'sample_type':kind,'box':box,'image':str(pi.relative_to(OUT)),'mask':str(pm.relative_to(OUT)),'counts':counts,'image_sha256':sha(pi),'mask_sha256':sha(pm)})
  print(role['slide_id'],len([p for p in manifest['patches'] if p['slide']==role['slide_id']]),'fields',flush=True);im.close()
 # Check that all gradient data sources belong to the declared teaching group.
 assert {s['slide'] for s in manifest['sources']}=={'MT25_rep3','20#MT28_gfp','25717_gfp'}
 assert sum(len(s['root_ids']) for s in manifest['sources'])==14
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');shutil.copy2(__file__,OUT/Path(__file__).name);print('Total',len(manifest['patches']),'fields. No training yet.',flush=True)
if __name__=='__main__':main()
