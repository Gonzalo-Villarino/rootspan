"""Read-only source audit. Writes inspection figures/metadata, not training data."""
from pathlib import Path
import base64,hashlib,io,json,collections
import numpy as np
from PIL import Image,ImageDraw,ImageFont
Image.MAX_IMAGE_PIXELS=300000000
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
FONT='/System/Library/Fonts/Supplemental/Arial.ttf'
def font(n):return ImageFont.truetype(FONT,n)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def poly_summary(d):
 errors=[];counts=collections.Counter();w,h=d['imageWidth'],d['imageHeight']
 for i,s in enumerate(d['shapes']):
  counts[s['label']]+=1;p=np.array(s['points'],dtype=float)
  if not np.isfinite(p).all():errors.append([i,'nonfinite coordinates'])
  if (p[:,0]<0).any() or (p[:,0]>=w).any() or (p[:,1]<0).any() or (p[:,1]>=h).any():errors.append([i,'coordinates outside source bounds'])
  if s.get('shape_type','polygon')=='polygon' and len(p)<3:errors.append([i,'polygon fewer than3 points'])
 return {'labels':dict(counts),'structural_warnings':errors,'image_wh':[w,h]}
def peris(d):return sorted([s for s in d['shapes'] if s['label']=='Periderm'],key=lambda s:np.mean(np.array(s['points'])[:,1]))
records=[];overview=[];arch=json.loads((ROOT/'02_Data/01_PAT_Examples/SOURCE_MANIFEST.json').read_text())
for rec in arch['examples']:
 name=rec['example'];ip=ROOT/'02_Data/01_PAT_Examples/01_Images'/f'{name}.png';jp=ROOT/'02_Data/01_PAT_Examples/02_Annotations'/f'{name}.json';d=json.loads(jp.read_text());rgb=Image.open(ip).convert('RGB');original=np.array(rgb);embedded=np.array(Image.open(io.BytesIO(base64.b64decode(d['imageData']))).convert('RGB'));equal=np.array_equal(original,embedded);del embedded
 summary=poly_summary(d);summary.update(slide=name,image=str(ip),annotations=str(jp),image_sha256=sha(ip),annotation_sha256=sha(jp),decoded_pixels_sha256=hashlib.sha256(original.tobytes()).hexdigest(),embedded_image_matches=equal,expected_file_hashes_match=all(sha(ROOT/'02_Data/01_PAT_Examples'/f['relative_destination'])==f['sha256'] for f in rec['files']));del original
 ps=peris(d);ov=rgb.copy();ov.thumbnail((1500,1100));scale=ov.width/rgb.width;dr=ImageDraw.Draw(ov)
 for s in d['shapes']:
  color={'Periderm':'#ff4055','Endoderm':'#00d090','LR':'#a97af5'}.get(s['label'],'yellow');p=[tuple(np.array(q)*scale) for q in s['points']];dr.line(p+[p[0]],fill=color,width=1)
 for i,s in enumerate(ps,1):
  q=np.mean(np.array(s['points']),axis=0)*scale;dr.text(tuple(q+np.array([0,-25])),str(i),font=font(25),fill='white',stroke_width=1,stroke_fill='black')
 ov.save(OUT/'visual_checks'/f'{name}_annotations_overview.png');overview.append((name,rgb.copy()))
 sheet=Image.new('RGB',(1400,1710),'white');dr=ImageDraw.Draw(sheet);dr.text((15,12),f'{name}: six archived Periderm starts',font=font(27),fill='black');dr.text((15,50),'Raw image above; saved outlines below. No model predictions. Native crops, displayed at70%.',font=font(19),fill='#374151');boxes=[]
 for i,s in enumerate(ps):
  pp=np.array(s['points']);left=pp[pp[:,0]<=pp[:,0].min()+35];cx=int(np.mean(left[:,0]));cy=int(np.mean(left[:,1]));x0=max(0,min(rgb.width-1000,cx-500));y0=max(0,min(rgb.height-360,cy-180));box=[x0,y0,x0+1000,y0+360];boxes.append({'root':i+1,'box':box,'archived_periderm_min_x':float(pp[:,0].min())});raw=rgb.crop(box);ann=raw.copy();ad=ImageDraw.Draw(ann)
  for sh in d['shapes']:
   p=[(q[0]-x0,q[1]-y0) for q in sh['points']];color={'Periderm':'#ff4055','Endoderm':'#00d090','LR':'#a97af5'}.get(sh['label'],'yellow');ad.line(p+[p[0]],fill=color,width=2)
  x=10+(i%2)*700;y=90+(i//2)*540;dr.text((x,y),f'Root {i+1}',font=font(22),fill='black');sheet.paste(raw.resize((680,245)),(x,y+32));sheet.paste(ann.resize((680,245)),(x,y+279))
 sheet.save(OUT/'visual_checks'/f'{name}_transition_review.png');summary['transition_crops']=boxes;records.append(summary)
 # Avoid holding all full-resolution images in RAM for display contact sheet.
 overview[-1][1].thumbnail((700,440))
userpath=ROOT/'02_Data/02_My_First_Annotations/MT25_rep3_root5_refined_GV.json';ud=json.loads(userpath.read_text());u=poly_summary(ud);u.update(path=str(userpath),sha256=sha(userpath));old=json.loads((ROOT/'02_Data/01_PAT_Examples/02_Annotations/MT25_rep3.json').read_text());dif=[]
for i,(a,b) in enumerate(zip(peris(old),peris(ud)),1):
 ax=min(q[0] for q in a['points']);bx=min(q[0] for q in b['points']);dif.append({'root':i,'archived_P_start_min_x':ax,'GV_P_start_min_x':bx,'GV_minus_archived_x_px':bx-ax,'interpretation':'Outline position disagreement; not a true length error or a biological judgement.'})
manifest=json.loads((ROOT/'02_Data/05_Training_and_Evaluation/02_Evaluation_Images/evaluation_manifest.json').read_text())
for rec in manifest['images']:
 ip=Path(rec['labelme_png']);rgb=Image.open(ip).convert('RGB');arr=np.array(rgb);name=ip.stem.replace('_evaluation','');rr={'slide':name,'image':str(ip),'original_tiff':rec['copied_original'],'image_wh':list(rgb.size),'image_sha256':sha(ip),'decoded_pixels_sha256':hashlib.sha256(arr.tobytes()).hexdigest(),'expected_png_hash_matches':sha(ip)==rec['png_sha256'],'full_slide_human_json_found':False};records.append(rr);del arr;rgb.thumbnail((700,440));overview.append((name,rgb))
contact=Image.new('RGB',(1440,1590),'white');dr=ImageDraw.Draw(contact);dr.text((18,15),'Five distinct slides on this Mac',font=font(32),fill='black');dr.text((18,58),'Display thumbnails only. Source images and annotations unchanged. No training run.',font=font(22),fill='#374151')
for i,(name,im) in enumerate(overview):
 x=10+(i%2)*720;y=100+(i//2)*495;dr.text((x,y),name,font=font(25),fill='black');contact.paste(im,(x,y+38))
contact.save(OUT/'visual_checks/FIVE_SLIDES.png')
result={'date_local':'2026-11-16','timezone':'Europe/Zurich','unique_source_slide_count':len(records),'source_images':records,'GV_current_annotation_summary':u,'MT25_outline_position_differences':dif,'training_run':False,'sources_changed':False,'audit_scope':'Dimensions, source hashes, pixel identity for archived JSONs, label counts, finite coordinates/bounds/point counts, overview and transition crop inspections. Not a complete biological or polygon topology certification.'}
(OUT/'inventory_and_annotation_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'slides':[{'name':x['slide'],'labels':x.get('labels'),'warnings':x.get('structural_warnings')} for x in records],'GV_labels':u['labels'],'GV_warnings':u['structural_warnings'],'MT25_transition_x_differences':dif},indent=2))
