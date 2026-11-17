"""Draft main-root interior regions for biological review, NEVER automatic ground truth.
Uses the teaching COL0 image and archived P boundary; does not read predictions or holdout labels.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage as ndi
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'02_Data/07_Additional_Archived_Annotations_20261116';OUT=ROOT/'02_Data/08_Endoderm_Drafts_NEED_REVIEW';OUT.mkdir(exist_ok=False);Image.MAX_IMAGE_PIXELS=300_000_000
im=Image.open(SRC/'COL0_Slide1.png');d=json.loads((SRC/'COL0_Slide1.json').read_text());shapes=sorted(d['shapes'],key=lambda s:np.mean(np.array(s['points'])[:,1]));records=[];panels=[]
for rid,s in enumerate(shapes,1):
 pts=np.array(s['points']);tx=pts[:,0].min();near=pts[pts[:,0]<tx+100];cy=np.median(near[:,1]);x=int(np.clip(tx-768,0,im.width-1024));y=int(np.clip(cy-256,0,im.height-512));rgb=np.array(im.crop((x,y,x+1024,y+512)));g=ndi.gaussian_filter(rgb[:,:,1].astype('float32'),[2,8]);position=float(cy-y);centres={};tops={};bots={}
 # Trace the main horizontal bright band from its archive P anchor towards the left.
 for xx in range(768,95,-8):
  lo=max(0,int(position)-70);hi=min(512,int(position)+71);prof=np.median(g[:,max(0,xx-8):min(1024,xx+9)],axis=1);peak=lo+int(np.argmax(prof[lo:hi]));level=max(5.,.20*float(prof[peak]));a=peak;b=peak
  while a>lo and prof[a-1]>level:a-=1
  while b<hi-1 and prof[b+1]>level:b+=1
  cen=(a+b)/2;position=.7*cen+.3*position;half=max(4.,min(32.,.30*(b-a+1)));centres[xx]=position;tops[xx]=position-half;bots[xx]=position+half
 xs=list(range(128,577,16));poly=[(xx,tops[xx]) for xx in xs]+[(xx,bots[xx]) for xx in xs[::-1]]
 stem=f'COL0_root{rid}_Endoderm_DRAFT';ip=OUT/f'{stem}.png';jp=OUT/f'{stem}.json';Image.fromarray(rgb).save(ip)
 shape={'label':'PROPOSED_Endoderm_NEEDS_REVIEW','points':[[float(a),float(b)] for a,b in poly],'group_id':None,'description':'Assistant draft based on position left of archived Periderm and visible root interior. Biological identity NOT confirmed. Not used for training.','shape_type':'polygon','flags':{'assistant_proposal':True,'biological_review_pending':True}}
 doc={'version':'5.0.0','flags':{'review_only_do_not_train':True},'shapes':[shape],'imagePath':ip.name,'imageData':None,'imageHeight':512,'imageWidth':1024};jp.write_text(json.dumps(doc,indent=2)+'\n')
 vis=Image.fromarray(rgb);draw=ImageDraw.Draw(vis);draw.line(poly+[poly[0]],fill=(0,215,255),width=3);ppts=[tuple(p-[x,y]) for p in pts];draw.line(ppts+[ppts[0]],fill=(235,55,65),width=2);panels.append((rid,vis));records.append({'root':rid,'image':str(ip),'draft_json':str(jp),'source_box_xyxy':[x,y,x+1024,y+512],'draft_class':'PROPOSED_Endoderm_NEEDS_REVIEW','biological_identity_confirmed':False,'used_in_training':False})
canvas=Image.new('RGB',(1700,1530),'white');draw=ImageDraw.Draw(canvas);font=lambda n:ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',n);draw.text((20,15),'Endoderm drafts for biological review — NOT used for training',font=font(31),fill='black');draw.text((20,58),'Cyan outline = proposed main-root interior. Red outline = archived Periderm. Confirm cell identity, not colour.',font=font(23),fill='black')
for i,(rid,v) in enumerate(panels):
 xx=20+(i%2)*840;yy=110+(i//2)*465;draw.text((xx,yy),f'COL0 root {rid}',font=font(25),fill='black');canvas.paste(v.resize((820,410),Image.Resampling.LANCZOS),(xx,yy+37))
canvas.save(OUT/'REVIEW_SIX_DRAFTS.png');(OUT/'draft_manifest.json').write_text(json.dumps({'status':'PROPOSALS_ONLY_NOT_TRAINING_DATA','source_slide':'COL0_Slide1','reason':'Additional archived slide has P outlines only. Main-root tissue to the left is a proposed Endoderm source; classification needs biological confirmation.','inference':'No model prediction or checking image used to draw these drafts.','review_rule':'Confirm tissue identity and that cyan stays within the main root. Reject uncertain or branch regions. Do not approve solely because the outline is left of P.','drafts':records},indent=2)+'\n');print(OUT/'REVIEW_SIX_DRAFTS.png')
