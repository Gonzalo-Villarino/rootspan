from pathlib import Path
import sys,json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
P=Path(__file__).resolve().parents[1];D=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'06_Models/Locked_Transition_v1_20261116'))
from frozen_core import standardize
Image.MAX_IMAGE_PIXELS=300_000_000
out=D/'existing_field_labels';out.mkdir(exist_ok=False)
read=lambda f:json.loads(f.read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
original=read(P/'02_Data/10_Cell_Appearance_Review/confirmed_appearance_labels_03.json');records=[]
for e in original['examples']:
 image=P/f'04_Results/65_Mixed_Appearance_Evidence_Check/three_way_five_windows/teaching_{e["id"]}_processed.png'
 ident='COL0_'+e['id'];dest=out/(ident+'.png');dest.write_bytes(image.read_bytes())
 records.append({'id':ident,'slide':'COL0','root':e['root'],'class':e['class_index'],'image':str(dest.relative_to(P)),'image_sha256':sha(dest),'label_record':'02_Data/10_Cell_Appearance_Review/confirmed_appearance_labels_03.json','record_sha256':sha(P/'02_Data/10_Cell_Appearance_Review/confirmed_appearance_labels_03.json'),'existing_processed_image':str(image.relative_to(P)),'selected_for_training':None})
source=P/'02_Data/07_Additional_Archived_Annotations_20261116/A04_Slide2.png';rgb=Image.open(source).convert('RGB');source_sha=sha(source)
for root,name in [(1,'71_Locked_A04_Root1_Check'),(2,'72_Locked_A04_Root2_Check'),(3,'67_Fixed_A04_Root3_Check')]:
 record=P/f'04_Results/{name}/expert_appearance_review_01.json';r=read(record)
 for f in r['fields']:
  box=f['source_box_xyxy'];raw=np.asarray(rgb.crop(box));processed,info=standardize(raw)
  ident=f'A04_r{root}_{f["name"]}';dest=out/(ident+'.png');Image.fromarray(processed).save(dest)
  records.append({'id':ident,'slide':'A04','root':root,'class':f['appearance_state'],'image':str(dest.relative_to(P)),'image_sha256':sha(dest),'source':str(source.relative_to(P)),'source_sha256':source_sha,'reviewed_source_box':box,'preprocessing':info,'label_record':str(record.relative_to(P)),'record_sha256':sha(record),'expert_words':r.get('verbatim',r.get('exact_statement')),'label_scope':'Qualitative whole reviewed field. Central processed field inherits approximate appearance, not pixel labels.','selected_for_training':None})
(out/'audit_manifest.json').write_text(json.dumps({'training_not_started':True,'candidate_count':len(records),'records':records},indent=2)+'\n')
canvas=Image.new('RGB',(4*340,4*310),'white');d=ImageDraw.Draw(canvas);font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',20)
names=['Endoderm','Mixed','Periderm']
for i,r in enumerate(records):
 x=(i%4)*340;y=(i//4)*310
 d.text((x+8,y+5),r['id']+' · '+names[r['class']],font=font,fill='black')
 canvas.paste(Image.open(P/r['image']),(x+8,y+34))
canvas.save(out/'FIELD_AUDIT.png')
print('Prepared',len(records),'existing labelled fields; training selection pending review.')
