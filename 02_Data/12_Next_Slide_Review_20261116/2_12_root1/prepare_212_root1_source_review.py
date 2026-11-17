"""Source-only review of 2_12 root1 before frozen appearance inference."""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import gaussian_filter1d

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'02_Data/12_Next_Slide_Review_20261116/2_12_root1'
Image.MAX_IMAGE_PIXELS=300_000_000
font=lambda n:ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',n)
plan=json.loads((OUT/'pre_review_plan.json').read_text())
source=Path(plan['source'])
with source.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==plan['source_sha256']
im=Image.open(source).convert('RGB');bounds=plan['source_box_xyxy']
overview=im.crop(bounds).resize((1800,360),Image.Resampling.LANCZOS)
fig=Image.new('RGB',(1840,1030),'white');d=ImageDraw.Draw(fig)
d.text((20,14),'Next image: 2_12, root 1 (the upper root)',font=font(32),fill='black')
d.text((20,60),'Source microscopy only. No model prediction, sharpening or new training.',font=font(25),fill='#334155')
fig.paste(overview,(20,105));d=ImageDraw.Draw(fig)
fields=[]
for i,(letter,cx,yl,yh) in enumerate([('A',4700,250,950),('B',6900,50,450),('C',9000,300,900)]):
    roi=np.array(im.crop((cx-200,yl,cx+200,yh)))
    cy=yl+int(np.argmax(gaussian_filter1d(roi[:,:,1].astype(float).sum(1),10)))
    box=[cx-200,max(0,cy-140),cx+200,cy+140]
    crop=im.crop(box);crop.save(OUT/f'{letter}_native.png')
    xy=[20+(cx-bounds[0])*.24,105+(cy-bounds[1])*.24]
    d.rectangle((xy[0]-48,xy[1]-33.6,xy[0]+48,xy[1]+33.6),outline='white',width=2)
    label_y=max(108,xy[1]-67)
    d.text((xy[0]-10,label_y),letter,font=font(28),fill='white',stroke_width=1,stroke_fill='black')
    x=20+i*610
    d.text((x,493),f'{letter}: main-root cells',font=font(27),fill='black')
    # Enlarged nearest-neighbour display; original microscopy data retained.
    fig.paste(crop.resize((580,406),Image.Resampling.NEAREST),(x,535))
    fields.append({'name':letter,'source_box_xyxy':box,'source_center_xy':[cx,cy],
                   'biological_label':None,'choice_basis':'Source-only position, root continuity and visible cells; no model predictions consulted.'})
d=ImageDraw.Draw(fig)
d.text((20,965),'For A, B and C: Endoderm, mixed / transition, Periderm, or uncertain?',font=font(26),fill='black')
d.text((20,1000),'Look at the main root. The boxes locate the views; they are not tissue annotations.',font=font(22),fill='#334155')
fig.save(OUT/'2_12_ROOT1_SOURCE_REVIEW.png')
record=dict(plan,review_fields=fields,prepared_utc=datetime.now(timezone.utc).isoformat(),
            geometry_note='Field y centres follow source-green intensity inside manually delimited top-root search bands; no tissue label inferred.',
            display='Overview downscaled; details enlarged nearest-neighbour. Native crops preserved; no sharpening or generative restoration.',
            expert_reference_pending=True,prediction_occurred=False,source_file_unchanged=True)
(OUT/'source_review_details.json').write_text(json.dumps(record,indent=2)+'\n')
shutil.copy2(__file__,OUT/Path(__file__).name)
(OUT/'READ_ME.txt').write_text('2_12 ROOT 1 — SOURCE REVIEW BEFORE PREDICTION\n\nInspect upper main root only. A/B/C are unlabelled source views for expert review. No appearance inference or training occurred. Model head, encoder, preprocessing and rule remain frozen. This slide stays out of training/tuning. The fields were selected from source microscopy, not model success. Earlier unrelated U-Net predictions on this slide were viewed in the workspace, so this is not a pristine blind test. Source and native crops are unchanged.\n\nThe user should identify E, mixed/transition, P or uncertain and, where possible, an approximate left edge of mixed tissue before this detector is run. Do not infer pixel labels from these coarse field descriptions.\n')
print(json.dumps(fields,indent=2))
