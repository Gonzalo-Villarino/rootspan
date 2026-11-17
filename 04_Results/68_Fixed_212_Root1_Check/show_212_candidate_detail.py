"""Source microscopy at saved model T and B; display only, no new inference."""
from pathlib import Path
import json
import shutil
import numpy as np
from PIL import Image,ImageDraw
from appearance_experiment import ROOT,font

OUT=ROOT/'04_Results/68_Fixed_212_Root1_Check'
p=json.loads((OUT/'prediction_before_comparison.json').read_text())
e=json.loads((OUT/'expert_appearance_review_01.json').read_text())
assert p['candidate_T_crop_xy'] is not None
offset=np.array(p['source_crop_xyxy'][:2]);t=np.array(p['candidate_T_crop_xy'])+offset
b=e['field_B']['source_box_xyxy'];cx=int(round(t[0]));cy=int(round(t[1]))
left=[cx-500,max(0,cy-210),cx+500,max(0,cy-210)+420]
right=[6600,0,7600,420]
rgb=Image.open(OUT/'image.png').convert('RGB')
fig=Image.new('RGB',(1840,610),'white');d=ImageDraw.Draw(fig)
d.text((20,14),'2_12: inspect the cells at the model T',font=font(32),fill='black')
d.text((20,58),'The model chose a position left of B. Your earlier reference was a little to the right of B.',font=font(25),fill='#334155')
for i,(box,title) in enumerate(zip([left,right],['1  Cells at the model T','2  B and the cells just to its right'])):
    x=20+920*i;y=150;scale=.88
    panel=rgb.crop((box[0]-offset[0],box[1]-offset[1],box[2]-offset[0],box[3]-offset[1]))
    panel=panel.resize((880,370),Image.Resampling.LANCZOS)
    fig.paste(panel,(x,y));d=ImageDraw.Draw(fig)
    d.text((x,108),title,font=font(27),fill='black')
    if i==0:
        tx=x+(t[0]-box[0])*scale
        d.line((tx,y,tx,y+370),fill='#28b7ff',width=4)
        d.text((tx+10,y+15),'Model T',font=font(25),fill='#28b7ff',stroke_width=1,stroke_fill='black')
    else:
        d.rectangle((x+(b[0]-box[0])*scale,y+(b[1]-box[1])*370/420,
                     x+(b[2]-box[0])*scale,y+(b[3]-box[1])*370/420),outline='white',width=2)
        d.text((x+(b[0]-box[0])*scale+10,y+15),'B',font=font(26),fill='white',stroke_width=1,stroke_fill='black')
d=ImageDraw.Draw(fig)
d.text((20,540),'Both panels show 1,000 x 420 original pixels at the same display scale. Look at the main root only.',font=font(24),fill='#334155')
d.text((20,577),'No sharpening, new training or manual movement of T. The blue line still needs biological review.',font=font(23),fill='#075985')
fig.save(OUT/'212_MODEL_T_CELL_REVIEW.png')
(OUT/'closeup_source_boxes.json').write_text(json.dumps({'left_model_T_view':left,'right_B_view':right,'model_T_source_xy':t.tolist(),'source_crop_offset':offset.tolist(),'display_scale_xy':[.88,370/420],'not_a_prediction_rerun':True},indent=2)+'\n')
shutil.copy2(__file__,OUT/Path(__file__).name)
print(OUT/'212_MODEL_T_CELL_REVIEW.png')
