"""Post-check context diagnostic on already saved outputs; no new inference."""
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import numpy as np
from PIL import Image, ImageDraw
from appearance_experiment import ROOT, font

OUT=ROOT/'04_Results/67_Fixed_A04_Root3_Check'
pred=json.loads((OUT/'prediction_before_comparison.json').read_text())
expert=json.loads((OUT/'expert_appearance_review_01.json').read_text())
result=json.loads((OUT/'comparison_result.json').read_text())
offset=np.array(pred['source_crop_xyxy'][:2]);centers=np.array(pred['centers_xy_crop_pixels'])
states=np.array(pred['rule']['window_states']);rows=[]
for field in expert['fields']:
    a,b,c,d=field['source_box_xyxy'];xy=centers+offset
    ids=np.flatnonzero((xy[:,0]>=a)&(xy[:,0]<=c)&(xy[:,1]>=b)&(xy[:,1]<=d))
    counts={name:int((states[ids]==i).sum()) for i,name in enumerate(pred['rule']['state_order'])}
    rows.append({'field':field['name'],'expert_field_appearance':field['biological_label'],
                 'sample_ids':ids.tolist(),'model_window_counts':counts})
diagnostic={'recorded_utc':datetime.now(timezone.utc).isoformat(),
    'analysis_role':'Post-hoc descriptive context check of already saved predictions; no model selection or inference changes.',
    'window_membership':'Scan centre lies inside expert review rectangle. Rotated model field can extend beyond its rectangle.',
    'fields':rows,'not_independent_accuracy':True,
    'interpretation':'A is consistently Endoderm across sampled centres. B is mostly Endoderm with two mixed calls near its right edge. C has three Endoderm-only and three mixed calls. Mixed appearance in a larger expert field does not require every smaller field to contain both cell types. The original single-centre comparison is limited; it does not show that the detector entirely missed mixed cells in C.',
    'T_interpretation':'Blue model T lies before B, which expert identifies as Endoderm. This appears inconsistent with the agreed one-boundary regional convention; do not accept it for measuring lengths. Exact T reference remains unspecified.',
    'next':'Review a clearer transition source before further training or changing the rule. Keep this blurred example and fixed result as a difficult development check.',
    'new_inference':False,'training':False}
(OUT/'region_context_diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n')

colors=['#1aa65e','#b44dc5','#db303f','#777777']
fig=Image.new('RGB',(1760,1180),'white');d=ImageDraw.Draw(fig)
d.text((20,14),'A04 root 3: appearance recognition and T placement',font=font(32),fill='black')
d.text((20,60),'Your review: A = Endoderm, B = Endoderm, C = transition. You noted that the image is blurry.',font=font(25),fill='#334155')
fig.paste(Image.open(OUT/'image.png').resize((1720,420),Image.Resampling.LANCZOS),(20,105));d=ImageDraw.Draw(fig)
for field in expert['fields']:
    a,b,c,e=field['source_box_xyxy'];x=(a+c)/2-offset[0]
    d.text((20+x*.4-10,113),field['name'],font=font(27),fill='white')
tx=20+pred['candidate_T_crop_xy'][0]*.4
d.line((tx,105,tx,525),fill='#28b7ff',width=4)
d.text((tx+8,150),'Model T',font=font(24),fill='#28b7ff',stroke_width=1,stroke_fill='black')
d.text((20,548),'Unchanged model calls along the root:',font=font(25),fill='black')
for xy,state in zip(centers,states):
    x=20+xy[0]*.4;d.rectangle((x-12,588,x+12,618),fill=colors[state])
d.text((20,637),'Green = Endoderm only     Purple = mixed appearance     Red = Periderm only',font=font(25),fill='black')
for i,(row,near) in enumerate(zip(rows,result['local_field_comparison'])):
    x=20+580*i;count=row['model_window_counts']
    d.text((x,681),f'{row["field"]}: you = {row["expert_field_appearance"]}',font=font(25),fill='black')
    d.text((x,719),f'Model: {count["Endoderm_only"]} Endoderm; {count["both_mixed"]} mixed',font=font(25),fill='#334155')
    im=Image.open(OUT/'windows'/f'window_{near["nearest_scan_index"]:03d}.png')
    fig.paste(im.resize((400,320),Image.Resampling.NEAREST),(x+75,761))
d=ImageDraw.Draw(fig)
d.text((20,1100),'Blue T falls before your Endoderm field B: its placement is not accepted for length measurement.',font=font(25),fill='#075985')
d.text((20,1136),'Counts are overlapping windows, not accuracy. Bottom: central processed field for each area. No new training.',font=font(23),fill='#334155')
fig.save(OUT/'A04_ROOT3_FIELD_CONTEXT.png')
with (OUT/'READ_ME.txt').open('a') as f:
    f.write('\nCONTEXT UPDATE: '+diagnostic['interpretation']+'\n'+diagnostic['T_interpretation']+'\nSee A04_ROOT3_FIELD_CONTEXT.png and region_context_diagnostic.json. Do not interpret the initial nearest-centre comparison as absence of all mixed predictions in C.\n')
with (ROOT/'05_Notes/03_Progress_and_Next_Step.txt').open('a') as f:
    f.write('A04 context diagnostic (same saved predictions, no rerun): A 7 Endoderm/0 mixed; B 5 Endoderm/2 mixed; C 3 Endoderm/3 mixed at window centres inside reviewed rectangles. These overlap and are not accuracy statistics. C is not wholly missed; the initial single-centre comparison was incomplete. The early blue T remains unaccepted before expert-Endoderm B. Saved region_context_diagnostic.json and A04_ROOT3_FIELD_CONTEXT.png.\n')
shutil.copy2(__file__,OUT/Path(__file__).name)
print(json.dumps(rows,indent=2))
