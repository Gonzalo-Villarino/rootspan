"""Record expert interpretation and render source-based figures; no inference/training."""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '04_Results/66_Fixed_20_29_Check'
REVIEW = OUT / 'expert_review_20261116'
NEXT = ROOT / '02_Data/11_Next_Root_Review_20261116/A04_root3'
BLUE = '#28b7ff'
Image.MAX_IMAGE_PIXELS = 300_000_000

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    return json.loads(path.read_text())

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def font(n):
    return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', n)

def main():
    REVIEW.mkdir(exist_ok=False)
    now = datetime.now(timezone.utc).isoformat()
    candidate = ROOT / '06_Models/20261116_Two_Appearance_Candidate'
    manifest = read(candidate / 'candidate_manifest.json')
    protected = [OUT/'image.png', OUT/'prediction_before_reference_comparison.json',
                 OUT/'comparison_result.json', Path(manifest['head']), Path(manifest['encoder']),
                 ROOT/'06_Models/working_model.pt']
    before = {str(p): sha(p) for p in protected}
    result = read(OUT/'comparison_result.json')
    pred = read(OUT/'prediction_before_reference_comparison.json')
    assert pred['candidate_T_image_xy'] is None
    record = {
        'recorded_utc': now,
        'expert': 'Gonzalo Villarino',
        'user_statements': [
            'The mixed appearance in 20_29, including the left-hand purple windows',
            'In the last pic u showed me the white lane is in the T zone',
            'Also use the blue line vertically for T zone for consistency'],
        'interpretation': 'Expert confirms mixed appearance including the left-hand purple windows and confirms the displayed reference line lies within the transition zone.',
        'not_established': ['Exact left edge of the mixed zone', 'Automatic model T',
                            'Every individual cell identity or pixel accuracy', 'Automatic length accuracy'],
        'reference_position_image_xy': result['reference_boundary_midpoint_image_xy'],
        'reference_position_unchanged': True,
        'reference_origin': 'Gap midpoint inferred from saved regional outlines, subsequently confirmed by expert as within the T zone.',
        'updated_interpretation': 'The prior claim that left mixed calls are necessarily biological errors is superseded by this expert review. Regional labels alone cannot establish pure versus mixed cell appearance. The unchanged detector still returned no T because its rule requires earlier clear-Endoderm support.',
        'display_convention': 'Vertical blue for T; label reference versus model explicitly. This figure contains a reference line only.',
        'prediction_changed': False, 'annotations_changed': False, 'training_occurred': False,
        'model_candidate_T': None,
        'next': 'Review a different root with wider context before unchanged-model inference; keep uncertain biology unlabelled.'}
    save(REVIEW/'biological_review.json', record)
    save(ROOT/'05_Notes/04_T_Display_Convention.json', {
        'recorded_utc': now, 'requested_by': 'Gonzalo Villarino',
        'T_line_color': BLUE, 'T_line_orientation': 'vertical',
        'required_caption': 'Always identify reference T versus model T. An expert confirming a line is inside the zone does not establish its exact left edge.',
        'reference_and_model_together': 'Blue dashed reference, blue solid model, both explicitly labelled.',
        'existing_historical_figures': 'Retained as historical records; new figures follow this convention.'})

    # Regenerate scientific display from the unchanged source and numerical output.
    rgb = Image.open(OUT/'image.png').convert('RGB')
    photo = rgb.crop((0,280,1024,660)).resize((1280,475),Image.Resampling.LANCZOS)
    fig = Image.new('RGB',(1320,930),'white'); d = ImageDraw.Draw(fig)
    d.text((20,14),'20_29: your biological review',font=font(30),fill='black')
    d.text((20,57),'You confirm mixed cells, including in the left-hand purple windows.',font=font(24),fill='#334155')
    fig.paste(photo,(20,100)); d=ImageDraw.Draw(fig)
    x=20+result['reference_boundary_midpoint_image_xy'][0]*1.25
    d.line((x,100,x,575),fill=BLUE,width=4)
    d.text((x+10,116),'T reference',font=font(25),fill=BLUE,stroke_width=1,stroke_fill='black')
    d.text((x+10,149),'Confirmed within the T zone',font=font(21),fill=BLUE,stroke_width=1,stroke_fill='black')
    d.text((20,599),'Unchanged model appearance predictions:',font=font(24),fill='black')
    colors=['#1aa65e','#b44dc5','#db303f','#777777']
    for xy,state in zip(pred['source_centers_xy'],pred['rule']['window_states']):
        px=20+xy[0]*1.25
        d.rectangle((px-30,644,px+30,680),fill=colors[state])
    d.text((20,702),'Green = Endoderm only     Purple = mixed appearance     Red = Periderm only',font=font(23),fill='black')
    d.text((20,749),'Blue is your reference position. The detector did not return its own T.',font=font(24),fill='#075985')
    d.text((20,790),'Its rule needs clear Endoderm before the mixed zone; it found none in this crop.',font=font(23),fill='#334155')
    d.text((20,831),'The blue line was not moved. Being inside the zone does not locate its exact left edge.',font=font(22),fill='#334155')
    d.text((20,873),'No retraining or annotation changes. Window appearances are not pixel-segmentation masks.',font=font(22),fill='#334155')
    fig.save(REVIEW/'20_29_BLUE_T_REFERENCE.png')

    # Prospective source review: root 3, on a different source image. No model output read.
    source=ROOT/'02_Data/07_Additional_Archived_Annotations_20261116/A04_Slide2.png'
    image=Image.open(source).convert('RGB'); bounds=(6800,4580,11100,5630)
    crop=image.crop(bounds); crop.save(NEXT/'source_native.png')
    overview=crop.resize((1720,420),Image.Resampling.LANCZOS)
    fig=Image.new('RGB',(1760,1050),'white');d=ImageDraw.Draw(fig)
    d.text((20,12),'Next source review: A04, root 3',font=font(32),fill='black')
    d.text((20,58),'Original microscopy only. No prediction or T line is shown.',font=font(25),fill='#334155')
    fig.paste(overview,(20,102)); d=ImageDraw.Draw(fig)
    fields=[]
    for i,(name,cx) in enumerate(zip('ABC',[8400,9400,10200])):
        field=np.array(image.crop((cx-200,4800,cx+200,5450)))
        row=gaussian_filter1d(field[:,:,1].astype(float).sum(1),10)
        cy=4800+int(np.argmax(row))
        box=[cx-200,cy-140,cx+200,cy+140]
        native=image.crop(box);native.save(NEXT/f'{name}_native.png')
        px=20+(cx-bounds[0])*.4; py=102+(cy-bounds[1])*.4
        d.rectangle((px-80,py-56,px+80,py+56),outline='white',width=2)
        d.text((px-10,py-94),name,font=font(28),fill='white')
        xx=20+i*580
        d.text((xx,547),f'{name}: main-root cells',font=font(26),fill='black')
        fig.paste(native.resize((560,392),Image.Resampling.NEAREST),(xx,590))
        fields.append({'name':name,'source_box_xyxy':box,'biological_label':None})
    d=ImageDraw.Draw(fig)
    d.text((20,1005),'Review the main root: clear Endoderm, mixed, clear Periderm, or uncertain. Ignore lateral branches.',font=font(25),fill='#334155')
    fig.save(NEXT/'A04_ROOT3_SOURCE_REVIEW.png')
    save(NEXT/'source_review_plan.json',{
        'created_utc':now,'source_image':str(source),'source_sha256':sha(source),
        'root':3,'source_box_xyxy':list(bounds),'review_fields':fields,
        'selection':'Root 3 chosen from source-only six-root overview for a comparatively straight main root near the archived Periderm outline start. Wider context is included. Selection did not use current detector predictions.',
        'source_status':'Familiar A04 slide used in earlier U-Net checks; not a new blind test. This appearance head was taught on COL0. Published PAT encoder exposure unknown.',
        'pending':'Expert appearance review and transition interval before inference; insufficient/uncertain source can be excluded with recorded reason.',
        'frozen_head_sha256':manifest['head_sha256'],'frozen_encoder_sha256':manifest['encoder_sha256'],
        'model_settings_change_planned':False,'new_training':False,'inference_done':False,
        'calibration':'A04 calibration not confirmed; do not report micrometre lengths.',
        'reserved_2_12_used':False})
    note=('16 November 2026 — Expert review corrects interpretation; blue T convention\n'
          'Gonzalo confirms mixed appearance including the left-hand purple windows in 20_29 and says the white reference line is within the T zone. '
          'The earlier inference that those mixed calls must be biological errors is superseded. The fixed detector still returned no T; this remains distinct from expert approval of appearance. '
          'Changed the display reference to vertical blue at the identical coordinate, explicitly labelled reference. No exact left edge newly confirmed. '
          'Source data, masks, predictions and model weights unchanged. Saved biological_review.json and 20_29_BLUE_T_REFERENCE.png in 04_Results/66_Fixed_20_29_Check/expert_review_20261116. '
          'Prepared source-only A04 root3 review (three detail windows and broader context) before unchanged-model inference. Labels remain pending expert review. '
          '2_12 reserve unused. Word manual not edited.\n')
    with (ROOT/'05_Notes/03_Progress_and_Next_Step.txt').open('a') as f:f.write('\n'+note)
    (REVIEW/'READ_ME.txt').write_text(note+'\nThese expert observations supersede the biological-error interpretation in the original README, not the numerical no-T prediction.\n')
    old=OUT/'READ_ME.txt'
    shutil.copy2(old,REVIEW/'READ_ME_before_expert_review.txt')
    old.write_text('EXPERT REVIEW UPDATE — 16 November 2026\nGonzalo confirms mixed appearance including the left-hand purple windows. The earlier interpretation below that these calls must be biological errors is superseded. The numerical result remains no T. The former white reference line is now blue at the same location; Gonzalo confirms it is within the transition zone, not an exact left-edge measurement. See expert_review_20261116/biological_review.json and 20_29_BLUE_T_REFERENCE.png. Original report retained below for provenance.\n\n'+old.read_text())
    evaluation=candidate/'evaluation_20_29_20261116.json';old_eval=read(evaluation)
    save(REVIEW/'evaluation_before_expert_review.json',old_eval)
    old_eval['outcome_before_expert_review']=old_eval['outcome']
    old_eval['outcome']='No automatic T returned. Expert approves mixed appearance including left-hand purple windows; prior biological-error interpretation superseded.'
    old_eval['expert_review']=str(REVIEW/'biological_review.json')
    old_eval['next']='Review source-only A04 root3, then check fixed detector if appearance/context is suitable. No new teaching yet.'
    save(evaluation,old_eval)
    assert before=={str(p):sha(p) for p in protected}
    save(REVIEW/'verification.json',{'created_utc':now,'protected_files_unchanged':before,
         'T_reference_image_xy_unchanged':result['reference_boundary_midpoint_image_xy'],
         'no_training':True,'no_new_model_inference':True,'figure_visual_inspection':'Pending'})
    shutil.copy2(__file__,REVIEW/Path(__file__).name)
    print(REVIEW/'20_29_BLUE_T_REFERENCE.png')
    print(NEXT/'A04_ROOT3_SOURCE_REVIEW.png')

if __name__=='__main__':main()
