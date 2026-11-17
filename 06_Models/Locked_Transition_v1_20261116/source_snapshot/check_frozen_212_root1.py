"""Frozen appearance-detector check on 2_12 root1; no training or tuning.

The expert supplies only a qualitative pre-prediction reference: B looks more
Endoderm-like, with transition a little to its right. Never invent an exact T.
"""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import check_frozen_a04_root3 as fixed
from appearance_experiment import ROOT, font
from extend_root3_reference_route import graph_route

OUT=ROOT/'04_Results/68_Fixed_212_Root1_Check'
REVIEW=ROOT/'02_Data/12_Next_Slide_Review_20261116/2_12_root1'
read,save,sha,now=fixed.read,fixed.save,fixed.sha,fixed.now
Image.MAX_IMAGE_PIXELS=300_000_000

def prepare():
    OUT.mkdir(exist_ok=False)
    plan=read(REVIEW/'source_review_details.json');source=Path(plan['source'])
    assert sha(source)==plan['source_sha256']
    bounds=plan['source_box_xyxy'];rgb=np.array(Image.open(source).convert('RGB').crop(bounds))
    assert np.array_equal(rgb,np.array(Image.open(REVIEW/'source_native.png')))
    h,w=rgb.shape[:2];g=np.pad(rgb[:,:,1],((0,(-h)%4),(0,(-w)%4)))
    green=np.array(Image.fromarray(g).resize((g.shape[1]//4,g.shape[0]//4),Image.Resampling.BOX),dtype=float)
    foreground=ndi.binary_fill_holes(ndi.binary_closing(ndi.gaussian_filter(green,1)>16,iterations=2))
    labels,n=ndi.label(foreground);eligible=[]
    for k in range(1,n+1):
        yx=np.argwhere(labels==k)
        if yx[:,1].min()<=4 and yx[:,1].max()>=green.shape[1]-5:eligible.append((len(yx),k,yx))
    if not eligible:raise ValueError('No source-fluorescence component spans the prepared crop. Inspect before inference.')
    _,k,yx=max(eligible,key=lambda z:z[0]);xmin,xmax=yx[:,1].min(),yx[:,1].max()
    start=np.array([xmin,np.median(yx[yx[:,1]<=xmin+2,0])]);stop=np.array([xmax,np.median(yx[yx[:,1]>=xmax-2,0])])
    raw,ends=graph_route(labels==k,start,stop);raw=raw*4+1.5
    arc0=np.r_[0,np.cumsum(np.linalg.norm(np.diff(raw,axis=0),axis=1))]
    positions=np.r_[np.arange(0,arc0[-1],4),arc0[-1]]
    path=ndi.gaussian_filter1d(np.c_[np.interp(positions,arc0,raw[:,0]),np.interp(positions,arc0,raw[:,1])],5,axis=0,mode='nearest')
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    assert np.isfinite(path).all() and (np.diff(path[:,0])>=-3).all()
    assert np.max(np.linalg.norm(np.diff(path,axis=0),axis=1))<12
    Image.fromarray(rgb).save(OUT/'image.png')
    save(OUT/'image_only_route.json',{
        'created_utc':now(),'source_image':str(source),'source_sha256':sha(source),
        'source_crop_xyxy':bounds,'source_size_xy':[w,h],
        'path_xy_crop_pixels':path.tolist(),'distance_along_path_pixels':arc.tolist(),
        'source_only_geometry':True,'tissue_labels_used':False,
        'settings':{'downsample':4,'green_threshold':16,'gaussian_sigma_coarse':1,
                    'closing_iterations_coarse':2,'resample_px':4,'path_smoothing_sigma_samples':5},
        'geometry_bottom_right_zero_padding_yx':[(-h)%4,(-w)%4],
        'endpoint_distances_coarse_pixels':ends,
        'endpoints':'Local crop-edge positions, not biological cut end/hypocotyl.',
        'limits':'Branch context can occur inside rectangular windows even when path follows the main root.'})
    b=next(f for f in plan['review_fields'] if f['name']=='B')
    expert={'recorded_utc':now(),'expert':'Gonzalo Villarino',
            'exact_statements':['T zone between B to the right of B',
                'the square i see in B seems more endoderm but if u move a bit to the right of B i see transtion zone'],
            'field_B':dict(b,biological_description='More Endoderm-like / predominantly Endoderm'),
            'transition_description':'A little to the right of B; approximate visual relation, no exact coordinate or bounded interval supplied.',
            'A_and_C_labels':None,'exact_T_source_xy':None,'exact_T_interval':None,
            'pure_Endoderm_in_every_B_window_confirmed':False,
            'scope':'Pre-prediction qualitative reference on main-root cells only; no pixel labels or lateral-root labels.',
            'used_for_training':False}
    save(OUT/'expert_appearance_review_01.json',expert)
    if not (REVIEW/'expert_appearance_review_01.json').exists():save(REVIEW/'expert_appearance_review_01.json',expert)
    manifest=read(fixed.CANDIDATE/'candidate_manifest.json')
    save(OUT/'check_plan.json',{
        'created_utc':now(),'head_sha256':manifest['head_sha256'],'encoder_sha256':manifest['encoder_sha256'],
        'expert_reference_sha256':sha(OUT/'expert_appearance_review_01.json'),
        'prediction_method':'Reuse check_frozen_a04_root3.predict unchanged: 320x256 native path-aligned windows, 64-pixel stride, standardize, frozen PAT features, frozen two-output head, same fixed run rule. Batch size8.',
        'comparison':'Only describe model-T relation to the B source rectangle and all scan centres within B. No precise T error or accuracy percentage. Show source close-up for biological review.',
        'source_scope':plan['roles'],'prior_exposure':plan['prior_exposure'],
        'new_training':False,'no_tuning':True,'no_expert_T_input_to_model':True,
        'no_full_root_length_measurement':True})
    overview=Image.fromarray(rgb).resize((1800,360),Image.Resampling.LANCZOS);traced=overview.copy();d=ImageDraw.Draw(traced)
    d.line([(x*.24,y*.24) for x,y in path],fill='white',width=2)
    fig=Image.new('RGB',(1840,900),'white');d=ImageDraw.Draw(fig)
    d.text((20,14),'2_12 root 1: verify the upper main-root route',font=font(32),fill='black')
    d.text((20,59),'Top: source. Bottom: image-derived path. No tissue annotations used.',font=font(25),fill='#334155')
    fig.paste(overview,(20,105));fig.paste(traced,(20,510));fig.save(OUT/'PREPARED_MAIN_ROOT_PATH.png')
    for name in [Path(__file__).name,'check_frozen_a04_root3.py','appearance_experiment.py','extend_root3_reference_route.py','test_both_appearances.py']:
        shutil.copy2(Path(__file__).with_name(name),OUT/name)
    print('Prepared root1 route; saved approximate reference. No inference yet.',flush=True)

def predict():
    assert read(OUT/'route_visual_check.json')['passed']
    fixed.OUT=OUT
    fixed.predict()

def compare():
    assert not (OUT/'comparison_result.json').exists()
    p=read(OUT/'prediction_before_comparison.json');expert=read(OUT/'expert_appearance_review_01.json');route=read(OUT/'image_only_route.json')
    prediction_hash=sha(OUT/'prediction_before_comparison.json')
    offset=np.array(route['source_crop_xyxy'][:2]);centers=np.array(p['centers_xy_crop_pixels'])+offset
    states=np.array(p['rule']['window_states']);b=expert['field_B']['source_box_xyxy']
    ids=np.flatnonzero((centers[:,0]>=b[0])&(centers[:,0]<=b[2])&(centers[:,1]>=b[1])&(centers[:,1]<=b[3]))
    counts={name:int((states[ids]==i).sum()) for i,name in enumerate(p['rule']['state_order'])}
    t=None if p['candidate_T_crop_xy'] is None else (np.array(p['candidate_T_crop_xy'])+offset).tolist()
    relation='No T returned' if t is None else ('left of B rectangle' if t[0]<b[0] else 'within B rectangle' if t[0]<=b[2] else 'right of B rectangle')
    result={'created_utc':now(),'model_T_source_xy':t,'relation_to_B':relation,
            'expert_expected_relation':expert['transition_description'],
            'B_model_window_ids':ids.tolist(),'B_model_window_counts':counts,
            'qualitative_reference_only':True,'T_error_pixels':None,'new_length_measurement':False,
            'no_automatic_acceptance':True,'labels_or_weights_changed':False,
            'limits':'B-window counts overlap and are not accuracy. Model windows can extend beyond the reviewed box and include lateral context. No precise expert T was supplied.'}
    save(OUT/'comparison_result.json',result)
    colors=['#1aa65e','#b44dc5','#db303f','#777777'];rgb=Image.open(OUT/'image.png').convert('RGB')
    fig=Image.new('RGB',(1840,1100),'white');d=ImageDraw.Draw(fig)
    d.text((20,14),'2_12 root 1: unchanged model prediction',font=font(32),fill='black')
    d.text((20,59),'Your reference: B looks more like Endoderm; transition is a little farther right.',font=font(25),fill='#334155')
    fig.paste(rgb.resize((1800,360),Image.Resampling.LANCZOS),(20,105));d=ImageDraw.Draw(fig)
    bx0=20+(b[0]-offset[0])*.24;bx1=20+(b[2]-offset[0])*.24
    d.rectangle((bx0,105+(b[1]-offset[1])*.24,bx1,105+(b[3]-offset[1])*.24),outline='white',width=2)
    d.text((bx0+10,110),'B',font=font(27),fill='white',stroke_width=1,stroke_fill='black')
    if t is not None:
        tx=20+(t[0]-offset[0])*.24;d.line((tx,105,tx,465),fill='#28b7ff',width=4)
        d.text((tx+10,150),'Model T',font=font(25),fill='#28b7ff',stroke_width=1,stroke_fill='black')
    for xy,state in zip(centers,states):
        x=20+(xy[0]-offset[0])*.24;d.rectangle((x-7,487,x+7,515),fill=colors[state])
    d.text((20,533),'Green = Endoderm only     Purple = mixed     Red = Periderm only     Grey = neither supported',font=font(25),fill='black')
    d.text((20,575),'Closer source view: B and the main-root cells to its right',font=font(27),fill='black')
    # Fixed source view based on expert B location, not selected to flatter prediction.
    close=[6400,0,9400,650];factor=.6
    panel=rgb.crop((close[0]-offset[0],close[1]-offset[1],close[2]-offset[0],close[3]-offset[1])).resize((1800,390),Image.Resampling.LANCZOS)
    fig.paste(panel,(20,619));d=ImageDraw.Draw(fig)
    d.rectangle((20+(b[0]-close[0])*factor,619+(b[1]-close[1])*factor,20+(b[2]-close[0])*factor,619+(b[3]-close[1])*factor),outline='white',width=2)
    d.text((30+(b[0]-close[0])*factor,625),'B',font=font(28),fill='white',stroke_width=1,stroke_fill='black')
    if t is not None and close[0]<=t[0]<=close[2]:
        tx=20+(t[0]-close[0])*factor;d.line((tx,619,tx,1009),fill='#28b7ff',width=4)
        d.text((tx+10,630),'Model T',font=font(25),fill='#28b7ff',stroke_width=1,stroke_fill='black')
    note='No T returned by the fixed rule.' if t is None else 'Blue is the model T candidate ('+relation+'). Its biological position needs your review.'
    d.text((20,1024),note,font=font(24),fill='#075985')
    d.text((20,1061),'Same model and settings. No training, manual shift, exact T-error score or full-root length measurement.',font=font(23),fill='#334155')
    fig.save(OUT/'212_ROOT1_FIXED_RESULT.png')
    assert sha(OUT/'prediction_before_comparison.json')==prediction_hash
    text='2_12 ROOT1: FIXED APPEARANCE DETECTOR\n\nExpert reference before inference: '+expert['exact_statements'][-1]+'\n\n'+note+'\nB window counts: '+json.dumps(counts)+'. These windows overlap; they are not accuracy percentages. No A/C truth labels or exact T coordinates were invented.\n\nThe upper-root route was derived from source fluorescence with the same geometry settings and inspected before inference. The exact frozen A04 inference routine was reused, with the same head/encoder, input window size, normalization, thresholds, stride and run rule. Predictions saved before comparison. No training, tuning, normalization fitting or manual displacement.\n\n2_12 remains outside training/tuning. Earlier U-Net predictions on this slide were viewed, so this is not a pristine blind test; original published PAT image exposure unknown. No complete root or length validation.\n\nReproduce into a NEW --output directory using this program: --phase prepare; inspect route and record passed route_visual_check.json; --phase predict; --phase compare.\n\nWord manual not edited. All original sources and models unchanged.\n'
    (OUT/'READ_ME.txt').write_text(text)
    with (ROOT/'05_Notes/03_Progress_and_Next_Step.txt').open('a') as f:
        f.write('\n16 November 2026 — 2_12 root1 frozen prediction\n'+expert['exact_statements'][-1]+'. Saved as approximate pre-prediction reference, no exact T. '+note+' B window counts '+json.dumps(counts)+'. No training/tuning or source/mask changes. Source-only upper-root path inspected; saved original prediction, code, inputs, scores, comparison and figure in04_Results/68_Fixed_212_Root1_Check. Word manual not edited.\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','predict','compare'],required=True)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output.resolve();{'prepare':prepare,'predict':predict,'compare':compare}[args.phase]()
