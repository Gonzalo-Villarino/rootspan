"""Prospective A04 root-3 check. Fixed model; expert labels only used to compare.

Prepare and visually check the fluorescence route before running predict.
This is an appearance-window check, not a validated automatic length measurement.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from appearance_experiment import ROOT, standardize, font
from extend_root3_reference_route import graph_route
from test_both_appearances import boundary_rule, check_rule

OUT=ROOT/'04_Results/67_Fixed_A04_Root3_Check'
REVIEW=ROOT/'02_Data/11_Next_Root_Review_20261116/A04_root3'
CANDIDATE=ROOT/'06_Models/20261116_Two_Appearance_Candidate'
COLORS=['#1aa65e','#b44dc5','#db303f','#777777']
NAMES=['Endoderm','Mixed / transition','Periderm','Neither supported']
Image.MAX_IMAGE_PIXELS=300_000_000

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(p):return json.loads(p.read_text())
def save(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()

def prepare():
    OUT.mkdir(exist_ok=False)
    plan=read(REVIEW/'source_review_plan.json')
    original=Path(plan['source_image']);assert sha(original)==plan['source_sha256']
    bounds=plan['source_box_xyxy']
    rgb=np.array(Image.open(original).convert('RGB').crop(bounds));h,w=rgb.shape[:2]
    assert np.array_equal(rgb,np.array(Image.open(REVIEW/'source_native.png')))
    # Only the geometry calculation is padded to a multiple of 4, preserving
    # exact native coordinates under BOX downsampling. Original crop unchanged.
    g=np.pad(rgb[:,:,1],((0,(-h)%4),(0,(-w)%4)))
    green=np.array(Image.fromarray(g).resize((g.shape[1]//4,g.shape[0]//4),Image.Resampling.BOX),dtype=float)
    foreground=ndi.binary_fill_holes(ndi.binary_closing(ndi.gaussian_filter(green,1)>16,iterations=2))
    labels,n=ndi.label(foreground);eligible=[]
    for k in range(1,n+1):
        yx=np.argwhere(labels==k)
        if yx[:,1].min()<=4 and yx[:,1].max()>=green.shape[1]-5:eligible.append((len(yx),k,yx))
    if not eligible:raise ValueError('No fluorescence component spans the selected source crop; inspect before any inference.')
    _,k,yx=max(eligible,key=lambda z:z[0]);xmin,xmax=yx[:,1].min(),yx[:,1].max()
    start=np.array([xmin,np.median(yx[yx[:,1]<=xmin+2,0])])
    stop=np.array([xmax,np.median(yx[yx[:,1]>=xmax-2,0])])
    raw,ends=graph_route(labels==k,start,stop);raw=raw*4+1.5
    arc0=np.r_[0,np.cumsum(np.linalg.norm(np.diff(raw,axis=0),axis=1))]
    positions=np.r_[np.arange(0,arc0[-1],4),arc0[-1]]
    path=ndi.gaussian_filter1d(np.c_[np.interp(positions,arc0,raw[:,0]),np.interp(positions,arc0,raw[:,1])],5,axis=0,mode='nearest')
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    assert np.isfinite(path).all() and (np.diff(path[:,0])>=-3).all()
    assert np.max(np.linalg.norm(np.diff(path,axis=0),axis=1))<12
    Image.fromarray(rgb).save(OUT/'image.png')
    save(OUT/'image_only_route.json',{
        'created_utc':now(),'source_image':str(original),'source_sha256':sha(original),
        'source_crop_xyxy':bounds,'source_size_xy':[w,h],
        'path_xy_crop_pixels':path.tolist(),'distance_along_path_pixels':arc.tolist(),
        'source_only_geometry':True,'tissue_labels_used':False,
        'settings':{'downsample':4,'green_threshold':16,'gaussian_sigma_coarse':1,
                    'closing_iterations_coarse':2,'resample_px':4,'path_smoothing_sigma_samples':5},
        'geometry_bottom_right_zero_padding_yx':[(-h)%4,(-w)%4],
        'endpoint_distances_coarse_pixels':ends,
        'endpoints':'Local source-field endpoints, not biological L/J.',
        'limits':'Rectangular fields can include adjacent branch/background pixels.'})
    expert={'recorded_utc':now(),'expert':'Gonzalo Villarino',
            'exact_statement':'A and B endoderm C Transition zone. Very bluerry image tho',
            'image_quality':'Expert reports very blurry image; precision limited.',
            'fields':[dict(field,biological_label=label,appearance_state=state)
                      for field,label,state in zip(plan['review_fields'],['Endoderm','Endoderm','Mixed / transition'],[0,0,1])],
            'exact_T_left_edge':None,'clear_Periderm_reviewed':False,
            'scope':'Qualitative main-root field identities; not pixel annotations, exact T, or a whole-image label.',
            'used_for_training':False}
    save(REVIEW/'expert_appearance_review_01.json',expert)
    save(OUT/'expert_appearance_review_01.json',expert)
    frozen=read(CANDIDATE/'candidate_manifest.json')
    save(OUT/'check_plan.json',{
        'created_utc':now(),'source_review':str(REVIEW/'source_review_plan.json'),
        'expert_review_sha256':sha(OUT/'expert_appearance_review_01.json'),
        'frozen_head_sha256':frozen['head_sha256'],'frozen_encoder_sha256':frozen['encoder_sha256'],
        'scan':'Same 320 x 256 native path-aligned fields every 64 path pixels; identical standardize and fixed boundary_rule.',
        'comparison':'After prediction, compare the nearest scan field to each reviewed field centre. The geometric fields differ slightly; show exact processed input for inspection. No pixel accuracy or precise T error.',
        'batch_size':8,'new_training':False,'threshold_changes':False,
        'test_scope':'Familiar A04 development slide; different from local COL0 appearance-head teaching, not blind final validation.',
        'calibration':'A04 not confirmed; do not calculate micrometre lengths.',
        'quality_handling':'Retain expert blur flag; do not sharpen or exclude based on model output.'})
    overview=Image.fromarray(rgb).resize((1720,420),Image.Resampling.LANCZOS)
    traced=overview.copy();d=ImageDraw.Draw(traced)
    d.line([(x*.4,y*.4) for x,y in path],fill='white',width=2)
    fig=Image.new('RGB',(1760,1010),'white');d=ImageDraw.Draw(fig)
    d.text((20,15),'A04 root 3: main-root path check',font=font(30),fill='black')
    d.text((20,58),'Top: unchanged source. Bottom: fluorescence-derived path. No tissue labels used.',font=font(25),fill='#334155')
    fig.paste(overview,(20,100));fig.paste(traced,(20,555));fig.save(OUT/'PREPARED_MAIN_ROOT_PATH.png')
    for name in [Path(__file__).name,'appearance_experiment.py','extend_root3_reference_route.py','test_both_appearances.py']:
        shutil.copy2(Path(__file__).with_name(name),OUT/name)
    print('Prepared route and saved expert review. No prediction or training.',flush=True)

def predict():
    assert not (OUT/'prediction_before_comparison.json').exists()
    route=read(OUT/'image_only_route.json');manifest=read(CANDIDATE/'candidate_manifest.json')
    head_file=Path(manifest['head']);encoder_file=Path(manifest['encoder'])
    assert sha(head_file)==manifest['head_sha256'] and sha(encoder_file)==manifest['encoder_sha256']
    protected=[Path(route['source_image']),head_file,encoder_file,ROOT/'06_Models/working_model.pt',OUT/'expert_appearance_review_01.json']
    before={str(p):sha(p) for p in protected}
    rgb=np.array(Image.open(OUT/'image.png'));path=np.array(route['path_xy_crop_pixels']);arc=np.array(route['distance_along_path_pixels'])
    distance=np.arange(160,arc[-1]-160+1,64)
    centers=np.c_[np.interp(distance,arc,path[:,0]),np.interp(distance,arc,path[:,1])]
    prev=np.c_[np.interp(distance-80,arc,path[:,0]),np.interp(distance-80,arc,path[:,1])]
    nxt=np.c_[np.interp(distance+80,arc,path[:,0]),np.interp(distance+80,arc,path[:,1])]
    tangent=nxt-prev;tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    u,v=np.meshgrid(np.arange(320)-160+.5,np.arange(256)-128+.5)
    windows=[];transforms=[];folder=OUT/'windows';folder.mkdir(exist_ok=False)
    for i,(center,unit) in enumerate(zip(centers,tangent)):
        nx=center[0]+u*unit[0]-v*unit[1];ny=center[1]+u*unit[1]+v*unit[0]
        raw=np.stack([ndi.map_coordinates(rgb[:,:,c],[ny,nx],order=1,mode='constant',cval=0) for c in range(3)],axis=2)
        image,transform=standardize(raw);windows.append(image);transforms.append(transform)
        Image.fromarray(image).save(folder/f'window_{i:03d}.png')
    torch.set_num_threads(4);device='mps' if torch.backends.mps.is_available() else 'cpu'
    encoder=smp.encoders.get_encoder('resnet34',in_channels=3,depth=5,weights=None)
    state=torch.load(encoder_file,map_location='cpu',weights_only=True)
    encoder.load_state_dict({k.removeprefix('encoder.'):v for k,v in state.items() if k.startswith('encoder.')},strict=True)
    encoder.to(device).eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    features=[]
    with torch.inference_mode():
        for a in range(0,len(windows),8):
            x=torch.from_numpy(np.stack(windows[a:a+8]).transpose(0,3,1,2).copy()).float().to(device)/255
            layers=encoder((x-mean)/std)[1:]
            features.append(F.normalize(torch.cat([F.normalize(z.mean((2,3)),dim=1) for z in layers],dim=1),dim=1).cpu())
        features=torch.cat(features)
        head=torch.nn.Linear(features.shape[1],2);head.load_state_dict(torch.load(head_file,map_location='cpu',weights_only=True),strict=True)
        head.eval().requires_grad_(False);scores=head(features).sigmoid().numpy()
    assert np.isfinite(scores).all()
    check_rule();rule=boundary_rule(scores,distance)
    t=None if rule['selected'] is None else rule['selected']['T_distance_pixels']
    txy=None if t is None else [float(np.interp(t,arc,path[:,j])) for j in range(2)]
    save(OUT/'prediction_before_comparison.json',{
        'created_utc':now(),'image_sha256':sha(OUT/'image.png'),
        'head_sha256':sha(head_file),'encoder_sha256':sha(encoder_file),'device':device,
        'source_crop_xyxy':route['source_crop_xyxy'],'centers_xy_crop_pixels':centers.tolist(),
        'distance_along_local_path':distance.tolist(),'presence_scores':scores.tolist(),
        'score_order':['Endoderm_like','Periderm_like'],'rule':rule,
        'candidate_T_distance_pixels':t,'candidate_T_crop_xy':txy,
        'reference_T_input':False,'expert_labels_used_for_inference':False,
        'training_occurred':False,'weights_or_threshold_changed':False,
        'window_size':[320,256],'stride':64,'batch_size':8})
    save(OUT/'window_preprocessing.json',transforms)
    np.savez_compressed(OUT/'frozen_features_and_scores.npz',features=features.numpy(),scores=scores)
    assert before=={str(p):sha(p) for p in protected}
    save(OUT/'verification.json',{'protected_files_unchanged':before,'no_training':True,'model_and_threshold_unchanged':True})
    print(json.dumps({'windows':len(distance),'state_counts':{name:rule['window_states'].count(i) for i,name in enumerate(NAMES)},'T_crop_xy':txy},indent=2),flush=True)

def compare():
    assert not (OUT/'comparison_result.json').exists()
    pred=read(OUT/'prediction_before_comparison.json');prediction_hash=sha(OUT/'prediction_before_comparison.json')
    expert=read(OUT/'expert_appearance_review_01.json');route=read(OUT/'image_only_route.json')
    centers=np.array(pred['centers_xy_crop_pixels']);offset=np.array(route['source_crop_xyxy'][:2])
    states=pred['rule']['window_states'];comparisons=[]
    for f in expert['fields']:
        box=np.array(f['source_box_xyxy']);center=(box[:2]+box[2:])/2
        j=int(np.argmin(np.linalg.norm(centers+offset-center,axis=1)))
        comparisons.append({'field':f['name'],'expert_appearance':f['biological_label'],
            'expert_state':f['appearance_state'],'nearest_scan_index':j,
            'scan_center_source_xy':(centers[j]+offset).tolist(),
            'distance_from_review_field_center_pixels':float(np.linalg.norm(centers[j]+offset-center)),
            'model_appearance':NAMES[states[j]],'model_state':states[j],
            'presence_scores':pred['presence_scores'][j],'qualitative_agreement':states[j]==f['appearance_state'],
            'scope':'Nearby path-aligned field; not an identical field or pixel accuracy.'})
    result={'created_utc':now(),'expert_review':expert,'local_field_comparison':comparisons,
            'candidate_T_crop_xy':pred['candidate_T_crop_xy'],
            'candidate_T_source_xy':None if pred['candidate_T_crop_xy'] is None else (np.array(pred['candidate_T_crop_xy'])+offset).tolist(),
            'no_precise_reference_T':True,'no_T_distance_error_or_length_claim':True,
            'model_updated':False,'blur_note':'Expert reports blur; causal explanation for model errors not established.',
            'limits':'Three reviewed fields on one familiar slide; do not report overall accuracy. Published PAT pretraining exposure unknown.'}
    save(OUT/'comparison_result.json',result)
    # Scientific figure from unchanged microscopy, exact processed model fields and outputs.
    fig=Image.new('RGB',(1760,1160),'white');d=ImageDraw.Draw(fig)
    d.text((20,14),'A04 root 3: unchanged detector versus your review',font=font(32),fill='black')
    d.text((20,60),'Your review: A = Endoderm, B = Endoderm, C = transition. Image quality: blurry.',font=font(25),fill='#334155')
    photo=Image.open(OUT/'image.png').resize((1720,420),Image.Resampling.LANCZOS);fig.paste(photo,(20,105));d=ImageDraw.Draw(fig)
    for f in expert['fields']:
        box=np.array(f['source_box_xyxy']);x=(box[0]+box[2])/2-offset[0]
        d.text((20+x*.4-10,113),f['name'],font=font(27),fill='white')
    if pred['candidate_T_crop_xy'] is not None:
        tx=20+pred['candidate_T_crop_xy'][0]*.4
        d.line((tx,105,tx,525),fill='#28b7ff',width=4)
        d.text((tx+8,150),'Model T',font=font(24),fill='#28b7ff',stroke_width=1,stroke_fill='black')
    d.text((20,550),'Model appearance along the root (overlapping fields):',font=font(25),fill='black')
    for c,s in zip(centers,states):
        x=20+c[0]*.4;d.rectangle((x-12,591,x+12,621),fill=COLORS[s])
    d.text((20,640),'Green: Endoderm only   Purple: mixed   Red: Periderm only   Grey: neither supported',font=font(25),fill='black')
    for i,c in enumerate(comparisons):
        x=20+580*i
        d.text((x,687),f'{c["field"]}: you = {c["expert_appearance"]}',font=font(25),fill='black')
        d.text((x,724),f'Model nearby = {c["model_appearance"]}',font=font(25),fill=COLORS[c['model_state']])
        photo=Image.open(OUT/'windows'/f'window_{c["nearest_scan_index"]:03d}.png')
        fig.paste(photo.resize((400,320),Image.Resampling.NEAREST),(x+75,767))
    d=ImageDraw.Draw(fig)
    note='No model T: the fixed rule did not find the required Endoderm -> mixed -> Periderm sequence.' if pred['candidate_T_crop_xy'] is None else 'Blue = model T candidate. No exact reference T was given, so its location still needs biological review.'
    d.text((20,1100),note,font=font(23),fill='#075985')
    d.text((20,1134),'Bottom: exact processed model inputs nearest A/B/C; slightly different fields from your original review. No training.',font=font(22),fill='#334155')
    fig.save(OUT/'A04_ROOT3_FIXED_RESULT.png')
    assert sha(OUT/'prediction_before_comparison.json')==prediction_hash
    lines=['A04 ROOT 3: UNCHANGED APPEARANCE DETECTOR','',expert['exact_statement'],'',
           'No new teaching, weights, threshold or decision-rule changes. Predictions saved before comparison.',
           'Only the nearest scan window to each reviewed field is compared; geometric framing differs slightly.']
    lines += [f"{c['field']}: expert {c['expert_appearance']}; model nearby {c['model_appearance']}." for c in comparisons]
    lines += ['',note,'Blur is recorded, not proven to cause any discrepancy. No exact T or new length accuracy is established.',
              'This slide has been inspected in earlier development; it is not a fresh blind test. Original PAT exposure unknown.',
              'All original images, labels and model weights unchanged. Word manual not edited.']
    (OUT/'READ_ME.txt').write_text('\n'.join(lines)+'\n')
    with (ROOT/'05_Notes/03_Progress_and_Next_Step.txt').open('a') as f:
        f.write('\n16 November 2026 — A04 root3 expert review and frozen check\n'+' '.join(lines[2:])+'\nResults: 04_Results/67_Fixed_A04_Root3_Check.\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','predict','compare'],required=True)
    parser.add_argument('--output',type=Path,default=OUT,help='Use a new directory for a repeat; original results are never overwritten.')
    args=parser.parse_args();OUT=args.output.resolve()
    {'prepare':prepare,'predict':predict,'compare':compare}[args.phase]()
