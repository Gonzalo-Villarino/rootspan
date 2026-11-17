"""Fixed 20_29 check: image route, unchanged appearance inference, then reference.

No training. The reference mask is read only by compare(), after prediction.
The route uses fluorescence geometry; it is not a predicted tissue boundary.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import hashlib
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

CANDIDATE=ROOT/'06_Models/20261116_Two_Appearance_Candidate'
DEFAULT_OUT=ROOT/'04_Results/66_Fixed_20_29_Check'


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(p,obj):
    p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def load_record(p):return json.loads(p.read_text())


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    plan=load_record(CANDIDATE/'next_check_plan.json')
    source=Path(plan['input_image']);assert sha(source)==plan['input_image_sha256']
    rgb=np.array(Image.open(source).convert('RGB'));h,w=rgb.shape[:2]
    # Choose the connected object spanning the two crop edges, using only image
    # fluorescence. Lateral branches may remain attached to this component;
    # the shortest path goes between main-root edge locations, not down a branch.
    factor=4;offset=(factor-1)/2
    small=np.array(Image.fromarray(rgb[:,:,1]).resize((w//factor,h//factor),Image.Resampling.BOX),dtype=float)
    smooth=ndi.gaussian_filter(small,1)
    foreground=ndi.binary_fill_holes(ndi.binary_closing(smooth>16,iterations=2))
    components,n=ndi.label(foreground);eligible=[]
    for k in range(1,n+1):
        yx=np.argwhere(components==k)
        if yx[:,1].min()<=4 and yx[:,1].max()>=small.shape[1]-5:
            eligible.append((len(yx),k,yx))
    if not eligible:raise ValueError('No image-only component spans the crop. Inspect before proceeding; do not use tissue labels to force a route.')
    _,k,yx=max(eligible,key=lambda x:x[0]);component=components==k
    xmin,xmax=int(yx[:,1].min()),int(yx[:,1].max())
    start=np.array([xmin,float(np.median(yx[yx[:,1]<=xmin+2,0]))])
    stop=np.array([xmax,float(np.median(yx[yx[:,1]>=xmax-2,0]))])
    raw,ends=graph_route(component,start,stop);raw=raw*factor+offset
    arc0=np.r_[0,np.cumsum(np.linalg.norm(np.diff(raw,axis=0),axis=1))]
    positions=np.r_[np.arange(0,arc0[-1],4),arc0[-1]]
    path=np.c_[np.interp(positions,arc0,raw[:,0]),np.interp(positions,arc0,raw[:,1])]
    path=ndi.gaussian_filter1d(path,5,axis=0,mode='nearest')
    # No extrapolation to biological endpoints: this is a local image crop.
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    assert np.isfinite(path).all() and (np.diff(path[:,0])>=-3).all()
    assert np.max(np.linalg.norm(np.diff(path,axis=0),axis=1))<12
    route={'created_utc':datetime.now(timezone.utc).isoformat(),'source_image':str(source),
       'source_sha256':sha(source),'source_size_xy':[w,h],'path_xy_image_pixels':path.tolist(),
       'distance_along_path_pixels':arc.tolist(),'settings':{'downsample':4,'green_threshold':16,
       'gaussian_sigma_coarse':1,'binary_closing_iterations_coarse':2,'path_resample_pixels':4,
       'path_smoothing_sigma_samples':5},'endpoint_distances_coarse_pixels':ends,
       'source_only_geometry':True,'reference_mask_read':False,'tissue_labels_used':False,
       'endpoint_meaning':'Local main-root route inside crop; not biological cut end or hypocotyl junction.',
       'method':'Fluorescence threshold, connected component spanning crop, skeleton shortest path, smoothing.',
       'lateral_root_note':'Route avoids the downward branch. Rectangular inference windows can still contain surrounding branch pixels.'}
    save(out/'image_only_route.json',route)
    Image.fromarray(rgb).save(out/'image.png')
    display=Image.fromarray(rgb).crop((0,280,w,660)).resize((1280,475),Image.Resampling.LANCZOS)
    original=display.copy();d=ImageDraw.Draw(display)
    d.line([(float(x)*1280/w,(float(y)-280)*475/380) for x,y in path],fill='#00d7ff',width=3)
    fig=Image.new('RGB',(1320,1100),'white');d=ImageDraw.Draw(fig)
    d.text((20,14),'20_29 root 4: main-root route from the image',font=font(28),fill='black')
    d.text((20,56),'Top: source image. Bottom: traced path. No reference annotations used.',font=font(23),fill='#334155')
    fig.paste(original,(20,100));fig.paste(display,(20,610))
    fig.save(out/'PREPARED_MAIN_ROOT_PATH.png')
    shutil.copy2(__file__,out/Path(__file__).name)
    for name in ['appearance_experiment.py','test_both_appearances.py','extend_root3_reference_route.py']:
        shutil.copy2(Path(__file__).with_name(name),out/name)
    print('Prepared image-only main-root route: '+str(out/'PREPARED_MAIN_ROOT_PATH.png'),flush=True)


def predict(out):
    assert not (out/'prediction_before_reference_comparison.json').exists()
    manifest=load_record(CANDIDATE/'candidate_manifest.json')
    head_file=Path(manifest['head']);encoder_file=Path(manifest['encoder'])
    assert sha(head_file)==manifest['head_sha256'] and sha(encoder_file)==manifest['encoder_sha256']
    default_file=ROOT/'06_Models/working_model.pt';default_hash=sha(default_file)
    route=load_record(out/'image_only_route.json');source=Path(route['source_image'])
    assert sha(source)==route['source_sha256']
    rgb=np.array(Image.open(source).convert('RGB'));path=np.array(route['path_xy_image_pixels'])
    arc=np.array(route['distance_along_path_pixels']);distance=np.arange(160,arc[-1]-160+1,64)
    if len(distance)<9:raise ValueError('Insufficient path for the unchanged three-run support rule.')
    centers=np.c_[np.interp(distance,arc,path[:,0]),np.interp(distance,arc,path[:,1])]
    before=np.c_[np.interp(distance-80,arc,path[:,0]),np.interp(distance-80,arc,path[:,1])]
    after=np.c_[np.interp(distance+80,arc,path[:,0]),np.interp(distance+80,arc,path[:,1])]
    tangent=after-before;tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    u,v=np.meshgrid(np.arange(320)-160+.5,np.arange(256)-128+.5)
    windows=[];transforms=[];folder=out/'windows';folder.mkdir(exist_ok=False)
    for i,(center,unit) in enumerate(zip(centers,tangent)):
        nx=center[0]+u*unit[0]-v*unit[1];ny=center[1]+u*unit[1]+v*unit[0]
        raw=np.stack([ndi.map_coordinates(rgb[:,:,c],[ny,nx],order=1,mode='constant',cval=0) for c in range(3)],axis=2)
        image,transform=standardize(raw);windows.append(image);transforms.append(transform)
        Image.fromarray(image).save(folder/f'window_{i:02d}.png')
    torch.set_num_threads(4);device='mps' if torch.backends.mps.is_available() else 'cpu'
    encoder=smp.encoders.get_encoder('resnet34',in_channels=3,depth=5,weights=None)
    state=torch.load(encoder_file,map_location='cpu',weights_only=True)
    encoder.load_state_dict({k.removeprefix('encoder.'):v for k,v in state.items() if k.startswith('encoder.')},strict=True)
    encoder.to(device).eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    x=torch.from_numpy(np.stack(windows).transpose(0,3,1,2).copy()).float().to(device)/255
    with torch.inference_mode():
        layers=encoder((x-mean)/std)[1:]
        features=F.normalize(torch.cat([F.normalize(z.mean((2,3)),dim=1) for z in layers],dim=1),dim=1).cpu()
        head=torch.nn.Linear(features.shape[1],2);head.load_state_dict(torch.load(head_file,map_location='cpu',weights_only=True),strict=True)
        head.eval().requires_grad_(False);scores=head(features).sigmoid().numpy()
    assert np.isfinite(scores).all() and (scores>=0).all() and (scores<=1).all()
    check_rule();rule=boundary_rule(scores,distance)
    t=None if rule['selected'] is None else rule['selected']['T_distance_pixels']
    xy=None if t is None else [float(np.interp(t,arc,path[:,j])) for j in range(2)]
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'input_image':str(source),
       'image_sha256':sha(source),'candidate_manifest':str(CANDIDATE/'candidate_manifest.json'),
       'head_sha256':sha(head_file),'encoder_sha256':sha(encoder_file),'default_model_sha256':default_hash,
       'device':device,'image_only_prediction':True,'reference_labels_read':False,
       'reference_T_input':False,'training_occurred':False,'weights_changed':False,
       'settings_changed':False,'window_width_height':[320,256],'stride':64,
       'source_centers_xy':centers.tolist(),'distance_along_local_path':distance.tolist(),
       'presence_score_order':['Endoderm_like','Periderm_like'],'presence_scores':scores.tolist(),
       'rule':rule,'candidate_T_distance_pixels':t,'candidate_T_image_xy':xy,
       'scope':'Different image from local appearance-head teaching; familiar development image, not a new blind test. Public PAT pretraining exposure unknown.',
       'limits':'Rectangular windows may contain nearby lateral-root pixels; no individual-cell detector or full-root length measurement.'}
    save(out/'prediction_before_reference_comparison.json',record)
    save(out/'window_preprocessing.json',transforms)
    np.savez_compressed(out/'frozen_features_and_scores.npz',features=features.numpy(),scores=scores)
    assert sha(head_file)==manifest['head_sha256'] and sha(encoder_file)==manifest['encoder_sha256'] and sha(default_file)==default_hash
    print(json.dumps({'windows':len(distance),'window_states':rule['window_states'],
                      'candidate_T_image_xy':xy,'candidate_T_distance_pixels':t},indent=2),flush=True)


def compare(out):
    assert not (out/'comparison_result.json').exists()
    prediction_file=out/'prediction_before_reference_comparison.json'
    prediction=load_record(prediction_file);prediction_hash=sha(prediction_file)
    plan=load_record(CANDIDATE/'next_check_plan.json')
    mask_path=Path(plan['reference_mask']);assert sha(mask_path)==plan['reference_mask_sha256']
    # This is the first reference-mask read in this program.
    truth=np.array(Image.open(mask_path));rgb=np.array(Image.open(out/'image.png'))
    route=load_record(out/'image_only_route.json');path=np.array(route['path_xy_image_pixels']);arc=np.array(route['distance_along_path_pixels'])
    positions=np.arange(0,arc[-1],1)
    points=np.c_[np.interp(positions,arc,path[:,0]),np.interp(positions,arc,path[:,1])]
    ij=np.rint(points).astype(int);codes=truth[ij[:,1],ij[:,0]]
    e=positions[codes==2];p=positions[codes==1]
    monotonic=bool(len(e) and len(p) and e.max()<p.min())
    interval=None;reference=None;reference_xy=None
    if monotonic:
        interval=[float(e.max()),float(p.min())];reference=float(np.mean(interval))
        reference_xy=[float(np.interp(reference,arc,path[:,j])) for j in range(2)]
    t=prediction['candidate_T_distance_pixels']
    error=None if t is None or reference is None else t-reference
    state=np.array(prediction['rule']['window_states']);scores=np.array(prediction['presence_scores'])
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'image':'20_29 root4',
       'prediction_saved_before_reference_mask_read':True,'prediction_sha256':prediction_hash,
       'reference_mask':str(mask_path),'reference_mask_sha256':sha(mask_path),
       'reference_method':'Gap between last Endoderm-labelled path pixel and first Periderm-labelled path pixel, only if order is unambiguous. Midpoint is a working comparison, not an exact manually marked T.',
       'reference_class_order_unambiguous':monotonic,'reference_boundary_gap_along_path_pixels':interval,
       'reference_boundary_midpoint_along_path_pixels':reference,'reference_boundary_midpoint_image_xy':reference_xy,
       'candidate_T_along_path_pixels':t,'candidate_T_image_xy':prediction['candidate_T_image_xy'],
       'difference_from_outline_reference_midpoint_pixels':error,
       'reference_path_class_counts':{str(c):int((codes==c).sum()) for c in np.unique(codes)},
       'window_counts':{name:int((state==i).sum()) for i,name in enumerate(prediction['rule']['state_order'])},
       'diagnostic_runs':prediction['rule']['runs'],'new_training':False,'model_modified':False,
       'separate_image_check':'Local head was taught on COL0 only. 20_29 is familiar development; published-encoder pretraining membership unknown. Not a blind final test.',
       'not_a_full_root_length_measurement':True}
    save(out/'comparison_result.json',result)
    # Visual comparison: source crop, reference outlines, then raw window states.
    bounds=(0,280,1024,660);scale=1.25;photo=Image.fromarray(rgb).crop(bounds).resize((1280,475),Image.Resampling.LANCZOS)
    fig=Image.new('RGB',(1320,920),'white');d=ImageDraw.Draw(fig)
    d.text((20,15),'20_29 root 4: unchanged transition detector',font=font(29),fill='black')
    subtitle='No supported T returned by the fixed rule.' if t is None else 'Blue: model T. White: boundary inferred from your saved outlines.'
    d.text((20,58),subtitle,font=font(24),fill='#334155');fig.paste(photo,(20,105));d=ImageDraw.Draw(fig)
    if reference_xy is not None:
        x=20+reference_xy[0]*scale;d.line((x,105,x,580),fill='white',width=3);d.text((x+8,116),'Outline reference',font=font(22),fill='white',stroke_width=1,stroke_fill='black')
    if prediction['candidate_T_image_xy'] is not None:
        x=20+prediction['candidate_T_image_xy'][0]*scale;d.line((x,105,x,580),fill='#28b7ff',width=3)
        d.text((x+8,155),'Model T',font=font(22),fill='#28b7ff',stroke_width=1,stroke_fill='black')
    colors=['#1aa65e','#b44dc5','#db303f','#777777']
    d.text((20,604),'Predicted appearance at sampled positions:',font=font(24),fill='black')
    for xy,s in zip(prediction['source_centers_xy'],state):
        x=20+xy[0]*scale;d.rectangle((x-30,645,x+30,679),fill=colors[int(s)])
    d.text((20,698),'Green: Endoderm only   Purple: both / mixed   Red: Periderm only   Grey: neither',font=font(22),fill='black')
    if t is None:
        msg='The unchanged rule did not find the required E -> mixed -> P pattern.'
    elif error is not None:
        msg=f'Model T is {abs(error):.0f} path pixels {"right" if error>0 else "left"} of the outline-reference midpoint.'
    else:msg='A model candidate was returned, but the saved outlines do not define one unambiguous comparison boundary.'
    d.text((20,741),msg,font=font(24),fill='black')
    d.text((20,784),'No teaching or settings changes. This is a familiar development image, not a new blind test.',font=font(22),fill='#334155')
    d.text((20,825),'White is a reference from your outlines, not a model result. Source image and saved models unchanged.',font=font(21),fill='#334155')
    fig.save(out/'20_29_FIXED_MODEL_RESULT.png')
    # Full score plot retains the pattern behind a missing or accepted candidate.
    chart=Image.new('RGB',(1320,560),'white');d=ImageDraw.Draw(chart)
    d.text((20,15),'Why the fixed rule did or did not return T',font=font(29),fill='black')
    d.text((20,56),'Separate appearance scores; not calibrated confidence.',font=font(23),fill='#334155')
    x0,y0,w,h=90,115,1160,280;distance=np.array(prediction['distance_along_local_path'])
    xx=lambda a:x0+w*a/arc[-1];yy=lambda a:y0+h*(1-a)
    for level in [0,.5,1]:d.line((x0,yy(level),x0+w,yy(level)),fill='#bbbbbb');d.text((35,yy(level)-10),str(level),font=font(20),fill='black')
    for j,c in enumerate(['#1aa65e','#db303f']):d.line([(xx(s),yy(v)) for s,v in zip(distance,scores[:,j])],fill=c,width=3)
    if reference is not None:d.line((xx(reference),y0,xx(reference),y0+h),fill='black',width=2)
    if t is not None:d.line((xx(t),y0,xx(t),y0+h),fill='#0078bc',width=2)
    d.text((90,420),'Green: Endoderm-like appearance    Red: Periderm-like appearance',font=font(22),fill='black')
    d.text((90,461),'Mixed requires both scores >= 0.5 for at least 3 consecutive windows.',font=font(22),fill='black')
    d.text((90,502),'The rule also requires clear Endoderm before and clear Periderm after.',font=font(22),fill='black');chart.save(out/'20_29_PRESENCE_SCORES.png')
    assert sha(prediction_file)==prediction_hash
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','predict','compare','all'],default='all')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT);args=parser.parse_args();out=args.output.resolve()
    if args.phase in ['prepare','all']:prepare(out)
    if args.phase in ['predict','all']:predict(out)
    if args.phase in ['compare','all']:compare(out)


if __name__=='__main__':main()
