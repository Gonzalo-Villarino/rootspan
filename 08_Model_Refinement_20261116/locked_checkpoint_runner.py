"""Read-only transition checkpoint runner: image + reviewed main-root path -> T."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse
import json
import platform
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from frozen_core import standardize,continuity_rule
from verify_checkpoint import verify

BUNDLE=Path(__file__).resolve().parent
Image.MAX_IMAGE_PIXELS=300_000_000

def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def predict(image_file,route_file):
    config=json.loads((BUNDLE/'settings.json').read_text())
    route=json.loads(Path(route_file).read_text());rgb=np.array(Image.open(image_file).convert('RGB'))
    path=np.array(route['path_xy_crop_pixels'],dtype=float);arc=np.array(route['distance_along_path_pixels'],dtype=float)
    if path.ndim!=2 or path.shape!=(len(arc),2) or len(arc)<2:raise ValueError('Invalid main-root path.')
    if not np.isfinite(path).all() or not np.isfinite(arc).all() or (np.diff(arc)<=0).any():raise ValueError('Path must be finite and ordered left to right.')
    if not np.allclose(arc,np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))],atol=1e-5):raise ValueError('Arc distance differs from path geometry.')
    if np.any(path<0) or np.any(path[:,0]>=rgb.shape[1]) or np.any(path[:,1]>=rgb.shape[0]):raise ValueError('Path outside image; use crop coordinates.')
    if 'source_size_xy' in route and route['source_size_xy']!=[rgb.shape[1],rgb.shape[0]]:raise ValueError('Image dimensions differ from supplied route.')
    distance=np.arange(160,arc[-1]-160+1,config['stride_pixels'])
    if len(distance)<9:raise ValueError('Too little path context for the fixed three-state rule.')
    centres=np.c_[np.interp(distance,arc,path[:,0]),np.interp(distance,arc,path[:,1])]
    prev=np.c_[np.interp(distance-80,arc,path[:,0]),np.interp(distance-80,arc,path[:,1])]
    nxt=np.c_[np.interp(distance+80,arc,path[:,0]),np.interp(distance+80,arc,path[:,1])]
    tangent=nxt-prev;tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    u,v=np.meshgrid(np.arange(320)-160+.5,np.arange(256)-128+.5)
    images=[]
    for centre,unit in zip(centres,tangent):
        nx=centre[0]+u*unit[0]-v*unit[1];ny=centre[1]+u*unit[1]+v*unit[0]
        raw=np.stack([ndi.map_coordinates(rgb[:,:,c],[ny,nx],order=1,mode='constant',cval=0) for c in range(3)],axis=2)
        image,_=standardize(raw);images.append(image)
    torch.set_num_threads(config['cpu_threads'])
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    encoder=smp.encoders.get_encoder('resnet34',in_channels=3,depth=5,weights=None)
    state=torch.load(BUNDLE/'weights/pat_state_dict.pt',map_location='cpu',weights_only=True)
    encoder.load_state_dict({k.removeprefix('encoder.'):v for k,v in state.items() if k.startswith('encoder.')},strict=True)
    encoder.to(device).eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    features=[]
    with torch.inference_mode():
        for a in range(0,len(images),config['batch_size']):
            x=torch.from_numpy(np.stack(images[a:a+config['batch_size']]).transpose(0,3,1,2).copy()).float().to(device)/255
            levels=encoder((x-mean)/std)[1:]
            features.append(F.normalize(torch.cat([F.normalize(z.mean((2,3)),dim=1) for z in levels],dim=1),dim=1).cpu())
        features=torch.cat(features)
        head=torch.nn.Linear(features.shape[1],2)
        head.load_state_dict(torch.load(BUNDLE/'weights/presence_head.pt',map_location='cpu',weights_only=True),strict=True)
        head.eval().requires_grad_(False);scores=head(features).sigmoid().numpy()
    rule=continuity_rule(scores,distance)
    t=None if rule['selected'] is None else rule['selected']['T_distance_pixels']
    xy=None if t is None else [float(np.interp(t,arc,path[:,j])) for j in range(2)]
    offset=np.array(route.get('source_crop_xyxy',[0,0])[:2])
    return {'checkpoint':config['version'],'device':device,'python':platform.python_version(),
        'image':str(Path(image_file).resolve()),'route':str(Path(route_file).resolve()),
        'distance_along_local_path':distance.tolist(),'centres_xy_crop_pixels':centres.tolist(),
        'presence_scores':scores.tolist(),'score_order':['Endoderm_like','Periderm_like'],
        'rule':rule,'T_path_pixels':t,'T_crop_xy':xy,
        'T_source_xy':None if xy is None else (np.array(xy)+offset).tolist(),
        'training_occurred':False,'full_root_length_measurement':False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-check',action='store_true');p.add_argument('--image',type=Path);p.add_argument('--route',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    verify(BUNDLE)
    if BUNDLE==args.output.resolve() or BUNDLE in args.output.resolve().parents:raise ValueError('Write results outside the locked checkpoint.')
    if args.self_check:
        if args.image or args.route:raise ValueError('Self-check uses bundled reference inputs.')
        image=BUNDLE/'fixtures/212_image.png';route=BUNDLE/'fixtures/212_route.json'
    else:
        if args.image is None or args.route is None:p.error('Supply --image and --route, or --self-check.')
        image,route=args.image,args.route
    args.output.mkdir(parents=True,exist_ok=False)
    result=predict(image,route);save(args.output/'prediction.json',result)
    if args.self_check:
        expected=json.loads((BUNDLE/'fixtures/212_original_prediction.json').read_text())
        reference=json.loads((BUNDLE/'fixtures/212_approved_decoder.json').read_text())
        difference=float(np.max(np.abs(np.array(result['presence_scores'])-np.array(expected['presence_scores']))))
        if difference>1e-5:raise AssertionError(f'Image-score replay changed by {difference}; inspect runtime differences.')
        assert result['T_path_pixels']==reference['new_T_path_pixels']
        assert np.allclose(result['T_source_xy'],reference['new_T_source_xy'],atol=1e-6)
        checks=[]
        for f in sorted((BUNDLE/'fixtures/profiles').glob('*.json')):
            record=json.loads(f.read_text());q=record['prediction']
            scores=q.get('presence_scores',q.get('predicted_presence_scores'))
            distance=q.get('distance_along_local_path',q.get('distance_native_pixels'))
            rule=continuity_rule(scores,distance)
            t=None if rule['selected'] is None else rule['selected']['T_distance_pixels']
            assert t==record['expected_T_path_pixels'],f.name
            checks.append({'case':record['case'],'T_path_pixels':t,'passed':True})
        report={'passed':True,'image_to_scores_max_absolute_difference':difference,
                'approved_212_T_reproduced':result['T_source_xy'],'saved_profile_replays':checks,
                'scope':'Computational reproduction, not new biological validation.'}
        save(args.output/'self_check.json',report);print(json.dumps(report,indent=2))
    else:print(json.dumps({'T_source_xy':result['T_source_xy'],'output':str(args.output)},indent=2))

if __name__=='__main__':main()
