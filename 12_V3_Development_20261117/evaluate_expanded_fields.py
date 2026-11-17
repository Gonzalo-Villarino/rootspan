"""Fixed one-shot 1_7 comparison. Manual references are read only after predictions."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
D=Path(__file__).resolve().parent;P=D.parent
B=P/'06_Models/Locked_Transition_v1_20261116';sys.path.insert(0,str(B))
from frozen_core import standardize,continuity_rule
from verify_checkpoint import verify

def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def project(xy,path,arc):
    v=np.diff(path,axis=0);u=np.clip(((xy-path[:-1])*v).sum(1)/np.maximum((v*v).sum(1),1e-12),0,1)
    q=path[:-1]+u[:,None]*v;i=np.argmin(np.linalg.norm(q-xy,axis=1))
    return float(arc[i]+u[i]*(arc[i+1]-arc[i]))

def main():
    assert json.loads((D/'expanded_fields_trial/gate.json').read_text())['passed']
    verify(B,require_read_only=True)
    out=D/'expanded_fields_one_seven';out.mkdir(exist_ok=False)
    base=P/'11_V21_Development_20261117/regression/1_7'
    cfg=json.loads((B/'settings.json').read_text());torch.set_num_threads(cfg['cpu_threads'])
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    encoder=smp.encoders.get_encoder('resnet34',in_channels=3,depth=5,weights=None)
    state=torch.load(B/'weights/pat_state_dict.pt',map_location='cpu',weights_only=True)
    encoder.load_state_dict({k.removeprefix('encoder.'):v for k,v in state.items() if k.startswith('encoder.')},strict=True)
    encoder.to(device).eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    heads={}
    for name,file in [('V1',B/'weights/presence_head.pt'),('expanded',D/'expanded_fields_trial/experimental_presence_head.pt')]:
        h=torch.nn.Linear(1024,2);h.load_state_dict(torch.load(file,map_location='cpu',weights_only=True));h.eval().requires_grad_(False);heads[name]=h
    inputs=[Path(__file__),D/'EXPANDED_FIELDS_PLAN.txt',D/'expanded_fields_trial/experimental_presence_head.pt']
    inputs += [base/f'track_{r:02d}'/f for r in [1,2,6] for f in ['image.png','route.json','locked_prediction.json']]
    save(out/'frozen_inputs_sha256.json',{str(f.relative_to(P)):sha(f) for f in inputs})
    predictions={}
    for r in [1,2,6]:
        folder=base/f'track_{r:02d}';route=json.loads((folder/'route.json').read_text())
        rgb=np.array(Image.open(folder/'image.png').convert('RGB'));path=np.array(route['path_xy_crop_pixels']);arc=np.array(route['distance_along_path_pixels'])
        distance=np.arange(160,arc[-1]-160+1,cfg['stride_pixels'])
        centres=np.c_[np.interp(distance,arc,path[:,0]),np.interp(distance,arc,path[:,1])]
        prev=np.c_[np.interp(distance-80,arc,path[:,0]),np.interp(distance-80,arc,path[:,1])]
        nxt=np.c_[np.interp(distance+80,arc,path[:,0]),np.interp(distance+80,arc,path[:,1])]
        tangent=nxt-prev;tangent/=np.linalg.norm(tangent,axis=1)[:,None]
        u,v=np.meshgrid(np.arange(320)-160+.5,np.arange(256)-128+.5);images=[]
        for c,unit in zip(centres,tangent):
            x=c[0]+u*unit[0]-v*unit[1];y=c[1]+u*unit[1]+v*unit[0]
            raw=np.stack([ndi.map_coordinates(rgb[:,:,k],[y,x],order=1,mode='constant',cval=0) for k in range(3)],axis=2)
            images.append(standardize(raw)[0])
        fs=[]
        with torch.inference_mode():
            for start in range(0,len(images),cfg['batch_size']):
                x=torch.from_numpy(np.stack(images[start:start+cfg['batch_size']]).transpose(0,3,1,2).copy()).float().to(device)/255
                levels=encoder((x-mean)/std)[1:]
                fs.append(F.normalize(torch.cat([F.normalize(z.mean((2,3)),dim=1) for z in levels],dim=1),dim=1).cpu())
            features=torch.cat(fs);outputs={}
            for name,h in heads.items():
                scores=h(features).sigmoid().numpy();rule=continuity_rule(scores,distance)
                outputs[name]={'scores':scores.tolist(),'T':None if rule['selected'] is None else rule['selected']['T_distance_pixels'],'rule':rule}
        old=json.loads((folder/'locked_prediction.json').read_text())
        delta=float(np.max(abs(np.array(outputs['V1']['scores'])-old['presence_scores'])))
        assert delta<=1e-5 and outputs['V1']['T']==old['T_path_pixels']
        np.savez_compressed(out/f'root{r}_features.npz',features=features.numpy(),distance=distance)
        predictions[r]={'root':r,'outputs':outputs,'distance':distance.tolist(),'V1_replay_max_score_difference':delta,'reference_input':False}
        save(out/f'root{r}_predictions_before_comparison.json',predictions[r]);print('Saved both predictions for root',r,flush=True)
    # Reference points are used only now, after all three immutable prediction files exist.
    refpath=P/'04_Results/73_V1_Manual_Comparison_Pilot/01_Manual_References/manual_reference_index.json'
    refs=json.loads(refpath.read_text())['records'];rows=[]
    for ref in refs:
        r=ref['root_id'];q=predictions[r];route=json.loads((base/f'track_{r:02d}'/'route.json').read_text())
        path=np.array(route['path_xy_crop_pixels'])+route['source_crop_xyxy'][:2];arc=np.array(route['distance_along_path_pixels']);pts=np.array(ref['points_xy'])
        tp=project(pts[np.argmin(pts[:,0])],path,arc);j=route['endpoint']['measurement_distance_pixels'];mid=j/2
        old=q['outputs']['V1']['T'];new=q['outputs']['expanded']['T'];olde=abs(old-tp);newe=None if new is None else abs(new-tp)
        row={'root':r,'manual_T_arc_pixels':tp,'V1_T_arc_pixels':old,'expanded_T_arc_pixels':new,'V1_absolute_T_error_um':olde/.5299,'expanded_absolute_T_error_um':None if new is None else newe/.5299,'midpoint_absolute_T_error_um':abs(mid-tp)/.5299,'absolute_T_error_reduction_pixels':None if new is None else olde-newe,'passed':new is not None and olde-newe>64 and newe<abs(mid-tp),'manual_length_um':ref['length_pixels']/.5299,'fixed_endpoint_arc_pixels':j,'V1_fixed_endpoint_length_um':(j-old)/.5299,'expanded_fixed_endpoint_length_um':None if new is None else (j-new)/.5299,'endpoint_not_validated':True}
        rows.append(row)
    result={'passed':all(r['passed'] for r in rows),'rows':rows,'reference_sha256':sha(refpath),'independent_validation':False,'reserved_15_27_used':False,'all_predictions_saved_before_reference_comparison':True,'promoted':False}
    save(out/'comparison.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
