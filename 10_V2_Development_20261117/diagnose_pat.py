"""Compare the released PAT pixel classifier with V1 on saved image-only routes.

Diagnostic only: does not train, invent annotations, or change the GUI/V1.
Samples native 1024-pixel fields, scores only a narrow main-root centre band.
"""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
import torch
import segmentation_models_pytorch as smp

ROOT=Path(__file__).resolve().parents[1]
Image.MAX_IMAGE_PIXELS=200_000_000

def sample(rgb,path,arc,s,size=1024):
    c=np.array([np.interp(s,arc,path[:,k]) for k in range(2)])
    v=np.array([np.interp(s+80,arc,path[:,k])-np.interp(s-80,arc,path[:,k]) for k in range(2)])
    v/=np.linalg.norm(v)
    u,w=np.meshgrid(np.arange(size)-size/2+.5,np.arange(size)-size/2+.5)
    x=c[0]+u*v[0]-w*v[1];y=c[1]+u*v[1]+w*v[0]
    # Restrict source before interpolation; avoid per-window copying full slide.
    x0=max(0,int(x.min())-1);y0=max(0,int(y.min())-1)
    x1=min(rgb.width,int(x.max())+3);y1=min(rgb.height,int(y.max())+3)
    a=np.array(rgb.crop((x0,y0,x1,y1)))
    return np.stack([map_coordinates(a[:,:,k],[y-y0,x-x0],order=1,mode='constant') for k in range(3)],axis=2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);device='mps' if torch.backends.mps.is_available() else 'cpu'
    weights=ROOT/'06_Models/Locked_Transition_v1_20261116/weights/pat_state_dict.pt'
    model=smp.UnetPlusPlus(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=4,activation='sigmoid')
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    model.to(device).eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    rgb=Image.open(a.source).convert('RGB');reports=[]
    for folder in sorted(a.baseline.glob('track_*')):
        if not (folder/'route.json').exists():continue
        r=json.loads((folder/'route.json').read_text());path=np.array(r['path_xy_crop_pixels'])+r['source_crop_xyxy'][:2];arc=np.array(r['distance_along_path_pixels']);distance=np.arange(160,arc[-1]-160,256)
        records=[]
        for s in distance:
            im=sample(rgb,path,arc,s)
            x=torch.from_numpy(im.transpose(2,0,1).copy()).float().to(device)[None]/255
            with torch.inference_mode():pr=model((x-mean)/std)[0,:,488:536,384:640].cpu().numpy()
            labels=pr.argmax(0);valid=(labels==1)|(labels==2)
            per=float((labels==1).sum()/max(1,valid.sum()));end=float((labels==2).sum()/max(1,valid.sum()))
            records.append({'s':float(s),'periderm_fraction':per,'endoderm_fraction':end,'tissue_fraction':float(valid.mean()),'mean_scores_B_P_E_LR':pr.mean((1,2)).tolist()})
        record={'track':folder.name,'diagnostic_only':True,'model':'released PAT 4-class pixel decoder','weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'training':False,'source':str(a.source.resolve()),'baseline':str(folder.resolve()),'records':records}
        (a.output/(folder.name+'.json')).write_text(json.dumps(record,indent=2)+'\n');reports.append(record)
        print(folder.name, 'P fractions at 15/40/65/82/94%:',[round(records[np.argmin(abs(distance-f*arc[-1]))]['periderm_fraction'],2) for f in [.15,.4,.65,.82,.94]],flush=True)
    print('Saved diagnostic only:',a.output,flush=True)
if __name__=='__main__':main()
