"""Image-only flip/scale ensemble; human mask is read only after inference for development scoring."""
from pathlib import Path
import json,argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from predict import load,predict,PAL
from preprocess import prepare
from check_reference import metrics
from tile_check import tiled
ROOT=Path(__file__).resolve().parents[1]
def ensemble(m,d,rgb,size=512,stride=256,flips=True):
 results=[]
 for axes in ([(),(0,),(1,),(0,1)] if flips else [()]):
  a=np.flip(rgb,axes).copy() if axes else rgb
  p=tiled(m,d,a,size,stride)
  if axes:p=np.flip(p,tuple(i+1 for i in axes)).copy()
  results.append(p)
 return np.mean(results,axis=0)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--folder',type=Path,required=True);args=ap.parse_args();folder=args.folder;out=folder/'inference_ensemble';out.mkdir(exist_ok=False)
 plan=json.loads((folder/'experiment_plan.json').read_text());arch='small_unet' if 'SmallUNet' in plan.get('architecture','') else 'pat';m,d=load(folder/'best_weights.pt',arch)
 rgb=np.array(Image.open(ROOT/'04_Results/52_Reference_Check_20261116/image.png'));truth=np.array(Image.open(ROOT/'04_Results/52_Reference_Check_20261116/reference_mask.png'));allp=[];record={}
 for scale in [.75,1.,1.25]:
  sub=np.array(Image.fromarray(rgb).resize((round(1024*scale),round(1024*scale)),Image.Resampling.BILINEAR))
  p=ensemble(m,d,prepare(sub,plan.get('contrast_preprocessing','none')))
  if scale!=1:p=F.interpolate(torch.from_numpy(p)[None],size=(1024,1024),mode='bilinear',align_corners=False)[0].numpy()
  allp.append(p);name=f'scale_{scale}';pred=(p.argmax(0)+1).astype('uint8');met=metrics(pred,truth);record[name]=met;np.savez_compressed(out/f'{name}_probabilities.npz',probabilities=p);Image.fromarray(pred).save(out/f'{name}_prediction.png');print(name,met['tissue_mean_iou'],[met['per_class'][c]['recall'] for c in ['Periderm','Endoderm','Background']],flush=True)
 p=np.mean(allp,axis=0);pred=(p.argmax(0)+1).astype('uint8');record['scale_average']=metrics(pred,truth);np.savez_compressed(out/'scale_average_probabilities.npz',probabilities=p);Image.fromarray(pred).save(out/'scale_average_prediction.png');Image.fromarray(PAL[pred]).save(out/'scale_average_colours.png');(out/'metrics.json').write_text(json.dumps(record,indent=2)+'\n');(out/'method.json').write_text(json.dumps({'input':'image only','flips':['none','vertical','horizontal','both'],'scales':[.75,1.,1.25],'tile_size':512,'stride':256,'role':'Familiar development crop; settings compared with existing annotations. Not independent validation.'},indent=2)+'\n');print('scale_average',record['scale_average']['tissue_mean_iou'],flush=True)
if __name__=='__main__':main()
