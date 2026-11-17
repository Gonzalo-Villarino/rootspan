from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image
from predict import load,predict,PAL
from preprocess import prepare
from check_reference import metrics
ROOT=Path(__file__).resolve().parents[1]
def tiled(model,device,rgb,size=512,stride=256):
 h,w=rgb.shape[:2]
 def starts(n):return sorted(set(list(range(0,max(1,n-size+1),stride))+[max(0,n-size)]))
 total=np.zeros((4,h,w),dtype='float32');den=np.zeros((h,w),dtype='float32')
 for y in starts(h):
  for x in starts(w):
   sub=rgb[y:y+size,x:x+size];hh,ww=sub.shape[:2];win=np.maximum(np.hanning(hh)[:,None]*np.hanning(ww)[None,:],.03).astype('float32');total[:,y:y+hh,x:x+ww]+=predict(model,device,sub)*win;den[y:y+hh,x:x+ww]+=win
 return total/den
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--folder',type=Path,required=True);a.add_argument('--final',action='store_true');args=a.parse_args();plan=json.loads((args.folder/'experiment_plan.json').read_text());arch='small_unet' if 'SmallUNet' in plan.get('architecture','') else 'pat';name='final_weights.pt' if args.final else 'best_weights.pt';model,device=load(args.folder/name,arch);rgb=np.array(Image.open(ROOT/'04_Results/52_Reference_Check_20261116/image.png'));truth=np.array(Image.open(ROOT/'04_Results/52_Reference_Check_20261116/reference_mask.png'));pre=prepare(rgb,plan.get('contrast_preprocessing','none'));p=tiled(model,device,pre);pred=(p.argmax(0)+1).astype('uint8');suffix='final_tiled' if args.final else 'best_tiled';Image.fromarray(pred).save(args.folder/f'{suffix}_prediction.png');np.savez_compressed(args.folder/f'{suffix}_probabilities.npz',probabilities=p);met=metrics(pred,truth);(args.folder/f'{suffix}_metrics.json').write_text(json.dumps(met,indent=2)+'\n');print(json.dumps(met,indent=2))
