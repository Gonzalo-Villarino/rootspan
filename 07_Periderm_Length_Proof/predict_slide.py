# Uses original PAT methods/code: Copyright (c) 2023 Salk Harnessing Plants Initiative.
# Original MIT notice: 00_Software/PAT_Reference/LICENSE. See THIRD_PARTY_NOTICES.md.
"""Full-resolution slide inference, preserving source coordinates and all earlier results.

1024px model tiles with64px context removed on each edge;896px cores cover each
source pixel exactly once. This is a documented adaptation, not byte-identical
reproduction of PAT's original16px overlap and colour-channel-max stitching.
"""
from pathlib import Path
import argparse,hashlib,json,math,shutil,time
import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp
Image.MAX_IMAGE_PIXELS=250_000_000
ROOT=Path(__file__).resolve().parents[1]
WEIGHTS={'working3':ROOT/'04_Results/39_PAT_Transfer_Learning/run_20261115T140159Z/model_weights.pt','published4':ROOT/'00_Software/PAT_Reference/pat_state_dict.pt'}
EXPECTED={'working3':'234b7de1969d80dd1523ce52afc89d5952b5e9ba4a63cdd7b3c8bff5b1c13633','published4':'71ba82d7eab40c0b8489f47ee373fd0ea08f2e2e33457c62152a1f8240d68348'}
PAL=np.array([[20,24,30],[220,50,65],[25,172,100],[160,75,215]],dtype='uint8')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(image,output,model_name):
 if output.exists():raise FileExistsError('Use a new output directory to preserve previous runs.')
 imagehash=sha(image);weights=WEIGHTS[model_name];assert sha(weights)==EXPECTED[model_name]
 with Image.open(image) as im:
  if im.mode!='RGB':raise ValueError('8-bit RGB microscopy input required; no silent conversion of calibrated16-bit data.')
  rgb=np.array(im)
 h,w=rgb.shape[:2];output.mkdir(parents=True);device='mps' if torch.backends.mps.is_available() else 'cpu';classes=4 if model_name=='published4' else 3
 model=smp.UnetPlusPlus(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=classes,activation=None);model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True);model.to(device).eval().requires_grad_(False)
 mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
 mask=np.zeros((h,w),dtype='uint8');qmap=np.zeros((h,w),dtype='uint16');core=896;halo=64;done=0;total=math.ceil(h/core)*math.ceil(w/core);start=time.monotonic()
 for y in range(0,h,core):
  for x in range(0,w,core):
   tile=np.zeros((1024,1024,3),dtype='uint8');x0=max(0,x-halo);y0=max(0,y-halo);x1=min(w,x+core+halo);y1=min(h,y+core+halo);dx=x0-(x-halo);dy=y0-(y-halo);tile[dy:dy+y1-y0,dx:dx+x1-x0]=rgb[y0:y1,x0:x1]
   t=torch.from_numpy(tile.transpose(2,0,1).copy()).float()[None].to(device)/255
   with torch.inference_mode():logits=model((t-mean)/std)[0];pred=logits.argmax(0).cpu().numpy();score=(logits.sigmoid() if classes==4 else logits.softmax(0)).cpu().numpy()
   if classes==4:p=score[1];e=score[2]
   else:p=score[0];e=score[1];pred=np.array([1,2,0],dtype='uint8')[pred]
   ch=min(core,h-y);cw=min(core,w-x);mask[y:y+ch,x:x+cw]=pred[halo:halo+ch,halo:halo+cw]
   q=p/np.maximum(p+e,1e-8);qmap[y:y+ch,x:x+cw]=np.rint(q[halo:halo+ch,halo:halo+cw]*65535).astype('uint16');done+=1
   if done%25==0 or done==total:print(f'{model_name}: {done}/{total} tiles, {time.monotonic()-start:.0f}s',flush=True)
 Image.fromarray(mask).save(output/'raw_classes.png');Image.fromarray(qmap).save(output/'periderm_fraction_scores.png');preview=Image.fromarray(PAL[mask]);preview.thumbnail((1800,1300),Image.Resampling.NEAREST);preview.save(output/'raw_segmentation_overview.png')
 sourcepreview=Image.fromarray(rgb);sourcepreview.thumbnail((1800,1300));sourcepreview.save(output/'source_overview.png');shutil.copy2(__file__,output/Path(__file__).name)
 assert sha(image)==imagehash and sha(weights)==EXPECTED[model_name]
 record={'source_image':str(image.resolve()),'source_sha256':imagehash,'source_size_wh':[w,h],'model_name':model_name,'weights':str(weights),'weights_sha256':EXPECTED[model_name],'class_codes':{'Background':0,'Periderm':1,'Endoderm':2,'Lateral_root':3},'lateral_root_available':classes==4,'prediction_input':'Image only; no annotations read.','scores':'uint16/65535 represents P score/(P score+E score), not calibrated biological probability. Published model sigmoid; working3 model softmax.','tiling':{'size':1024,'context_each_side':64,'core_stride':896,'aggregation':'Each core pixel retained once; black exterior padding. No resizing.'},'elapsed_seconds':time.monotonic()-start,'device':device,'versions':{'torch':torch.__version__,'smp':smp.__version__,'numpy':np.__version__},'original_files_unchanged':True,'evaluation':'Proof of concept; no claim of independent performance.'}
 (output/'prediction_record.json').write_text(json.dumps(record,indent=2)+'\n');print('Saved '+str(output),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--image',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--model',choices=list(WEIGHTS),default='working3');a=p.parse_args();run(a.image,a.output,a.model)
