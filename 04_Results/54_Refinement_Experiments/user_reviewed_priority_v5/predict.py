"""Image-only four-class inference. No reference/annotation argument or import."""
from pathlib import Path
import argparse,json,hashlib,importlib.util
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
ROOT=Path(__file__).resolve().parents[1];PAL=np.zeros((256,3),dtype='uint8');PAL[0]=(48,52,59);PAL[1]=(219,48,63);PAL[2]=(26,166,94);PAL[3]=(255,220,58);PAL[4]=(160,85,210);PAL[255]=(220,80,180)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(weights,architecture='pat'):
 from small_unet import SmallUNet
 device='mps' if torch.backends.mps.is_available() else 'cpu';m=SmallUNet() if architecture=='small_unet' else smp.UnetPlusPlus(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=4,activation=None);m.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True);m.to(device).eval().requires_grad_(False);return m,device
def predict(model,device,rgb,scale=1.,flips=False):
 mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
 def one(arr):
  h,w=arr.shape[:2];arr=np.pad(arr,((0,(-h)%32),(0,(-w)%32),(0,0)),mode='edge');x=torch.from_numpy(arr.transpose(2,0,1).copy()).float()[None].to(device)/255
  with torch.inference_mode():prob=model((x-mean)/std).softmax(1)[0,:,:h,:w].cpu().numpy()
  assert np.isfinite(prob).all();return prob
 h,w=rgb.shape[:2]
 if scale!=1:rgb=np.array(Image.fromarray(rgb).resize((round(w*scale),round(h*scale)),Image.Resampling.BILINEAR))
 hh,ww=rgb.shape[:2]
 if max(hh,ww)<=1024:
  p=one(rgb)
  if flips:p=(p+one(rgb[:,::-1])[:,:,::-1]+one(rgb[::-1])[:,::-1]+one(rgb[::-1,::-1])[:,::-1,::-1])/4
 else:
  def starts(n):return sorted(set(list(range(0,max(1,n-1024+1),768))+[max(0,n-1024)]))
  p=np.zeros((4,hh,ww),dtype='float32');den=np.zeros((hh,ww),dtype='float32')
  for y in starts(hh):
   for x in starts(ww):
    sub=rgb[y:y+1024,x:x+1024];sh,sw=sub.shape[:2];win=np.maximum(np.hanning(sh)[:,None]*np.hanning(sw)[None,:],.05).astype('float32');p[:,y:y+sh,x:x+sw]+=one(sub)*win;den[y:y+sh,x:x+sw]+=win
  p/=den
 if scale!=1:p=F.interpolate(torch.from_numpy(p)[None],size=(h,w),mode='bilinear',align_corners=False)[0].numpy()
 assert p.shape==(4,h,w) and np.isfinite(p).all() and np.allclose(p.sum(0),1,atol=1e-5);return p

def main():
 a=argparse.ArgumentParser();a.add_argument('--image',type=Path,required=True);a.add_argument('--weights',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--box',nargs=4,type=int);a.add_argument('--scale',type=float,default=1.);a.add_argument('--architecture',choices=['pat','small_unet'],default='pat');a.add_argument('--contrast',choices=['none','grayscale_clahe'],default='none');args=a.parse_args();assert not args.output.exists();Image.MAX_IMAGE_PIXELS=300_000_000
 with Image.open(args.image) as im:
  assert im.mode=='RGB','Use an explicitly converted RGB source.'
  if args.box:
   x,y,w,h=args.box;assert x>=0 and y>=0 and w>0 and h>0 and x+w<=im.width and y+h<=im.height;im=im.crop((x,y,x+w,y+h))
  assert im.width*im.height<=16_777_216,'Use a crop for this predictor.';rgb=np.array(im)
 from preprocess import prepare
 m,device=load(args.weights,args.architecture);p=predict(m,device,prepare(rgb,args.contrast),args.scale);pred=(p.argmax(0)+1).astype('uint8');args.output.mkdir(parents=True);Image.fromarray(rgb).save(args.output/'image.png');Image.fromarray(pred).save(args.output/'prediction.png');Image.fromarray(PAL[pred]).save(args.output/'prediction_colours.png');np.savez_compressed(args.output/'probabilities.npz',probabilities=p)
 record={'image':str(args.image.resolve()),'image_sha256':sha(args.image),'weights':str(args.weights.resolve()),'weights_sha256':sha(args.weights),'box_xywh':args.box,'scale':args.scale,'input':'RGB image only. No annotations/mask read.','class_codes':{'1':'Periderm','2':'Endoderm','3':'Background','4':'Lateral_root'},'model':args.architecture,'contrast_preprocessing':args.contrast,'postprocessing':'None; probability blending for larger tiled inputs.','role':'Exploratory model-only prediction. Human review needed; not validated for unattended measurement.'};(args.output/'prediction_details.json').write_text(json.dumps(record,indent=2)+'\n');print(args.output/'prediction_colours.png')
if __name__=='__main__':main()
