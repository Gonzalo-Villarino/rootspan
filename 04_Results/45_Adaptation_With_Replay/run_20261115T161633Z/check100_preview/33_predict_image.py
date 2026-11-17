"""Reusable IMAGE-ONLY prediction on an RGB crop; no Labelme/JSON mask needed.

Requires explicit weights. Use --box X Y WIDTH HEIGHT to crop a large slide.
Outputs raw model predictions; does not enforce a left/right transition.
Tiles crops larger than1024 with overlap and probability averaging.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
import segmentation_models_pytorch as smp
Image.MAX_IMAGE_PIXELS=200_000_000
PAL=np.array([[48,52,59],[219,48,63],[26,166,94],[255,220,58]],dtype='uint8')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load_model(weights):
 device='mps' if torch.backends.mps.is_available() else 'cpu'
 model=smp.UnetPlusPlus(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=3,activation=None)
 model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True);model.to(device).eval().requires_grad_(False)
 return model,device
def predict(model,device,rgb):
 assert rgb.dtype==np.uint8 and rgb.ndim==3 and rgb.shape[2]==3
 h,w=rgb.shape[:2];mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
 def one(a):
  ih,iw=a.shape[:2];ph=(-ih)%32;pw=(-iw)%32;a=np.pad(a,((0,ph),(0,pw),(0,0)),mode='edge')
  x=torch.from_numpy(a.transpose(2,0,1).copy()).float()[None].to(device)/255
  with torch.inference_mode():p=model((x-mean)/std).softmax(1)[0,:,:ih,:iw].cpu().numpy()
  assert np.isfinite(p).all();return p
 if max(h,w)<=1024:return one(rgb)
 def starts(n):return sorted(set(list(range(0,max(1,n-1024+1),768))+[max(0,n-1024)]))
 total=np.zeros((3,h,w),dtype='float32');den=np.zeros((h,w),dtype='float32')
 for y in starts(h):
  for x in starts(w):
   a=rgb[y:y+1024,x:x+1024];ih,iw=a.shape[:2];win=np.maximum(np.hanning(ih)[:,None]*np.hanning(iw)[None,:],.05).astype('float32');total[:,y:y+ih,x:x+iw]+=one(a)*win;den[y:y+ih,x:x+iw]+=win
 assert (den>0).all();return total/den
def figure(rgb,pred,path):
 im=Image.fromarray(rgb);scale=min(1,700/max(im.size));sz=(round(im.width*scale),round(im.height*scale));canvas=Image.new('RGB',(sz[0]*2+60,sz[1]+180),'white');d=ImageDraw.Draw(canvas);font=lambda n:ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',n)
 for j,(title,pic) in enumerate([('Microscopy image',im),('Raw model prediction',Image.fromarray(PAL[pred]))]):
  x=20+j*(sz[0]+20);d.text((x,18),title,font=font(25),fill='black');canvas.paste(pic.resize(sz,Image.Resampling.LANCZOS if j==0 else Image.Resampling.NEAREST),(x,58))
 d.text((20,sz[1]+78),'Red = Periderm | Green = Endoderm | Yellow = Background',font=font(21),fill='black')
 d.text((20,sz[1]+113),'Prediction uses the image only. Review biologically; accuracy is not established on a new image.',font=font(17),fill='black')
 d.text((20,sz[1]+144),'Three classes only: lateral roots and uncertain transitions need special care.',font=font(17),fill='black');canvas.save(path)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--image',required=True,type=Path);p.add_argument('--weights',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--box',nargs=4,type=int,metavar=('X','Y','WIDTH','HEIGHT'));a=p.parse_args()
 assert not a.output.exists(),'Choose a new output folder; prior results are preserved.'
 sourcehash=sha(a.image);weighthash=sha(a.weights)
 with Image.open(a.image) as im:
  if im.mode!='RGB':raise ValueError('This model expects an 8-bit RGB image. Convert scientific grayscale/16-bit data explicitly first; no silent conversion.')
  if a.box:
   x,y,w,h=a.box;assert w>0 and h>0 and x>=0 and y>=0 and x+w<=im.width and y+h<=im.height;im=im.crop((x,y,x+w,y+h))
  if im.width*im.height>16_777_216:raise ValueError('Use --box to select a region of at most16 million pixels for this practice predictor.')
  rgb=np.array(im)
 model,device=load_model(a.weights);prob=predict(model,device,rgb);pred=(prob.argmax(0)+1).astype('uint8');a.output.mkdir(parents=True)
 Image.fromarray(rgb).save(a.output/'image.png');Image.fromarray(pred).save(a.output/'predicted_mask.png');Image.fromarray(PAL[pred]).save(a.output/'predicted_mask_colours.png');np.savez_compressed(a.output/'probabilities.npz',probabilities=prob);figure(rgb,pred,a.output/'OPEN_ME_prediction.png');shutil.copy2(__file__,a.output/Path(__file__).name)
 assert sha(a.weights)==weighthash and sha(a.image)==sourcehash
 record={'created_utc':datetime.now(timezone.utc).isoformat(),'image':str(a.image.resolve()),'image_sha256':sourcehash,'weights':str(a.weights.resolve()),'weights_sha256':weighthash,'box_xywh':a.box,'output_shape_hw':list(pred.shape),'device':device,'label_codes':{'Periderm':1,'Endoderm':2,'Background':3},'reference_or_annotations_read':False,'training_performed':False,'postprocessing':'None; overlapping-tile probability averaging only when image exceeds1024.','scope':'Exploratory prediction; no accuracy claim without an independent reviewed reference. Lateral roots are not a separate class.'}
 (a.output/'prediction_details.json').write_text(json.dumps(record,indent=2)+'\n');print('Saved '+str(a.output/'OPEN_ME_prediction.png'))
if __name__=='__main__':main()
