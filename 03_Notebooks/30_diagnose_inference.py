"""Bounded image-only inference comparison. No new labels or training."""
from pathlib import Path
import hashlib, json, shutil, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'04_Results/44_Inference_Diagnostics'
WEIGHTS=ROOT/'04_Results/39_PAT_Transfer_Learning/run_20261115T140159Z/model_weights.pt'
EXPECTED='234b7de1969d80dd1523ce52afc89d5952b5e9ba4a63cdd7b3c8bff5b1c13633'
PAL=np.array([[48,52,59],[219,48,63],[26,166,94],[255,220,58],[160,85,210]],dtype=np.uint8)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def metric(pred,truth):
 valid=np.isin(truth,[1,2,3]);cm=np.bincount((truth[valid].astype(int)-1)*4+pred[valid]-1,minlength=12).reshape(3,4)
 return {'confusion_reference_P_E_B_prediction_P_E_B_LR':cm.tolist(),'class_iou':{name:float(cm[i,i]/max(1,cm[i].sum()+cm[:,i].sum()-cm[i,i])) for i,name in enumerate(['Periderm','Endoderm','Background'])}}
def figure(rgb,preds,path,title):
 names=['Microscopy']+list(preds);panels=[rgb]+[PAL[v] for v in preds.values()]
 w=512;h=240;canvas=Image.new('RGB',(3*(w+20)+20,2*(h+65)+120),'white');d=ImageDraw.Draw(canvas)
 font=lambda n:ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',n)
 d.text((20,12),title,font=font(28),fill='black')
 for i,(name,im) in enumerate(zip(names,panels)):
  x=20+(i%3)*(w+20);y=62+(i//3)*(h+65)
  d.text((x,y),name,font=font(21),fill='black')
  # Same source crop for every panel, aspect ratio preserved.
  p=Image.fromarray(im).crop((0,250,1024,730)).resize((w,h),Image.Resampling.LANCZOS if i==0 else Image.Resampling.NEAREST)
  canvas.paste(p,(x,y+34))
 d.text((20,canvas.height-38),'Red Periderm | Green Endoderm | Yellow Background | Purple Lateral root. Diagnostic variants, no new training.',font=font(21),fill='black');canvas.save(path)
def main():
 assert sha(WEIGHTS)==EXPECTED
 OUT.mkdir(exist_ok=False);shutil.copy2(__file__,OUT/Path(__file__).name)
 device='mps' if torch.backends.mps.is_available() else 'cpu'
 model=smp.UnetPlusPlus(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=3,activation=None)
 model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True);model.to(device).eval().requires_grad_(False)
 mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
 def forward(rgb):
  x=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None].to(device)/255
  with torch.inference_mode():p=model((x-mean)/std).softmax(1)[0].cpu().numpy()
  assert np.isfinite(p).all();return p
 def scaled(rgb,size):
  p=forward(np.array(Image.fromarray(rgb).resize((size,size),Image.Resampling.BILINEAR)))
  return F.interpolate(torch.from_numpy(p)[None],size=(1024,1024),mode='bilinear',align_corners=False)[0].numpy()
 def tiled(rgb):
  total=np.zeros((3,1024,1024),dtype=np.float32);den=np.zeros((1024,1024),dtype=np.float32)
  win=np.maximum(np.hanning(512)[:,None]*np.hanning(512)[None,:],.05).astype('float32')
  for y in (0,256,512):
   for x in (0,256,512):total[:,y:y+512,x:x+512]+=forward(rgb[y:y+512,x:x+512])*win;den[y:y+512,x:x+512]+=win
  return total/den
 def cleanup(p):
  # Reuse unchanged existing rule, retaining its documented assumptions.
  import importlib.util
  spec=importlib.util.spec_from_file_location('continuity',ROOT/'03_Notebooks/26_check_and_show_improvement.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  return m.continuity(p)
 cases={
  '20_29_periderm':(ROOT/'04_Results/41_New_Image_20_29_Prediction/view_20261115T152946Z_kc80hx_h/20_29_x7988_y5618_image.png',None),
  '20_29_ambiguous':(ROOT/'04_Results/42_20_29_Transition_Prediction/view_20261115T153948Z_bmduly9w/20_29_x4864_y6144_image.png',None),
  '20_29_clearer':(ROOT/'04_Results/43_20_29_Clearer_Transition_Prediction/view_20261115T155056Z_vbrbuhdb/20_29_x5440_y4256_image.png',None),
  'MT25_development':(ROOT/'04_Results/31_Root4_Background_Check_Patch/MT25_rep3_root4_x8928_y5264_image.png',ROOT/'04_Results/31_Root4_Background_Check_Patch/MT25_rep3_root4_x8928_y5264_mask.png')}
 record={'purpose':'Inference sensitivity only; no training. All variants retained, not an independent best-of-test evaluation.','weights_sha256':EXPECTED,'cases':{}}
 buffers={}
 for name,(ip,mp) in cases.items():
  print('Checking '+name,flush=True);rgb=np.array(Image.open(ip));folder=OUT/name;folder.mkdir();buffers[name]=rgb
  native=forward(rgb);tile=tiled(rgb)
  multi=(native+scaled(rgb,512)+scaled(rgb,768)+scaled(rgb,1536))/4
  clean,boundary=cleanup(native)
  probabilities={'native1024':native,'overlap512':tile,'scale_average':multi}
  preds={k:(p.argmax(0)+1).astype('uint8') for k,p in probabilities.items()};preds['existing_continuity_rule']=clean
  details={'image':str(ip),'image_sha256':sha(ip),'cleanup_boundary_column':boundary,'variants':{}}
  for key,p in probabilities.items():np.savez_compressed(folder/f'{key}_probabilities.npz',probabilities=p)
  for key,p in preds.items():
   Image.fromarray(p).save(folder/f'{key}.png');details['variants'][key]={'pixel_counts_not_accuracy':{int(k):int(v) for k,v in zip(*np.unique(p,return_counts=True))}}
   if mp:details['variants'][key]['development_metrics']=metric(p,np.array(Image.open(mp)))
  record['cases'][name]=details
  (OUT/'diagnostics.json').write_text(json.dumps(record,indent=2)+'\n')
 del model
 if device=='mps':torch.mps.empty_cache()
 model=smp.UnetPlusPlus(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=4,activation=None)
 public=ROOT/'00_Software/PAT_Reference/pat_state_dict.pt';model.load_state_dict(torch.load(public,map_location='cpu',weights_only=True),strict=True);model.to(device).eval().requires_grad_(False)
 record['public_PAT_weights_sha256']=sha(public)
 for name,(ip,mp) in cases.items():
  folder=OUT/name;rgb=buffers[name];p=forward(rgb);pred=np.array([3,1,2,4],dtype='uint8')[p.argmax(0)];Image.fromarray(pred).save(folder/'public_PAT4.png');np.savez_compressed(folder/'public_PAT4_probabilities.npz',probabilities=p)
  record['cases'][name]['variants']['public_PAT4']={'pixel_counts_not_accuracy':{int(k):int(v) for k,v in zip(*np.unique(pred,return_counts=True))}}
  if mp:record['cases'][name]['variants']['public_PAT4']['development_metrics']=metric(pred,np.array(Image.open(mp)))
  preds={key:np.array(Image.open(folder/f'{key}.png')) for key in ['native1024','overlap512','scale_average','existing_continuity_rule','public_PAT4']}
  figure(rgb,preds,folder/'comparison.png',name+' | Fixed inference comparison; no training')
 assert sha(WEIGHTS)==EXPECTED
 (OUT/'diagnostics.json').write_text(json.dumps(record,indent=2)+'\n');print('Saved '+str(OUT),flush=True)
if __name__=='__main__':main()
