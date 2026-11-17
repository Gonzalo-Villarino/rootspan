"""Published PAT weights: image-only long-root prediction and explicit cleanup.

The cleanup is a NEW experimental single-transition rule inspired by PAT's
use of root continuity. It is NOT Ling's exact150px contour algorithm.
Assumption for this oriented primary root: Endoderm left, Periderm right;
allow either class to occupy the entire root if the scores support that.
The rule never reads annotation coordinates or the reference mask.
PAT training-image membership is unknown: do not claim independent evaluation.
"""
import json,hashlib,importlib.util
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from scipy import ndimage as ndi
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'04_Results/38_PAT_Root_Context'
BOX=[1000,4800,14200,6500]
TARGET=[8928,5264,9952,6288]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def metrics(p,t):
 valid=np.isin(t,[1,2,3]);cm=np.bincount(4*(t[valid].astype(int)-1)+p[valid]-1,minlength=16).reshape(4,4);out={}
 for c,n in enumerate(['Periderm','Endoderm','Background']):
  tp=int(cm[c,c]);nref=int(cm[c].sum());npred=int(cm[:,c].sum());out[n]={'correct':tp,'reference':nref,'recall':tp/nref,'precision':tp/max(1,npred),'iou':tp/max(1,nref+npred-tp)}
 return {'per_class':out,'confusion_4x4_P_E_B_LR':cm.tolist(),'tissue_mean_iou':sum(out[k]['iou'] for k in ['Periderm','Endoderm'])/2,'scope':'Labelled root4 pixels only; PAT training membership unknown.'}
def main():
 OUT.mkdir(exist_ok=False)
 modelpath=ROOT/'00_Software/PAT_Reference/pat_state_dict.pt';model=smp.UnetPlusPlus(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=4,activation='sigmoid');model.load_state_dict(torch.load(modelpath,map_location='cpu',weights_only=True),strict=True);model.to('mps').eval()
 Image.MAX_IMAGE_PIXELS=300000000;source=ROOT/'02_Data/02_My_First_Annotations/MT25_rep3_practice.png';rgb=np.array(Image.open(source).convert('RGB').crop(BOX));h,w=rgb.shape[:2]
 pre=smp.encoders.get_preprocessing_fn('resnet34','imagenet');total=np.zeros((4,h,w),dtype='float32');den=np.zeros((h,w),dtype='float32')
 ys=list(range(0,h-1023,512));xs=list(range(0,w-1023,512))
 if ys[-1]!=h-1024:ys.append(h-1024)
 if xs[-1]!=w-1024:xs.append(w-1024)
 win=np.maximum(np.hanning(1024)[:,None]*np.hanning(1024)[None,:],.05).astype('float32')
 with torch.inference_mode():
  for k,(y,x) in enumerate((y,x) for y in ys for x in xs):
   patch=pre(rgb[y:y+1024,x:x+1024]).astype('float32');tensor=torch.from_numpy(patch.transpose(2,0,1)).unsqueeze(0).to('mps');scores=model(tensor)[0].cpu().numpy();assert np.isfinite(scores).all();total[:,y:y+1024,x:x+1024]+=scores*win;den[y:y+1024,x:x+1024]+=win
   if k%10==0:print(f'PAT context window {k+1}/{len(ys)*len(xs)}',flush=True)
 assert (den>0).all();probs=total/den;raw=np.array([3,1,2,4],dtype='uint8')[probs.argmax(0)]
 # A connected primary-root candidate, selected from prediction only.
 tissue=np.isin(raw,[1,2]);connected=ndi.binary_closing(tissue,iterations=4);labs,n=ndi.label(connected);sizes=np.bincount(labs.ravel());sizes[0]=0;mainroot=labs==sizes.argmax();mainroot=ndi.binary_dilation(mainroot,iterations=5)&tissue
 # Sum conditional P/E log evidence along the predicted main root.
 denom=(probs[1]+probs[2]).clip(1e-6);lp=np.log((probs[1]/denom).clip(1e-6,1));le=np.log((probs[2]/denom).clip(1e-6,1));sp=(lp*mainroot).sum(0);se=(le*mainroot).sum(0)
 scores=np.r_[0,np.cumsum(se)]+np.r_[np.cumsum(sp[::-1])[::-1],0];boundary=int(scores.argmax());clean=raw.copy();xx=np.arange(w)[None,:];clean[mainroot&(xx<boundary)]=2;clean[mainroot&(xx>=boundary)]=1
 x0,y0=TARGET[0]-BOX[0],TARGET[1]-BOX[1];s=(slice(y0,y0+1024),slice(x0,x0+1024));rp=raw[s];cp=clean[s]
 pp=ROOT/'04_Results/31_Root4_Background_Check_Patch';pair=json.loads((pp/'patch_details.json').read_text());truth=np.array(Image.open(pp/pair['mask_file']));assert sha(pp/pair['mask_file'])==pair['mask_sha256']
 result={'method':'Published PAT model, then separate experimental single-transition rule; not a reproduction of the exact PAT postprocessor.','image_only_inference':True,'full_roi_box':BOX,'root4_comparison_box':TARGET,'transition_x_full_image':boundary+BOX[0],'raw_metrics':metrics(rp,truth),'postprocessed_metrics':metrics(cp,truth),'model_sha256':sha(modelpath),'reference_sha256':pair['mask_sha256'],'source_sha256':sha(source),'disclaimer':'PAT training-image membership unknown. Scores are only against labelled reference pixels. Whole-root rule assumes mature Periderm to the right. No labels used in inference or transition choice.'}
 (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');Image.fromarray(raw).save(OUT/'whole_root_raw.png');Image.fromarray(clean).save(OUT/'whole_root_postprocessed.png');Image.fromarray(rp).save(OUT/'root4_raw.png');Image.fromarray(cp).save(OUT/'root4_postprocessed.png');np.save(OUT/'root4_probabilities.npy',probs[:,s[0],s[1]])
 pal=np.zeros((256,3),dtype='uint8');pal[0]=(48,52,58);pal[1]=(218,54,65);pal[2]=(30,170,91);pal[3]=(255,212,59);pal[4]=(160,85,210);pal[255]=(209,77,199)
 canvas=Image.new('RGB',(1700,760),'white');d=ImageDraw.Draw(canvas);d.text((20,12),'PAT benchmark: raw prediction and a separate continuity rule',font=ImageFont.load_default(size=29),fill='black');panels=[Image.fromarray(rgb[s]),Image.fromarray(pal[truth]),Image.fromarray(pal[rp]),Image.fromarray(pal[cp])]
 for i,(name,p) in enumerate(zip(['Microscopy','Your reference','PAT raw: longer root context','After single-transition rule'],panels)):
  x=20+420*i;d.text((x,64),name,font=ImageFont.load_default(size=19),fill='black');canvas.paste(p.resize((400,400),Image.Resampling.NEAREST),(x,100));canvas.paste(p.crop((100,350,1000,680)).resize((400,147),Image.Resampling.NEAREST),(x,530))
 d.text((20,695),'Red Periderm | Green Endoderm | Yellow Background | Purple Lateral root | Grey unlabelled',font=ImageFont.load_default(size=20),fill='black');d.text((20,726),'Released PAT weights; prior exposure to this slide is unknown. Reference used only for scoring.',font=ImageFont.load_default(size=19),fill='black');canvas.save(OUT/'pat_raw_and_cleanup.png')
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
