"""Development diagnostic: use wider main-root evidence rather than only the central40percent.
No reference values enter boundary computation. All compared variants retained.
"""
from pathlib import Path
import json
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
from review_candidate import full_metrics,gate
from predict import PAL
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'04_Results/54_Refinement_Experiments/rule_derived_endoderm_v7';p=np.load(folder/'best_probabilities.npz')['probabilities'];raw=(p.argmax(0)+1).astype('uint8');tissue=np.isin(raw,[1,2]);h,w=tissue.shape;op=ndi.binary_opening(ndi.binary_closing(tissue,iterations=2),structure=np.ones((1,65),bool));lab,n=ndi.label(op);a=np.bincount(lab.ravel());a[0]=0;main=lab==a.argmax();xs=np.where(main.any(0))[0];top=[];bot=[]
for x in xs:
 col=main[:,x];labs=ndi.label(col)[0];ar=np.bincount(labs);ar[0]=0;yy=np.where(labs==ar.argmax())[0];top.append(yy.min());bot.append(yy.max())
top=np.interp(np.arange(w),xs,top);bot=np.interp(np.arange(w),xs,bot);centre=ndi.median_filter((top+bot)/2,101,mode='nearest');thickness=ndi.median_filter(bot-top+1,101,mode='nearest');yy=np.arange(h)[:,None];xx=np.arange(w)[None];band=(abs(yy-centre[None])<=.60*thickness[None]+3)&tissue;truth=np.array(Image.open(ROOT/'04_Results/52_Reference_Check_20261116/reference_mask.png'));out=folder/'cross_section_diagnostics';out.mkdir(exist_ok=False);record={}
for half in [.35,.45]:
 core=(abs(yy-centre[None])<=half*thickness[None])&tissue
 for method in ['mean_probability','median_probability','class_vote']:
  val=p[1]-p[0] if method!='class_vote' else np.where(raw==2,1.,-1.)
  if method=='median_probability':s=np.nanmedian(np.where(core,val,np.nan),axis=0)
  else:s=(val*core).sum(0)/np.maximum(core.sum(0),1)
  good=np.isfinite(s)&(core.sum(0)>=5);s=np.interp(np.arange(w),np.where(good)[0],s[good]);s=ndi.gaussian_filter1d(s,25);boundary=int(np.r_[0,np.cumsum(s)].argmax());pred=np.full((h,w),3,dtype='uint8');pred[raw==4]=4;pred[tissue]=255;pred[band&(xx<boundary)]=2;pred[band&(xx>=boundary)]=1
  if 0<boundary<w:pred[band&(abs(xx-boundary)<32)]=255
  met=full_metrics(pred,truth);name=f'{method}_halfwidth{half}';record[name]={'boundary_x':boundary,'metrics':met,'passes_development_gate':gate(met)};Image.fromarray(pred).save(out/f'{name}.png');print(name,boundary,met['tissue_mean_iou'],gate(met),flush=True)
(out/'diagnostics.json').write_text(json.dumps(record,indent=2)+'\n')
