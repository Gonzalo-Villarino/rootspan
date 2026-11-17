"""Exploratory regional interpretation of a horizontal primary root.

Raw probabilities only; no annotation or reference inputs. Four-class adaptation excludes predicted lateral roots from primary-root path. User has confirmed the hypocotyl/right orientation. This is NOT a new
neural network or PAT's exact postprocessing. It assumes a single nearly
horizontal main root spanning the crop, Endoderm left / Periderm right, and at
most one transition. A 64-pixel review band is a display/analysis exclusion,
NOT a statistical confidence interval. Unsupported branches remain review255.
"""
from pathlib import Path
import argparse,importlib.util,json,shutil
import numpy as np
from scipy import ndimage as ndi
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
PAL=np.zeros((256,3),dtype='uint8');PAL[0]=[48,52,59];PAL[1]=[219,48,63];PAL[2]=[26,166,94];PAL[3]=[255,220,58];PAL[4]=[160,85,210];PAL[255]=[140,140,150]
def interpret(prob):
 h,w=prob.shape[1:];raw=(prob.argmax(0)+1).astype('uint8');tissue=np.isin(raw,[1,2])
 opened=ndi.binary_opening(ndi.binary_closing(tissue,iterations=2),structure=np.ones((1,65),bool));lab,n=ndi.label(opened)
 if not n: return raw.copy(),{'applied':False,'reason':'No long horizontal component'}
 areas=np.bincount(lab.ravel());areas[0]=0;main=lab==areas.argmax();xs=np.where(main.any(0))[0]
 if len(xs)<.8*w:return raw.copy(),{'applied':False,'reason':'No dominant root spanning80 percent of the crop'}
 top=np.full(w,np.nan);bottom=top.copy()
 for x in xs:
  col=main[:,x];runs=ndi.label(col)[0];a=np.bincount(runs);a[0]=0;ys=np.where(runs==a.argmax())[0];top[x]=ys.min();bottom[x]=ys.max()
 # Smooth centre/width to suppress protrusions of a lateral branch. No image mask input.
 top=np.interp(np.arange(w),xs,top[xs]);bottom=np.interp(np.arange(w),xs,bottom[xs]);centre=ndi.median_filter((top+bottom)/2,size=101,mode='nearest');thickness=ndi.median_filter(bottom-top+1,size=101,mode='nearest')
 yy=np.arange(h)[:,None];core=(abs(yy-centre[None])<=.20*thickness[None])&tissue
 d=np.log(np.maximum(prob[1],1e-6))-np.log(np.maximum(prob[0],1e-6));count=core.sum(0);score=(d*core).sum(0)/np.maximum(count,1);valid=count>=5
 if valid.sum()<.8*w:return raw.copy(),{'applied':False,'reason':'Insufficient primary-root core'}
 score=np.interp(np.arange(w),np.where(valid)[0],score[valid]);score=ndi.gaussian_filter1d(score,25,mode='nearest');cumulative=np.concatenate([[0.],np.cumsum(score)]);boundary=int(cumulative.argmax())
 band=abs(yy-centre[None])<=.60*thickness[None]+3
 primary=band&tissue;out=np.full((h,w),3,dtype='uint8');out[raw==4]=4;out[tissue]=255
 xx=np.arange(w)[None];out[primary&(xx<boundary)]=2;out[primary&(xx>=boundary)]=1
 if 0<boundary<w:out[primary&(abs(xx-boundary)<32)]=255
 return out,{'applied':True,'boundary_x':boundary,'review_half_width_pixels':32,'assumptions':'Single near-horizontal primary root spanning crop; Endoderm left, Periderm right; zero or one transition.','review255':'Predicted tissue outside the estimated main-root band, plus64px band around inferred transition. Not a confidence interval.','primary_pixels':int(primary.sum()),'review_pixels':int((out==255).sum()),'reference_input':False,'lateral_root_policy':'Class4excludedfrommainroot;preservedasmodelprediction. Otheroff-axis tissue leftforreview255.'}
def met(pred,truth):
 valid=np.isin(truth,[1,2,3]);keep=valid&(pred!=255);rows=[]
 for c,name in [(1,'Periderm'),(2,'Endoderm'),(3,'Background')]:
  tp=int(((pred==c)&(truth==c)).sum());fn=int(((pred!=c)&(truth==c)).sum());fp=int(((pred==c)&(truth!=c)&valid).sum());support=int((truth==c).sum())
  rows.append({'name':name,'support':support,'correct':tp,'recall_counting_review_as_not_correct':tp/max(1,support),'iou_counting_review_as_not_correct':tp/max(1,tp+fn+fp),'reviewed_out':int(((truth==c)&(pred==255)).sum())})
 return {'classes':rows,'labelled_coverage':float(keep.sum()/max(1,valid.sum())),'overall_accuracy_on_retained_labels_only':float(((pred==truth)&keep).sum()/max(1,keep.sum())),'scope':'Review pixels counted as missed in recall/IoU; retained-only accuracy must be read with coverage.'}
