"""Diagnostic only: expected-error one-transition rule, using probabilities rather than log odds.
No human reference input to boundary estimation. Not the original PAT postprocessing.
"""
from pathlib import Path
import json
import numpy as np
from scipy import ndimage as ndi
ROOT=Path(__file__).resolve().parents[1]
def estimate(prob):
 raw=prob.argmax(0)+1;tissue=np.isin(raw,[1,2]);h,w=tissue.shape
 op=ndi.binary_opening(ndi.binary_closing(tissue,iterations=2),structure=np.ones((1,65),bool));lab,n=ndi.label(op)
 if n==0:return {'applied':False}
 a=np.bincount(lab.ravel());a[0]=0;main=lab==a.argmax();xs=np.where(main.any(0))[0]
 if len(xs)<.8*w:return {'applied':False}
 top=np.array([np.where(main[:,x])[0].min() for x in xs]);bottom=np.array([np.where(main[:,x])[0].max() for x in xs]);top=np.interp(np.arange(w),xs,top);bottom=np.interp(np.arange(w),xs,bottom);cen=ndi.median_filter((top+bottom)/2,101);thick=ndi.median_filter(bottom-top+1,101)
 core=(abs(np.arange(h)[:,None]-cen[None])<=.2*thick[None])&tissue;count=core.sum(0);q=(prob[1]-prob[0]);s=(q*core).sum(0)/np.maximum(count,1);v=count>=5;s=np.interp(np.arange(w),np.where(v)[0],s[v]);s=ndi.gaussian_filter1d(s,25);return {'applied':True,'boundary_x':int(np.r_[0,np.cumsum(s)].argmax()),'column_scores':s.tolist()}
if __name__=='__main__':
 folder=ROOT/'04_Results/54_Refinement_Experiments/baseline_refinement_v2/inference_ensemble';records={}
 for f in sorted(folder.glob('*_probabilities.npz')):
  p=np.load(f)['probabilities'];info=estimate(p);records[f.stem]=info;print(f.stem,info.get('boundary_x'),flush=True)
 records['method']='Expected classification-error one-transition rule. Image probabilities only; E left P right. No reference input. Diagnostic; not accepted method.'
 (folder/'expected_error_boundaries.json').write_text(json.dumps(records,indent=2)+'\n')
