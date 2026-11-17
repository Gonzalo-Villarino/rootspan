"""Experimental regional single-change decoder; no tissue or endpoint inputs.

Operational P-region includes mixture. Presence score is not calibrated. Full
E/P alternatives may win; the rule must then abstain. Fixed before evaluation.
"""
import numpy as np
WINDOW=320.
MIN_SPACED=3
MIN_MEAN_GAIN=.02

def spaced(d,ids):
 out=[]
 for i in ids:
  if not out or d[i]-d[out[-1]]>=WINDOW-1e-8:out.append(int(i))
 return out

def decode(p,d):
 p=np.asarray(p,float);d=np.asarray(d,float)
 if p.ndim!=1 or p.shape!=d.shape or not np.isfinite(p).all() or not np.isfinite(d).all() or (np.diff(d)<=0).any():raise ValueError('Invalid score sequence.')
 p=np.clip(p,1e-6,1-1e-6)
 e=-np.log1p(-p);r=-np.log(p)
 all_e=float(e.sum());all_p=float(r.sum());base=min(all_e,all_p)
 costs=np.cumsum(e)[:-1]+(r.sum()-np.cumsum(r)[:-1])
 candidates=[]
 for k,c in enumerate(costs,1):
  # Independent-spaced support on both sides, not simply overlapping pixels.
  before=spaced(d,np.flatnonzero((np.arange(len(p))<k)&(p<.5)))
  after=spaced(d,np.flatnonzero((np.arange(len(p))>=k)&(p>=.5)))
  if len(before)<MIN_SPACED or len(after)<MIN_SPACED:continue
  candidates.append({'index':k,'cost':float(c),'T_distance_pixels':float((d[k-1]+d[k])/2),'E_support':before,'P_support':after})
 best=min(candidates,key=lambda v:v['cost']) if candidates else None
 gain=None if best is None else (base-best['cost'])/len(p)
 accepted=best is not None and gain>=MIN_MEAN_GAIN
 return {'T_path_pixels':best['T_distance_pixels'] if accepted else None,'accepted':accepted,
   'best_candidate':best,'all_E_cost':all_e,'all_P_cost':all_p,'mean_gain':gain,
   'reason':'regional_change_supported' if accepted else 'insufficient_evidence_over_single_region',
   'near_optimal_range_pixels':None if not accepted else [min(c['T_distance_pixels'] for c in candidates if c['cost']<=best['cost']+.02*len(p)),max(c['T_distance_pixels'] for c in candidates if c['cost']<=best['cost']+.02*len(p))],
   'range_is_not_confidence_interval':True,'scores_not_calibrated':True}
