from pathlib import Path
import sys,json,hashlib,numpy as np
import torch
import torch.nn.functional as F
from decoder import decode
D=Path(__file__).resolve().parent;P=D.parent
B=P/'06_Models/Locked_Transition_v1_20261116/fixtures/profiles'
CASES={
 'COL0_root3':('04_Results/65_Mixed_Appearance_Evidence_Check/three_way_five_windows/frozen_features.npz','scan'),
 'A04_root3':('04_Results/67_Fixed_A04_Root3_Check/frozen_features_and_scores.npz','features'),
 '2_12_root1':('04_Results/68_Fixed_212_Root1_Check/frozen_features_and_scores.npz','features')}

def load(name):
 f,key=CASES[name];a=np.load(P/f)[key].astype('float64');r=json.loads((B/(name+'.json')).read_text());q=r['prediction']
 d=np.array(q.get('distance_along_local_path',q.get('distance_native_pixels')));t=r['expected_T_path_pixels']
 assert a.shape==(len(d),1024) and np.isfinite(a).all()
 selected=np.arange(len(d))[::5];selected=selected[abs(d[selected]-t)>=800]
 y=(d[selected]>t).astype(float)
 assert np.all(np.bincount(y.astype(int),minlength=2)>=3)
 return {'features':a,'distance':d,'T':t,'selected':selected,'labels':y,'source':f,'feature_key':key}

def fit(cases):
 torch.set_num_threads(4);torch.manual_seed(1617)
 xs=[];ys=[];ws=[]
 for item in cases:
  inds=item['selected'];y=item['labels'];counts=np.bincount(y.astype(int),minlength=2)
  xs.append(item['features'][inds]);ys.append(y);ws.append(1/(len(cases)*2*counts[y.astype(int)]))
 x=torch.tensor(np.concatenate(xs),dtype=torch.float64);y=torch.tensor(np.concatenate(ys),dtype=torch.float64)
 w=torch.tensor(np.concatenate(ws),dtype=torch.float64)
 head=torch.nn.Linear(1024,1,dtype=torch.float64)
 with torch.no_grad():head.weight.zero_();head.bias.zero_()
 optimizer=torch.optim.LBFGS(head.parameters(),lr=1,max_iter=250,line_search_fn='strong_wolfe',tolerance_grad=1e-9,tolerance_change=1e-12)
 losses=[]
 def closure():
  optimizer.zero_grad();logit=head(x).ravel()
  loss=(F.binary_cross_entropy_with_logits(logit,y,reduction='none')*w).sum()+.0005*head.weight.square().sum()
  loss.backward();losses.append(float(loss.detach()));return loss
 optimizer.step(closure);head.eval().requires_grad_(False)
 return head,{'samples':len(x),'objective_final':losses[-1],'closure_evaluations':len(losses),'weight_norm':float(head.weight.norm()),'losses':losses}

def score(head,x):
 with torch.inference_mode():return head(torch.tensor(x,dtype=torch.float64)).ravel().sigmoid().numpy()

def main():
 out=D/'regional_head_gate';out.mkdir(exist_ok=False)
 fingerprint={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [D/'HEAD_PLAN.txt',D/'decoder.py',Path(__file__)]+[P/x[0] for x in CASES.values()]}
 (out/'frozen_design_sha256.json').write_text(json.dumps(fingerprint,indent=2)+'\n')
 cases={k:load(k) for k in CASES};rows=[]
 for heldout in cases:
  head,history=fit([v for k,v in cases.items() if k!=heldout]);q=cases[heldout];p=score(head,q['features']);result=decode(p,q['distance'])
  controls=[]
  for label in [0,1]:
   ids=np.flatnonzero(q['distance']<=q['T']-800) if label==0 else np.flatnonzero(q['distance']>=q['T']+800)
   controls.append({'region':label, 'windows':len(ids),'T':decode(p[ids],q['distance'][ids])['T_path_pixels']})
  passed=result['T_path_pixels'] is not None and abs(result['T_path_pixels']-q['T'])<=640 and all(c['T'] is None for c in controls)
  row={'held_out_slide_root':heldout,'expected_T':q['T'],'result':result,'controls':controls,'passed':passed,'p_min':float(p.min()),'p_max':float(p.max()),'fit':{k:v for k,v in history.items() if k!='losses'}}
  torch.save(head.state_dict(),out/(heldout+'_fold.pt'))
  (out/(heldout+'_scores.json')).write_text(json.dumps({'distance':q['distance'].tolist(),'p_region':p.tolist(),'expected_T':q['T'],'result':result},indent=2)+'\n')
  rows.append(row);print(json.dumps(row,indent=2),flush=True)
 report={'passed':all(r['passed'] for r in rows),'folds':rows,'one_seven_features_read':False,'purpose':'Training-only exploratory sanity gate; not accuracy validation.','unique_roots':3,'unique_slides':3}
 (out/'gate.json').write_text(json.dumps(report,indent=2)+'\n')
 if not report['passed']:
  print('STOP: training-only gate failed. No final head promoted; 1_7 inference not run.',flush=True);return
 head,history=fit(list(cases.values()));torch.save(head.state_dict(),out/'candidate_head.pt')
 # The reviewed short20_29crop must not receive a forced T.
 f=P/'04_Results/66_Fixed_20_29_Check/frozen_features_and_scores.npz'
 a=np.load(f);features=a['features'];q=json.loads((B/'20_29_root4.json').read_text())['prediction'];d=q.get('distance_along_local_path',q.get('distance_native_pixels'))
 result=decode(score(head,features),d);report['20_29_control']=result
 report['passed']=result['T_path_pixels'] is None
 (out/'gate.json').write_text(json.dumps(report,indent=2)+'\n')
 print('Gate passed:',report['passed'],flush=True)
if __name__=='__main__':main()
