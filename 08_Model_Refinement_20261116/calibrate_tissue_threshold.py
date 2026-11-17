"""Calibrate P-versus-E decision using archived/user TEACHING labels only, then check20_29.
No development/reference pixels are used to choose the threshold. Existing BG/LR decisions retained for raw calibration.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from predict import load,predict,PAL
from review_candidate import full_metrics,gate
from main_root_rule import interpret
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'04_Results/54_Refinement_Experiments/rule_derived_endoderm_v7';data=ROOT/'04_Results/53_Selected_Teaching_Data';out=folder/'training_only_calibration';out.mkdir(exist_ok=False);man=json.loads((data/'manifest.json').read_text());m,d=load(folder/'best_weights.pt');ths=np.linspace(.02,.98,97);roots=[];scores=[]
for s in man['sources']:
 for rid in s['root_ids']:
  choices=[p for p in man['patches'] if p['slide']==s['slide'] and p['root']==rid and p['sample_type']=='transition' and min(p['counts']['1'],p['counts']['2'])>1000]
  if not choices:continue
  x=max(choices,key=lambda p:min(p['counts']['1'],p['counts']['2']));rgb=np.array(Image.open(data/x['image']));truth=np.array(Image.open(data/x['mask']));p=predict(m,d,rgb);q=p[0]/np.maximum(p[0]+p[1],1e-8);v1=q[truth==1];v2=q[truth==2];ss=np.array([.5*((v1>=t).mean()+(v2<t).mean()) for t in ths]);scores.append(ss);roots.append({'slide':s['slide'],'root':rid,'field':x['id'],'source':'Saved human/archived P/E labels, no rule-derivedCOL0 labels.','P_score_percentiles':np.percentile(v1,[10,50,90]).tolist(),'E_score_percentiles':np.percentile(v2,[10,50,90]).tolist(),'P_pixels':len(v1),'E_pixels':len(v2)});print(s['slide'],rid,flush=True)
means=np.mean(scores,axis=0);eligible=np.where(np.isclose(means,means.max(),atol=1e-8))[0];idx=eligible[np.argmin(abs(ths[eligible]-.5))];threshold=float(ths[idx]);record={'threshold':threshold,'training_balanced_accuracy':float(means[idx]),'training_roots':roots,'criterion':'Mean of P recall and E recall, averaged equally across14selectedteachingroots. Grid.02-.98. Tiesclosest.5. No20_29/A04/2_12input tofit.','grid':[{'threshold':float(t),'mean_balanced_accuracy':float(v)} for t,v in zip(ths,means)],'fit_complete_before_reference_scoring':True};(out/'calibration_fit.json').write_text(json.dumps(record,indent=2)+'\n')
# Only now load saved checking-image probabilities and reference for scoring.
p=np.load(folder/'best_probabilities.npz')['probabilities'];q=p[0]/np.maximum(p[0]+p[1],1e-8);raw=(p.argmax(0)+1).astype('uint8');pred=raw.copy();tissue=np.isin(raw,[1,2]);pred[tissue&(q>=threshold)]=1;pred[tissue&(q<threshold)]=2;ref=np.array(Image.open(ROOT/'04_Results/52_Reference_Check_20261116/reference_mask.png'));met=full_metrics(pred,ref)
newq=(q*(1-threshold))/np.maximum(q*(1-threshold)+(1-q)*threshold,1e-8);pc=p.copy();mass=p[0]+p[1];pc[0]=mass*newq;pc[1]=mass*(1-newq);rule,detail=interpret(pc);rmet=full_metrics(rule,ref);Image.fromarray(pred).save(out/'calibrated_raw_prediction.png');Image.fromarray(rule).save(out/'calibrated_rule_prediction.png');(out/'checking_results.json').write_text(json.dumps({'threshold':threshold,'raw_metrics':met,'rule_metrics':rmet,'raw_pass':gate(met),'rule_pass':gate(rmet),'rule':detail,'scope':'Familiar development check. Threshold fit only on teaching examples.'},indent=2)+'\n');print('THRESHOLD',threshold,'RAW',met['tissue_mean_iou'],'RULE',rmet['tissue_mean_iou'],'BOUNDARY',detail.get('boundary_x'),'GATE',gate(rmet),flush=True)
