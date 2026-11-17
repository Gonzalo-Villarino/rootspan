"""Check training fit separately from transfer to a new slide. No training here."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from predict import load,predict
from preprocess import prepare
from check_reference import metrics
ROOT=Path(__file__).resolve().parents[1];data=ROOT/'04_Results/53_Selected_Teaching_Data';folder=ROOT/'04_Results/54_Refinement_Experiments/small_from_scratch_v4';man=json.loads((data/'manifest.json').read_text());m,d=load(folder/'final_weights.pt','small_unet');records=[]
for slide in [s['slide'] for s in man['sources']]:
 q=[p for p in man['patches'] if p['slide']==slide and p['sample_type']=='transition' and p['counts']['1']>1000 and p['counts']['2']>1000]
 q=[p for p in q if all((np.array(Image.open(data/p['mask']))[128:640,128:640]==c).sum()>1000 for c in [1,2])]
 for p in q[::max(1,len(q)//2)][:2]:
  rgb=np.array(Image.open(data/p['image']));mask=np.array(Image.open(data/p['mask']))[128:640,128:640];x=prepare(rgb,'grayscale_clahe')[128:640,128:640];prob=predict(m,d,x);pred=(prob.argmax(0)+1).astype('uint8');met=metrics(pred,mask);records.append({'field':p['id'],'crop_xywh':[128,128,512,512],'metrics':met});print(p['id'],round(met['tissue_mean_iou'],4),[round(met['per_class'][c]['recall'],4) if met['per_class'][c]['recall'] is not None else None for c in ['Periderm','Endoderm','Lateral_root']],flush=True)
(folder/'training_fit_examples.json').write_text(json.dumps({'scope':'Selected training examples only. Same512size as training; cannot establish generalization.','records':records},indent=2)+'\n')
