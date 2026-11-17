from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image
from predict import load,predict
from preprocess import prepare
from check_reference import metrics
ROOT=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--folder',type=Path,required=True);args=a.parse_args();plan=json.loads((args.folder/'experiment_plan.json').read_text());data=ROOT/'04_Results/53_Selected_Teaching_Data';man=json.loads((data/'manifest.json').read_text());model,device=load(args.folder/'best_weights.pt');records=[]
for slide in [s['slide'] for s in man['sources']]:
 subset=[p for p in man['patches'] if p['slide']==slide and p['sample_type']=='transition' and p['counts']['1']>1000 and p['counts']['2']>1000]
 for q in subset[::max(1,len(subset)//2)][:2]:
  rgb=np.array(Image.open(data/q['image']));ma=np.array(Image.open(data/q['mask']));p=predict(model,device,prepare(rgb,plan.get('contrast_preprocessing','none')));pred=(p.argmax(0)+1).astype('uint8');met=metrics(pred,ma);records.append({'field':q['id'],'metrics':met});print(q['id'],met['tissue_mean_iou'],[met['per_class'][s]['recall'] for s in ['Periderm','Endoderm','Lateral_root']],flush=True)
(args.folder/'training_fit_examples.json').write_text(json.dumps(records,indent=2)+'\n')
