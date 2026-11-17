from pathlib import Path
import json,hashlib,shutil
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'04_Results/52_Reference_Check_20261116'
REF=ROOT/'02_Data/05_Training_and_Evaluation/02_Evaluation_Images/03_Your_Annotations/20_29_root4_reference_GV.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def metrics(pred,mask):
 v=np.isin(mask,[1,2,3,4]);cm=np.bincount((mask[v].astype(int)-1)*4+pred[v]-1,minlength=16).reshape(4,4)
 out={}
 for i,n in enumerate(['Periderm','Endoderm','Background','Lateral_root']):
  tp=int(cm[i,i]);nr=int(cm[i].sum());np_=int(cm[:,i].sum());out[n]={'reference_pixels':nr,'recall':tp/nr if nr else None,'precision':tp/np_ if np_ else None,'iou':tp/(nr+np_-tp) if nr+np_-tp else None,'correct_pixels':tp}
 return {'per_class':out,'tissue_mean_iou':float(np.mean([out[n]['iou'] for n in ['Periderm','Endoderm']])),'confusion_reference_rows_prediction_columns_P_E_B_LR':cm.tolist(),'scope':'Only manually annotated pixels in familiar development crop; not independent test accuracy.'}
def raster(d):
 bits=np.zeros((d['imageHeight'],d['imageWidth']),dtype='uint8')
 for s in d['shapes']:
  assert s['label'] in ['Periderm','Endoderm','Background']
  a=Image.new('1',(d['imageWidth'],d['imageHeight']));pts=[tuple(p) for p in s['points']];assert s['shape_type']=='polygon';ImageDraw.Draw(a).polygon(pts,fill=1);bits[np.array(a)]|={'Periderm':1,'Endoderm':2,'Background':4}[s['label']]
 lut=np.array([0,1,2,255,3,255,255,255],dtype='uint8');return lut[bits]
def main():
 d=json.loads(REF.read_text());mask=raster(d);Image.fromarray(mask).save(OUT/'reference_mask.png');shutil.copy2(REF,OUT/'annotations_snapshot.json');shutil.copy2(REF.with_suffix('.png'),OUT/'image.png')
 report={'annotation_sha256':sha(REF),'image_sha256':sha(REF.with_suffix('.png')),'labels':[s['label'] for s in d['shapes']],'counts':{str(c):int((mask==c).sum()) for c in [0,1,2,3,255]},'source':'Gonzalo manually labelled this crop 16Nov after viewing previous predictions; developmental reference, not blind test. Lateral root and uncertainty gaps not annotated.','comparisons':{}}
 for name in ['native1024','overlap512','scale_average','existing_continuity_rule','public_PAT4']:
  p=ROOT/'04_Results/44_Inference_Diagnostics/20_29_clearer'/f'{name}.png'
  if p.exists():report['comparisons'][name]=metrics(np.array(Image.open(p)),mask)
 (OUT/'reference_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
