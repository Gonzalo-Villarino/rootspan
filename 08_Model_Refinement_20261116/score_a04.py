"""Compare frozen predictions with archived Periderm outlines AFTER image-only inference."""
from pathlib import Path
import argparse,json,csv
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'04_Results/57_A04_Reserved_Check';Image.MAX_IMAGE_PIXELS=300_000_000
p=argparse.ArgumentParser();p.add_argument('--prediction',required=True,type=Path);p.add_argument('--measurements',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();a.output.mkdir(exist_ok=False)
ref=np.array(Image.open(BASE/'periderm_reference_root_ids.png'));pred=np.array(Image.open(a.prediction/'raw_classes.png'));assert ref.shape==pred.shape
truth=json.loads((BASE/'reference_lengths.json').read_text());measures=json.loads((a.measurements/'measurement_record.json').read_text());rr=measures['roots'];cost=np.array([[np.linalg.norm(np.array(t['archive_basal_xy'])-np.array(r['automatic_basal_xy'])) for r in rr] for t in truth]);match={int(i):int(j) for i,j in zip(*linear_sum_assignment(cost))} if rr else {};rows=[]
for i,t in enumerate(truth):
 rid=t['root_id'];v=ref==rid;n=int(v.sum());tp=int(((pred==1)&v).sum());rec={'reference_root':rid,'periderm_pixels':n,'matched_periderm_pixels':tp,'periderm_recall':tp/max(n,1),'archive_outline_length_px':t['archive_outline_length_px'],'matched_model_root':None,'model_length_px':None,'signed_length_difference_pct':None,'absolute_length_difference_pct':None,'basal_endpoint_distance_px':None,'transition_distance_px':None}
 if i in match:
  r=rr[match[i]];dist=float(cost[i,match[i]]);rec['basal_endpoint_distance_px']=dist
  if dist<=600:
   length=r['periderm_length_px'];rec.update(matched_model_root=r['root_id'],model_length_px=length,model_flags=r['flags'])
   if length is not None:
    pct=100*(length/t['archive_outline_length_px']-1);rec.update(signed_length_difference_pct=pct,absolute_length_difference_pct=abs(pct),transition_distance_px=float(np.linalg.norm(np.array(t['archive_start_xy'])-np.array(r['automatic_transition_xy']))))
  else:rec['match_status']='Endpoint too far away; no accepted automatic root match.'
 rows.append(rec)
valid=[r['absolute_length_difference_pct'] for r in rows if r['absolute_length_difference_pct'] is not None];report={'scope':'Reserved local slide, Periderm-only archive check. No Endoderm/Background truth. OriginalPATexposureunknown. Length references derived from saved polygons; not independent manual centreline measurements.','prediction':str(a.prediction),'measurements':str(a.measurements),'expected_roots':6,'detected_root_records':len(rr),'matched_length_roots':len(valid),'median_absolute_length_difference_pct':float(np.median(valid)) if valid else None,'max_absolute_length_difference_pct':max(valid) if valid else None,'pooled_periderm_recall':sum(r['matched_periderm_pixels'] for r in rows)/sum(r['periderm_pixels'] for r in rows),'rows':rows,'biological_endpoint_review':'pending','selection_policy':'This check is not used to select or tune another model in this run.'};(a.output/'scores.json').write_text(json.dumps(report,indent=2)+'\n')
with (a.output/'per_root_comparison.csv').open('w',newline='') as f:
 keys=['reference_root','matched_model_root','archive_outline_length_px','model_length_px','absolute_length_difference_pct','basal_endpoint_distance_px','transition_distance_px','periderm_recall'];w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)
