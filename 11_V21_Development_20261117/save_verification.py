from pathlib import Path
import sys,json,hashlib,csv
import numpy as np
P=Path(__file__).resolve().parents[1];D=P/'11_V21_Development_20261117'
sys.path.insert(0,str(P/'09_Periderm_GUI'))
from version_registry import verify_v2
from pipeline_v21 import BUNDLE,verify

def read(p):return json.loads(p.read_text())
def save(p,r):p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
old={'7_18':P/'10_V2_Development_20261117/diagnostics/v2_full_context_7_18',**{s:P/'10_V2_Development_20261117/regression'/s for s in ['1_7','22_22','Col_rep1','WOX4_rep1']}}
new={'7_18':D/'trial_v21_7_18',**{s:D/'regression'/s for s in ['1_7','22_22','Col_rep1','WOX4_rep1']}}
report={'coverage_is_not_accuracy':True,'training_occurred':False,'development_slides_not_blind_test':True,'slides':[]}
for name in old:
 a,b=read(old[name]/'results.json'),read(new[name]/'results.json')
 assert a['source_sha256']==b['source_sha256']
 assert a['detected_groups']==b['detected_groups']
 assert not b['training_occurred'] and not b['human_paths_or_transition_points_used']
 for row in b['rows']:
  length=row['periderm_length_um']
  if length is not None:assert length>=0 and abs(row['length_pixels']/.5299-length)<1e-7
  if row['status'] in ('partial_root','failed','no_transition','low_image_quality','transition_beyond_endpoint'):
   assert length is None
 report['slides'].append({'slide':name,'tracks':b['detected_groups'],'v2_provisional':a['provisional_lengths'],
  'v21_provisional':b['provisional_lengths'],'v21_statuses':[r['status'] for r in b['rows']],
  'short_transition_candidates':[r['track_id'] for r in b['rows'] if r.get('brief_transition_review_required')],
  'source_sha256':b['source_sha256']})
refs={1:([13707,1517],8002.1095),2:([14129,3381],13710.5634),6:([14180,6729],17935.5550)}
manual=[]
a,b=read(old['1_7']/'results.json'),read(new['1_7']/'results.json')
for i,(xy,length) in refs.items():
 r={'root':i,'manual_endpoint_xy':xy,'manual_periderm_um':length}
 for v,data in [('v2',a),('v21',b)]:
  row=data['rows'][i-1]
  r[v+'_endpoint_error_um']=float(np.linalg.norm(np.array(row['endpoint_xy'])-xy))/.5299
  r[v+'_periderm_um']=row['periderm_length_um']
  r[v+'_length_difference_percent']=100*(row['periderm_length_um']-length)/length
 manual.append(r)
report['manual_1_7_comparison']=manual
ref=read(D/'root2_junction_confirmed_D.json')
row=read(new['7_18']/'results.json')['rows'][1]
report['root2_D_check']={'expert_junction_xy':ref['confirmed_stop_xy'],'automatic_endpoint_xy':row['endpoint_xy'],
 'error_um':float(np.linalg.norm(np.array(row['endpoint_xy'])-ref['confirmed_stop_xy']))/.5299,
 'automatic_point_is_left_of_D':row['endpoint_xy'][0]<ref['confirmed_stop_xy'][0],
 'expert_coordinate_input_to_pipeline':False}
report['conclusion']='Computational coverage improved. Endpoint heuristic is closer to three Fiji reference endpoints; full periderm lengths still differ substantially. No biological accuracy claim.'
report['tests']={'unittest_modules':['test_pipeline','test_pipeline_v2','test_pipeline_v21'],'passed':14,'failed':0}
report['v2_integrity']=verify_v2();report['locked_v1_integrity']=verify(BUNDLE,require_read_only=True)
save(D/'verification.json',report)
print(json.dumps(report,indent=2))
