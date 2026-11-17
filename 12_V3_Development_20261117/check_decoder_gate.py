from pathlib import Path
import sys,json,hashlib,numpy as np
D=Path(__file__).resolve().parent;P=D.parent
from decoder import decode
B=P/'06_Models/Locked_Transition_v1_20261116/fixtures/profiles'
# Fixed procedure code fingerprints before loading evaluation profiles.
(D/'no_training_code_sha256.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [D/'decoder.py',D/'NO_TRAINING_PLAN.txt',Path(__file__)]},indent=2)+'\n')
for probability in [.01,.99,.5]:assert decode(np.full(40,probability),np.arange(40)*64.)['T_path_pixels'] is None
rows=[]
for f in sorted(B.glob('*.json')):
 r=json.loads(f.read_text());p=r['prediction'];scores=p.get('presence_scores',p.get('predicted_presence_scores'));d=p.get('distance_along_local_path',p.get('distance_native_pixels'))
 actual=decode(np.array(scores)[:,1],d);expected=r['expected_T_path_pixels']
 passed=actual['T_path_pixels'] is None if expected is None else (actual['T_path_pixels'] is not None and abs(actual['T_path_pixels']-expected)<=640)
 rows.append({'case':r['case'],'accepted_reference_T':expected,'candidate':actual,'gate_passed':passed})
report={'gate_passed':all(r['gate_passed'] for r in rows),'rows':rows,'controls_passed':True,'reference_scope':'Post-prediction accepted development positions, not independent accuracy.','one_seven_read':False}
(D/'no_training_gate.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
