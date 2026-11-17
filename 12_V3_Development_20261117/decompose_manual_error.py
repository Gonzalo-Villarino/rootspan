from pathlib import Path
import json,numpy as np
P=Path(__file__).resolve().parents[1];D=Path(__file__).resolve().parent
refs=json.loads((P/'04_Results/73_V1_Manual_Comparison_Pilot/01_Manual_References/manual_reference_index.json').read_text())['records']
base=P/'11_V21_Development_20261117/regression/1_7'
def project(xy,path,arc):
 v=np.diff(path,axis=0);t=np.clip(((xy-path[:-1])*v).sum(1)/np.maximum((v*v).sum(1),1e-12),0,1)
 q=path[:-1]+t[:,None]*v;i=np.argmin(np.linalg.norm(q-xy,axis=1))
 return float(arc[i]+t[i]*(arc[i+1]-arc[i])),float(np.linalg.norm(q[i]-xy))
rows=[]
for ref in refs:
 folder=base/f'track_{ref["root_id"]:02d}';route=json.loads((folder/'route.json').read_text());result=json.loads((folder/'result.json').read_text())
 path=np.array(route['path_xy_crop_pixels'])+route['source_crop_xyxy'][:2];arc=np.array(route['distance_along_path_pixels'])
 pts=np.array(ref['points_xy']);j=pts[np.argmax(pts[:,0])];t=pts[np.argmin(pts[:,0])]
 jp,jerr=project(j,path,arc);tp,terr=project(t,path,arc)
 jauto=route['endpoint']['measurement_distance_pixels'];tauto=result['T_path_pixels'];manual=ref['length_pixels']
 pieces={'endpoint_contribution_um':(jauto-jp)/.5299,'transition_contribution_um':(tp-tauto)/.5299,'route_contribution_um':(jp-tp-manual)/.5299}
 total=(jauto-tauto-manual)/.5299
 assert abs(sum(pieces.values())-total)<1e-6
 rows.append({'root':ref['root_id'],'manual_T_source_xy':t.tolist(),'manual_J_source_xy':j.tolist(),
  'automatic_T_source_xy':result['transition_xy'],'automatic_J_source_xy':result['endpoint_xy'],
  'manual_T_projection_distance_um':terr/.5299,'manual_J_projection_distance_um':jerr/.5299,
  'length_difference_um':total,**pieces})
out={'method':'Exact additive signed decomposition on the existing V2.1 automatic polyline, using nearest segment projection of manual T/J. Reference projection is for evaluation only, never inference. Positive = automatic length longer.',
 'reference_limit':'Only three manual ROIs on one development slide; no independent final accuracy.', 'rows':rows}
(D/'manual_error_decomposition.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
