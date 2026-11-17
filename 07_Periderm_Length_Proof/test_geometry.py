"""Small meaningful checks for geometric length, units, and saved-result consistency."""
import json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from measure_roots import arc_length,pixels_to_um,skeleton_path,close_short_gaps
ROOT=Path(__file__).resolve().parents[1]
checks=[]
def check(name,condition):
 assert bool(condition),name
 checks.append(name)
check('Euclidean diagonal uses sqrt(dx2+dy2)',abs(arc_length(np.array([[0,0],[3,4],[6,8]]))[-1]-10)<1e-12)
check('Translation does not alter length',arc_length(np.array([[30,40],[33,44],[36,48]]))[-1]==10)
check('Calibration is pixels divided by pixels per micrometre',abs(pixels_to_um(5299,.5299)-10000)<1e-9)
for invalid in [0,-1,float('nan'),float('inf')]:
 try:pixels_to_um(1,invalid)
 except ValueError:pass
 else:raise AssertionError('invalid calibration accepted')
checks.append('Nonpositive or nonfinite calibration rejected')
mask=np.zeros((100,600),bool);mask[40:61,20:521]=True;path,_=skeleton_path(mask)
check('Straight 500-pixel centreline recovered within 2 pixels',abs(arc_length(path)[-1]-500)<2)
xs=np.arange(20,521);ys=60+20*np.sin((xs-20)/500*2*np.pi);truth=np.column_stack([xs,ys]);im=Image.new('1',(550,130));ImageDraw.Draw(im).line(list(map(tuple,truth)),fill=1,width=17);path,_=skeleton_path(np.array(im))
check('Smooth curved centreline recovered within 3 percent',abs(arc_length(path)[-1]/arc_length(truth)[-1]-1)<.03)
f=np.array([False,True,False,True,False]);a=np.array([0.,50,100,150,200]);out=close_short_gaps(f,a,150)
check('Only enclosed short gaps filled; outer background not filled',np.array_equal(out,[False,True,True,True,False]))
r=json.loads((ROOT/'04_Results/50_Six_Root_Length_Proof/proof_record.json').read_text());check('Exactly six numbered roots', [x['root_id'] for x in r['roots']]==list(range(1,7)))
for root in r['roots']:
 check(f'Root {root["root_id"]} saved length matches its coordinates',abs(arc_length(np.array(root['path_xy']))[-1]-root['annotation_assisted_length_px'])<1e-6)
 check(f'Root {root["root_id"]} no invented manual measurement',root['manual_fiji_length_px'] is None)
 check(f'Root {root["root_id"]} path within image',((np.array(root['path_xy'])>=0)&(np.array(root['path_xy'])<np.array([15423,9543]))).all())
out={'all_passed':True,'checks':checks};(ROOT/'05_Notes/Reviews/20261115_PAT_Length_Proof/geometry_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
