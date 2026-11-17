"""Image-only periderm prediction and provisional lengths for a new horizontal-root slide.
Requires original-scale8-bit RGB. Right-hand basal orientation must be confirmed.
This exploratory pipeline does not replace anatomical review or manual validation.
"""
from pathlib import Path
import argparse,json
from predict_slide import run as predict
from measure_roots import run as measure
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--image',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--expected-roots',type=int);p.add_argument('--pixels-per-um',type=float);p.add_argument('--basal-side',required=True,choices=['right']);p.add_argument('--model',choices=['working3','published4'],default='working3');a=p.parse_args()
 if a.output.exists():raise FileExistsError('Choose a new output directory; previous results are preserved.')
 a.output.mkdir(parents=True);predict(a.image,a.output/'prediction',a.model)
 measure(a.output/'prediction',a.output/'measurements',px_per_um=a.pixels_per_um,calibration_status='User-supplied for this input' if a.pixels_per_um else 'Unconfirmed; pixels only',expected_roots=a.expected_roots,opening=4,transition_method='basal-run')
 (a.output/'START_HERE.txt').write_text('Open measurements/ALL_ROOTS_MEASUREMENT_LINES.png. Review every transition, basal endpoint and main-root path before accepting lengths. The CSV is provisional. No annotation mask was used and no model was trained.\n')
 print('Saved provisional measurements in '+str(a.output.resolve()))
