"""Check the pre-existing image-only main-root continuity rule, separate from U-Net."""
from pathlib import Path
import importlib.util,json
import numpy as np
from PIL import Image
from predict import PAL
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('regional',ROOT/'03_Notebooks/35_primary_root_regions.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
truth=np.array(Image.open(ROOT/'04_Results/52_Reference_Check_20261116/reference_mask.png'))
for name in ['four_class_v1','baseline_refinement_v2']:
 folder=ROOT/'04_Results/54_Refinement_Experiments'/name;p=np.load(folder/'best_probabilities.npz')['probabilities'];mask,info=r.interpret(p);Image.fromarray(mask).save(folder/'regional_with_review_mask.png');Image.fromarray(PAL[mask]).save(folder/'regional_with_review_colours.png');info['metrics']=r.met(mask,truth);(folder/'regional_with_review.json').write_text(json.dumps(info,indent=2)+'\n');print(name,json.dumps(info))
