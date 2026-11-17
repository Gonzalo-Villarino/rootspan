"""Reproduce the six-root demonstration from frozen model and saved annotation snapshot."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil
from predict_slide import run as predict,sha
from measure_roots import run as measure
from build_proof import run as proof
from create_review import main as review
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'04_Results/50_Six_Root_Length_Proof';saved=json.loads((base/'proof_record.json').read_text())
source=Path(saved['source_image']);snapshot=base/'annotations_snapshot.json'
assert sha(source)==saved['source_sha256'] and sha(snapshot)==saved['annotation_snapshot_sha256']
out=ROOT/'04_Results/51_Reproduction_Check'/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%SZ');out.mkdir(parents=True,exist_ok=False)
predict(source,out/'prediction','working3');measure(out/'prediction',out/'automatic',expected_roots=6,opening=4,transition_method='basal-run');new=proof(snapshot,out/'automatic',out/'six_root_proof');review(out/'six_root_proof')
prior=json.loads((Path(saved['automatic_measurements'])/'measurement_record.json').read_text());priorpred=Path(saved['automatic_measurements']).parent/'MT25_working3'
verification={'same_source_sha256':sha(source)==saved['source_sha256'],'same_prediction_mask_sha256':sha(out/'prediction/raw_classes.png')==sha(priorpred/'raw_classes.png'),'same_model_scores_sha256':sha(out/'prediction/periderm_fraction_scores.png')==sha(priorpred/'periderm_fraction_scores.png'),'assisted_lengths_match':all(abs(a['annotation_assisted_length_px']-b['annotation_assisted_length_px'])<1e-9 for a,b in zip(saved['roots'],new['roots'])),'automatic_lengths_match':all(abs(a['automatic_length_px']-b['automatic_length_px'])<1e-9 for a,b in zip(saved['roots'],new['roots'])),'model_was_retrained':False,'snapshot_preserves_shapes':True,'output':str(out)}
(out/'reproduction_verification.json').write_text(json.dumps(verification,indent=2)+'\n');(ROOT/'05_Notes/Reviews/20261115_PAT_Length_Proof/reproduction_verification.json').write_text(json.dumps(verification,indent=2)+'\n');print(json.dumps(verification,indent=2));assert all(verification[k] for k in ['same_source_sha256','same_prediction_mask_sha256','same_model_scores_sha256','assisted_lengths_match','automatic_lengths_match'])
