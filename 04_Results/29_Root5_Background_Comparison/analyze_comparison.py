"""Compare two saved predictions against the same updated root-5 reference.
This analysis reads saved predictions; it does not train or run either model.
"""
from pathlib import Path
import sys, json, hashlib, subprocess
import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parent
PROJECT=OUT.parents[1]
plan=json.loads((OUT/'comparison_plan.json').read_text())
oldview=Path(plan['baseline_prediction'])
newview=Path(sys.argv[1]).resolve()
oldpair=Path(plan['baseline_pair']); newpair=Path(plan['updated_pair'])
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def array(p):
    with Image.open(p) as f:return np.array(f)
a=read(oldpair/'patch_details.json');b=read(newpair/'patch_details.json')
od=read(oldview/'prediction_details.json');nd=read(newview/'prediction_details.json')
ot=read(Path(od['sources']['run_folder'])/'trial_details.json')
nt=read(Path(nd['sources']['run_folder'])/'trial_details.json')
for record,view,details,pair in [(ot,od,a,oldpair),(nt,nd,b,newpair)]:
    assert sha(Path(view['sources']['run_folder'])/'model_weights.pt')==record['model_weights_sha256']==view['sources']['weights_sha256']
    assert sha(pair/details['image_file'])==details['image_sha256']==view['sources']['image_sha256']
    assert sha(pair/details['mask_file'])==details['mask_sha256']==view['sources']['mask_sha256']
for k in ['seed','optimizer','learning_rate','gradient_clip_norm','image_preprocessing','requested_updates','completed_updates','max_loop_seconds','training_script_sha256','model_script_sha256','base_model_script_sha256']:
    assert ot[k]==nt[k],k
assert nt['completed_updates']==600
assert a['image_sha256']==b['image_sha256']
truth0=array(oldpair/a['mask_file']);truth=array(newpair/b['mask_file'])
before=array(oldview/'predicted_mask.png');after=array(newview/'predicted_mask.png')
assert truth.shape==truth0.shape==before.shape==after.shape==(1024,1024)
assert np.isin(before,[1,2,3]).all() and np.isin(after,[1,2,3]).all()
changed=truth0!=truth
assert int(changed.sum())==193215 and np.all(truth0[changed]==0) and np.all(truth[changed]==3)
labels=['Periderm','Endoderm','Background']
def metrics(pred,reference):
    conf=np.array([[np.count_nonzero((reference==i)&(pred==j)) for j in [1,2,3]] for i in [1,2,3]])
    return {'confusion_rows_reference_columns_prediction':conf.tolist(),
            'labelled_pixels':int(conf.sum()),'correct':int(conf.trace()),
            'per_class':{name:{'reference_pixels':int(conf[k].sum()),'correct_pixels':int(conf[k,k]),'percent_matched':100*float(conf[k,k])/int(conf[k].sum())} for k,name in enumerate(labels)}}
assert metrics(before,truth0)['confusion_rows_reference_columns_prediction']==ot['after_training']['confusion_rows_reference_columns_prediction']
assert metrics(after,truth)['confusion_rows_reference_columns_prediction']==nt['after_training']['confusion_rows_reference_columns_prediction']
yy=np.arange(1024)[:,None];peri=truth==1;cols=peri.any(axis=0);bottom=np.where(peri,yy,-1).max(axis=0)
lower=cols[None,:]&(yy>bottom[None,:])
oldgreen=lower&(before==2);newgreen=lower&(after==2)
def counts(values):return {str(int(k)):int(n) for k,n in zip(*np.unique(values,return_counts=True))}
result={
 'purpose':'Compare effect of added lower Background labels on the same root-5 training example',
 'baseline_prediction':str(oldview),'updated_prediction':str(newview),
 'baseline_run':od['sources']['run_folder'],'updated_run':nd['sources']['run_folder'],
 'image_patch':str(newpair/b['image_file']),'numeric_updated_mask':str(newpair/b['mask_file']),
 'only_input_change':'193,215 formerly unlabelled pixels become Background; identical image and tissue labels',
 'same_code_and_settings_verified':True,'completed_updates_each':600,
 'baseline_scored_against_updated_reference':metrics(before,truth),
 'updated_scored_against_updated_reference':metrics(after,truth),
 'lower_band':{'definition':'In columns with labelled Periderm, pixels below its bottommost labelled pixel',
     'old_green_pixels':int(oldgreen.sum()),'new_green_pixels':int(newgreen.sum()),
     'green_pixel_reduction_percent':100*(1-int(newgreen.sum())/int(oldgreen.sum())),
     'updated_reference_codes_at_old_green_pixels':counts(truth[oldgreen]),
     'updated_reference_codes_at_new_green_pixels':counts(truth[newgreen]),
     'new_predictions_at_old_green_pixels':counts(after[oldgreen]),
     'new_green_not_present_before':int(np.count_nonzero(newgreen&~oldgreen))},
 'newly_labelled_background':{'pixels':int(changed.sum()),'before_predictions':counts(before[changed]),'after_predictions':counts(after[changed])},
 'reference_counts':counts(truth),'labelled_percent':float(100*np.isin(truth,[1,2,3]).mean()),
 'limits':['No new slide or root was tested; this is the same training example.',
           'Unlabelled edge pixels have no confirmed answer; a colour change alone does not establish biological correctness.',
           'One fresh run per label set, same seed and settings; not replicated evidence of general performance.'],
 'further_training_or_inference_in_this_analysis':False}
(OUT/'comparison_results.json').write_text(json.dumps(result,indent=2)+'\n')
subprocess.run(['/usr/local/bin/Rscript',str(OUT/'draw_comparison.R'),str(OUT)],check=True)
print(json.dumps({'lower_band':result['lower_band'],'old_on_updated_reference':result['baseline_scored_against_updated_reference']['per_class'],'new_on_updated_reference':result['updated_scored_against_updated_reference']['per_class'],'newly_labelled_background':result['newly_labelled_background'],'labelled_percent':result['labelled_percent']},indent=2))
print('Figure:',OUT/'root5_before_after_closeup.png')
