"""Compare old/new image-only predictions, then read masks for scoring.

Root3 is a teaching-fit diagnostic. 20_29 is the familiar development image
used for checkpoint selection. Neither is presented as a new independent test.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from predict import load, predict, PAL
from check_reference import metrics
from main_root_rule import interpret

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'04_Results/62_Reviewed_Root3_Training_Comparison'
NEW=ROOT/'04_Results/54_Refinement_Experiments/reviewed_root3_v9'
OLD=ROOT/'04_Results/54_Refinement_Experiments/rule_derived_endoderm_v7'
DATA=ROOT/'04_Results/61_Reviewed_Root3_Teaching_Correction'
DEV=ROOT/'04_Results/52_Reference_Check_20261116'


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def dump(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def gate(m):
    c=m['per_class']
    return bool(all(c[k]['recall'] is not None and c[k]['precision'] is not None and
                    c[k]['recall']>=.85 and c[k]['precision']>=.85 for k in ['Periderm','Endoderm'])
                and c['Background']['recall'] is not None and c['Background']['recall']>=.98
                and m['tissue_mean_iou']>=.75)


def figure(title,scope,rgb,truth,old,new,box,output,metrics_old,metrics_new,T=None):
    font=lambda n:ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',n)
    width=740;height=round((box[3]-box[1])*width/(box[2]-box[0]))
    canvas=Image.new('RGB',(1540,2*(height+58)+240),'white');d=ImageDraw.Draw(canvas)
    d.text((20,12),title,font=font(30),fill='black')
    d.text((20,55),scope,font=font(23),fill='#273444')
    panels=[('1  Microscopy image',rgb),('2  Reviewed answer key',PAL[truth]),
            ('3  Earlier model — raw prediction',PAL[old]),('4  New model — raw prediction',PAL[new])]
    for i,(label,arr) in enumerate(panels):
        x=20+(i%2)*760;y=100+(i//2)*(height+58)
        d.text((x,y),label,font=font(24),fill='black')
        photo=Image.fromarray(arr).crop(box).resize((width,height),Image.Resampling.LANCZOS if i==0 else Image.Resampling.NEAREST)
        if T is not None:
            tx=(T-box[0])*width/(box[2]-box[0]);pd=ImageDraw.Draw(photo)
            for yy in range(0,height,16):pd.line((tx,yy,tx,min(yy+8,height-1)),fill='white',width=2)
            pd.text((tx+8,8),'T',font=font(22),fill='white',stroke_width=2,stroke_fill='black')
        canvas.paste(photo,(x,y+36))
    y=100+2*(height+58)
    a=metrics_old['per_class'];b=metrics_new['per_class']
    d.text((20,y),f'Match within labelled pixels only: Periderm {100*a["Periderm"]["recall"]:.1f}% → {100*b["Periderm"]["recall"]:.1f}%     Endoderm {100*a["Endoderm"]["recall"]:.1f}% → {100*b["Endoderm"]["recall"]:.1f}%',font=font(24),fill='black')
    d.text((20,y+37),'Red = Periderm | Green = Endoderm | Yellow = Background | Purple = lateral root',font=font(23),fill='black')
    d.text((20,y+73),'Grey in the answer key = unlabelled. Predictions cover every pixel; unlabelled space is not scored.',font=font(23),fill='black')
    d.text((20,y+108),'Raw model output shown. No manual boundary or tissue mask was copied into either prediction.',font=font(22),fill='#374151')
    canvas.save(output)


def main():
    assert (NEW/'result.json').exists(),'Training must finish before the comparison.'
    output=OUT/'predictions';output.mkdir(exist_ok=False)
    before=json.loads((OUT/'before_training_record.json').read_text())
    input_paths={'root3':DATA/'images/COL0_root3_reviewed_T_x12237_y5643.png',
                 'development_20_29':DEV/'image.png'}
    rgbs={name:np.array(Image.open(path)) for name,path in input_paths.items()}
    results={};rules={}
    # Pure inference: no reference masks or expert T coordinates enter here.
    for model_name,folder in [('old_v7',OLD),('new_v9',NEW)]:
        weights=folder/'best_weights.pt';model,device=load(weights)
        for example,rgb in rgbs.items():
            dest=output/model_name/example;dest.mkdir(parents=True)
            probability=predict(model,device,rgb)
            classes=(probability.argmax(0)+1).astype(np.uint8)
            rule_classes,rule_details=interpret(probability)
            Image.fromarray(classes).save(dest/'raw_prediction.png')
            Image.fromarray(PAL[classes]).save(dest/'raw_prediction_colours.png')
            Image.fromarray(rule_classes).save(dest/'direction_rule_prediction.png')
            np.savez_compressed(dest/'probabilities.npz',probabilities=probability)
            dump(dest/'inference_record.json',{'input_image':str(input_paths[example]),
                'input_image_sha256':sha(input_paths[example]),'weights':str(weights),'weights_sha256':sha(weights),
                'prediction_input':'RGB image only','reference_input':False,'expert_T_input':False,
                'native_crop_dimensions':list(rgb.shape[:2][::-1]),'raw_postprocessing':'None',
                'direction_rule_separate_from_raw':rule_details,'device':device})
            results[(model_name,example)]=classes;rules[(model_name,example)]=rule_details
        del model
        if torch.backends.mps.is_available():torch.mps.empty_cache()

    # Only now load answer keys for comparison. The directional rule above
    # was also computed before loading expert positions or reference masks.
    correction=json.loads((DATA/'root3_class_correction.json').read_text())
    masks={'root3':np.array(Image.open(correction['new_transition_patch']['mask'])),
           'development_20_29':np.array(Image.open(DEV/'reference_mask.png'))}
    scores={}
    for example,truth in masks.items():
        scores[example]={}
        for model_name in ['old_v7','new_v9']:
            m=metrics(results[(model_name,example)],truth)
            m['scope']=('Only labelled interior teaching pixels of COL0 root3; not an independent test. No labelled Background or lateral-root pixels in this mask.' if example=='root3' else 'Only annotated pixels of familiar20_29development crop, reused for checkpoint selection; not a blind test.')
            scores[example][model_name]=m
    # Check the rerun agrees with each saved checkpoint-selection prediction.
    for name,folder in [('old_v7',OLD),('new_v9',NEW)]:
        saved=np.array(Image.open(folder/'best_prediction.png'))
        assert np.array_equal(saved,results[(name,'development_20_29')])
        original=json.loads((folder/'best_result.json').read_text())
        assert np.isclose(original['metrics']['tissue_mean_iou'],scores['development_20_29'][name]['tissue_mean_iou'])
    t_local=correction['working_T_source_x']-correction['new_transition_patch']['box'][0]
    interval=[v-correction['new_transition_patch']['box'][0] for v in correction['reviewed_approximate_T_zone_source_x']]
    direction={}
    for model_name in ['old_v7','new_v9']:
        rd=rules[(model_name,'root3')];b=rd.get('boundary_x')
        direction[model_name]={'image_only_rule':rd,'expert_working_T_x_in_patch':t_local,
            'expert_approximate_T_zone_x_in_patch':interval,
            'boundary_distance_from_working_T_pixels':abs(b-t_local) if b is not None else None,
            'boundary_distance_from_reviewed_zone_pixels':max(interval[0]-b,b-interval[1],0) if b is not None else None,
            'scope':'Additional image-only one-E-left/P-right-change interpretation, not raw segmentation; evaluated on a teaching crop. No automatic J or full periderm length assessed.'}
    figure('COL0 root 3 — did correcting the answer key help?',
        'Teaching example: the new model trained on this corrected root. This is a learning check.',
        rgbs['root3'],masks['root3'],results[('old_v7','root3')],results[('new_v9','root3')],
        (200,250,768,550),OUT/'ROOT3_BEFORE_AND_AFTER.png',scores['root3']['old_v7'],scores['root3']['new_v9'],t_local)
    figure('20_29 — check a different slide without teaching it to this run',
        'Familiar development example, used to choose checkpoints. This is not a new blind test.',
        rgbs['development_20_29'],masks['development_20_29'],results[('old_v7','development_20_29')],results[('new_v9','development_20_29')],
        (0,300,1024,640),OUT/'20_29_BEFORE_AND_AFTER.png',scores['development_20_29']['old_v7'],scores['development_20_29']['new_v9'])
    for key in ['old_v7','same_v6_initial','default_working_model','historical_trainer']:
        assert sha(before[key]['path'])==before[key]['sha256']
    model_result=json.loads((NEW/'result.json').read_text())
    new_checks=json.loads((NEW/'checks.json').read_text())
    old_checks=json.loads((OLD/'checks.json').read_text())
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'training':model_result,
        'comparison':scores,'root3_direction_rule_diagnostic':direction,
        'development_gate_passed':gate(scores['development_20_29']['new_v9']),
        'development_gate':'Both tissue precision/recall>=.85; annotatedBGrecall>=.98; tissue meanIoU>=.75.',
        'best_checkpoint_old':json.loads((OLD/'best_result.json').read_text())['step'],
        'best_checkpoint_new':model_result['best_step'],
        'matched_step_500_development':{'old_v7':next(x for x in old_checks if x['step']==500),
                                        'new_v9':next(x for x in new_checks if x['step']==500)},
        'matched_final_step_development':{'old_v7':old_checks[-1],'new_v9':new_checks[-1]},
        'no_new_blind_test':True,'independent_length_validation':False,
        'default_model_unchanged':True,'prior_models_unchanged':True,
        'no_automatic_length_claim':'Human reviewed route and reference lengths are separate. Neither this local correction nor these checks demonstrate automatic T/J recognition and accurate lengths on new slides.',
        'data_change_scope':'Corrected labels plus one additional transition-centred crop; crop sampling also changes. This is not a pure one-variable label-only causal experiment.'}
    dump(OUT/'comparison_results.json',report)
    dump(OUT/'verification.json',{'pure_inference_before_reference_loading':True,
        'saved_and_repeated_development_predictions_identical':True,
        'old_weights_and_default_model_hashes_unchanged':True,
        'labels_only_used_for_scoring':True,'new_training_complete':True,
        'completed_updates':model_result['completed_steps'],'figure_visual_review_pending':True})
    shutil.copy2(__file__,OUT/Path(__file__).name)
    print(json.dumps({'scores':scores,'direction_root3':direction,
        'best_new_step':model_result['best_step'],'development_gate_passed':report['development_gate_passed']},indent=2))


if __name__=='__main__':main()
