"""Compare two predicted appearance-presence outputs with the 3-way head.

Window labels come from expert review. This is not an individual-cell detector.
No expert T is passed to the presence classifier or its fixed boundary rule.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import shutil
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from appearance_experiment import ROOT, font

OUT=ROOT/'04_Results/65_Mixed_Appearance_Evidence_Check'
THREE=OUT/'three_way_five_windows'
OLD=ROOT/'04_Results/64_Appearance_Endoderm_Correction/added_endoderm_01'
DATA=ROOT/'02_Data/10_Cell_Appearance_Review'


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def boundary_rule(scores,distance):
    """Require both predicted appearances; do not force a mixed segment."""
    e=scores[:,0]>=.5;p=scores[:,1]>=.5
    state=np.full(len(scores),3,dtype=int)
    state[e&~p]=0;state[e&p]=1;state[~e&p]=2
    cut=np.r_[0,np.flatnonzero(np.diff(state))+1,len(state)]
    runs=[{'state':int(state[a]),'start':int(a),'stop_exclusive':int(b),'samples':int(b-a)} for a,b in zip(cut[:-1],cut[1:])]
    candidates=[]
    for i,r in enumerate(runs):
        if r['state']!=1 or r['samples']<3:continue
        prior=[s for s in runs[:i] if s['state']==0 and s['samples']>=3]
        later=[s for s in runs[i+1:] if s['state']==2 and s['samples']>=3]
        if not prior or not later:continue
        a=r['start']
        if not a:continue
        candidates.append({'mixed_run':r,'preceding_clear_Endoderm':prior[-1],
                           'following_clear_Periderm':later[0],
                           'T_distance_pixels':float((distance[a-1]+distance[a])/2)})
    return {'threshold':.5,'state_order':['Endoderm_only','both_mixed','Periderm_only','neither'],
            'window_states':state.tolist(),'runs':runs,'eligible_candidates':candidates,
            'selected':candidates[0] if candidates else None,
            'policy':'First >=3-window mixed run with an earlier >=3-window E-only run and later >=3-window P-only run. No manual T or offset.'}


def check_rule():
    for v in [[.99,.01],[.01,.99],[.99,.99],[.01,.01]]:
        assert boundary_rule(np.tile(v,(30,1)),np.arange(30)*64)['selected'] is None
    for n,expected in [(2,False),(4,True)]:
        scores=np.array([[.99,.01]]*8+[[.99,.99]]*n+[[.01,.99]]*8)
        result=boundary_rule(scores,np.arange(len(scores))*64)
        assert (result['selected'] is not None)==expected


def plot(distance,scores,rule,t,ref,out):
    im=Image.new('RGB',(1450,640),'white');d=ImageDraw.Draw(im)
    d.text((25,15),'Separate appearance predictions on root 3',font=font(29),fill='black')
    d.text((25,57),'A mixed call requires both outputs above0.5. Scores are not calibrated confidence.',font=font(22),fill='#334155')
    x0,y0,w,h=90,125,1300,270
    xx=lambda s:x0+w*s/distance[-1];yy=lambda p:y0+h*(1-p)
    for v in [0,.5,1]:d.line((x0,yy(v),x0+w,yy(v)),fill='#bbbbbb');d.text((38,yy(v)-10),str(v),font=font(19),fill='black')
    for k,color in enumerate(['#1aa65e','#db303f']):
        d.line([(xx(s),yy(p)) for s,p in zip(distance,scores[:,k])],fill=color,width=3)
    d.text((90,418),'Green: Endoderm-like appearance',font=font(22),fill='#14844b')
    d.text((660,418),'Red: Periderm-like appearance',font=font(22),fill='#db303f')
    colors=['#1aa65e','#b44dc5','#db303f','#777777']
    for i,k in enumerate(rule['window_states']):
        d.rectangle((xx(distance[i]-32),465,xx(distance[i]+32),500),fill=colors[k])
    for a,c in [(ref,'black'),(t,'#0078bc')]:
        if a is not None:d.line((xx(a),y0,xx(a),505),fill=c,width=2)
    d.text((90,521),'Purple: both predicted. Grey: neither supported. Black: reviewed T. Blue: model candidate.',font=font(21),fill='black')
    d.text((90,560),'Distance along supplied main-root path (native pixels)',font=font(21),fill='black')
    for v in np.arange(0,distance[-1]+1,4000):d.text((xx(v)-10,591),f'{v:,.0f}',font=font(19),fill='black')
    im.save(out)


def main():
    check_rule();torch.set_num_threads(4)
    labels=json.loads((DATA/'confirmed_appearance_labels_03.json').read_text())
    assert labels['confirmed'] and [e['class_index'] for e in labels['examples']]==[0,1,2,0,0]
    plan=json.loads((OUT/'presence_experiment_plan.json').read_text())
    # Fixed features use the identical published encoder and preprocessing as
    # the three-way comparison. Evaluation coordinates and labels do not train it.
    existing=np.load(THREE/'frozen_features.npz')
    scan_features=torch.from_numpy(existing['scan'])
    teaching_features=torch.from_numpy(existing['teaching'])
    scan_record=json.loads((THREE/'root3_prediction_before_reference_comparison.json').read_text())
    distance=np.array(scan_record['distance_from_left_end_native_pixels'])
    images=[np.array(Image.open(THREE/f'teaching_{e["id"]}_processed.png')) for e in labels['examples']]
    rng=np.random.default_rng(1617);augmented=[]
    for a in images:
        for j in range(48):
            z=a.astype(float)/255
            if j:
                z=np.clip(z**rng.uniform(.9,1.1)*rng.uniform(.8,1.2),0,1)
                if j%2:z=z[:,::-1]
                if j%4>=2:z=z[::-1]
            augmented.append((z*255).astype('uint8').copy())
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    encoder=smp.encoders.get_encoder('resnet34',in_channels=3,depth=5,weights=None)
    original=root_weights=ROOT/'00_Software/PAT_Reference/pat_state_dict.pt'
    weights=torch.load(original,map_location='cpu',weights_only=True)
    encoder.load_state_dict({k.removeprefix('encoder.'):v for k,v in weights.items() if k.startswith('encoder.')})
    encoder.to(device).eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    fs=[]
    with torch.inference_mode():
        for start in range(0,len(augmented),16):
            x=torch.from_numpy(np.stack(augmented[start:start+16]).transpose(0,3,1,2).copy()).float().to(device)/255
            layers=encoder((x-mean)/std)[1:]
            fs.append(F.normalize(torch.cat([F.normalize(z.mean((2,3)),dim=1) for z in layers],dim=1),dim=1).cpu())
    x_all=torch.cat(fs)
    assert torch.allclose(x_all[::48],teaching_features,atol=1e-5)
    target_map=np.array([[1,0],[1,1],[0,1]],dtype='float32')
    appearances=np.array([e['class_index'] for e in labels['examples']])
    outputs={}
    for n,name in [(4,'two_presence_four_windows'),(5,'two_presence_five_windows')]:
        dest=OUT/name;dest.mkdir(exist_ok=False)
        torch.manual_seed(1617);head=torch.nn.Linear(x_all.shape[1],2)
        optimizer=torch.optim.AdamW(head.parameters(),lr=.02,weight_decay=.05)
        counts=np.bincount(appearances[:n],minlength=3)
        group=np.repeat(appearances[:n],48)
        target=torch.from_numpy(target_map[group]);sample_weight=torch.from_numpy((1/counts[group]).astype('float32'))
        x=x_all[:n*48];history=[]
        save(dest/'plan.json',{**plan,'run_name':name,'unique_teaching_windows':n,
             'expert_examples':[e['id'] for e in labels['examples'][:n]],'class_window_counts':counts.tolist(),
             'encoder_sha256':sha(root_weights),'full_labels_sha256':sha(DATA/'confirmed_appearance_labels_03.json')})
        for update in range(1,501):
            optimizer.zero_grad();per_sample=F.binary_cross_entropy_with_logits(head(x),target,reduction='none').mean(1)
            loss=(per_sample*sample_weight).sum()/sample_weight.sum();loss.backward();optimizer.step()
            if update==1 or update%100==0:history.append({'update':update,'loss_before_update':float(loss.detach())})
        head.eval();torch.save(head.state_dict(),dest/'presence_head.pt')
        with torch.inference_mode():scores=head(scan_features).sigmoid().numpy();fit=head(teaching_features).sigmoid().numpy()
        rule=boundary_rule(scores,distance)
        # Save the prediction BEFORE reading reviewed T for scoring.
        prediction={'input':'Frozen image features from windows on a human-supplied main-root path',
                    'predicted_presence_scores':scores.tolist(),'output_order':['Endoderm_like','Periderm_like'],
                    'distance_native_pixels':distance.tolist(),'rule':rule,'reference_T_input':False,
                    'not_an_individual_cell_detector':True,'new_root3_correction_used':n==5}
        save(dest/'prediction_before_reference_comparison.json',prediction);save(dest/'training_history.json',history)
        save(dest/'teaching_scores.json',{'all_five_window_ids':[e['id'] for e in labels['examples']],
              'trained_ids':[e['id'] for e in labels['examples'][:n]],'scores':fit.tolist(),
              'expected_presence_targets':target_map[appearances].tolist(),
              'not_independent_accuracy':True})
        ref=json.loads((OLD/'result.json').read_text())
        t=rule['selected']['T_distance_pixels'] if rule['selected'] else None
        delta=None if t is None else t-ref['expert_working_T_distance_native_pixels']
        interval=ref['expert_approximate_T_interval_distance_native_pixels']
        result={'run_name':name,'teaching_windows':n,'candidate_T_distance_pixels':t,
                'difference_from_reviewed_T_pixels':delta,
                'difference_from_reviewed_T_um':None if delta is None else delta/.5299,
                'candidate_inside_reviewed_interval':False if t is None else interval[0]<=t<=interval[1],
                'new_error_field177_presence_scores':scores[177].tolist(),
                'new_error_field177_state':rule['window_states'][177],
                'region_still_familiar_teaching_check':True,'independent_validation':False,
                'default_model_promoted':False,'scores_not_calibrated_confidence':True}
        save(dest/'result.json',result);plot(distance,scores,rule,t,ref['expert_working_T_distance_native_pixels'],dest/'PRESENCE_PROFILE.png')
        shutil.copy2(__file__,dest/Path(__file__).name);outputs[name]=result
        print(json.dumps(result,indent=2),flush=True)
    old=json.loads((OLD/'result.json').read_text());extra=json.loads((THREE/'result.json').read_text())
    for name,r in [('three_way_four_windows',old),('three_way_five_windows',extra)]:
        outputs[name]={'candidate_T_distance_pixels':r['root3_T_candidate_distance_native_pixels'],
                      'difference_from_reviewed_T_pixels':r['difference_from_working_T_native_pixels'],
                      'difference_from_reviewed_T_um':r['difference_from_working_T_native_pixels']/.5299,
                      'candidate_inside_reviewed_interval':r['candidate_within_expert_interval']}
    save(OUT/'comparison_results.json',outputs)


if __name__=='__main__':main()
