"""One preregistered trial; no coordinates or labels from evaluation enter fitting."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

D=Path(__file__).resolve().parent; P=D.parent
B=P/'06_Models/Locked_Transition_v1_20261116'
sys.path.insert(0,str(B))
from frozen_core import continuity_rule, run_checks

def save(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    out=D/'expanded_fields_trial';out.mkdir(exist_ok=False)
    audit=json.loads((D/'existing_field_labels/audit_manifest.json').read_text())
    excluded={'A04_r1_B','A04_r2_B','A04_r3_B'}
    rows=audit['records']
    for r in rows:
        r['selected_for_training']=r['id'] not in excluded
        r['decision_reason']='Existing explicit appearance label, retained with field-level limitations' if r['selected_for_training'] else 'Unresolved relationship between earlier field appearance and later T reference; excluded, not relabelled'
    selected=[r for r in rows if r['selected_for_training']]
    assert len(selected)==11
    for r in selected:
        n=sum(x['class']==r['class'] and x['slide']==r['slide'] for x in selected)
        r['total_field_weight']=1/(3*2*n)
    save(out/'field_selection.json',rows)
    files=[Path(__file__),D/'EXPANDED_FIELDS_PLAN.txt',B/'frozen_core.py',P/'00_Software/PAT_Reference/pat_state_dict.pt']+[P/r['image'] for r in rows]+[P/r['label_record'] for r in rows]
    save(out/'frozen_design_sha256.json',{str(f.relative_to(P)):sha(f) for f in files})
    torch.set_num_threads(4);run_checks()
    images=[np.array(Image.open(P/r['image']).convert('RGB')) for r in selected]
    rng=np.random.default_rng(1617);aug=[]
    for a in images:
        assert a.shape==(256,320,3)
        for j in range(48):
            z=a.astype(float)/255
            if j:
                z=np.clip(z**rng.uniform(.9,1.1)*rng.uniform(.8,1.2),0,1)
                if j%2:z=z[:,::-1]
                if j%4>=2:z=z[::-1]
            aug.append((z*255).astype('uint8').copy())
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    enc=smp.encoders.get_encoder('resnet34',in_channels=3,depth=5,weights=None)
    weights=torch.load(P/'00_Software/PAT_Reference/pat_state_dict.pt',map_location='cpu',weights_only=True)
    enc.load_state_dict({k.removeprefix('encoder.'):v for k,v in weights.items() if k.startswith('encoder.')})
    enc.to(device).eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    fs=[]
    with torch.inference_mode():
        for start in range(0,len(aug),16):
            x=torch.from_numpy(np.stack(aug[start:start+16]).transpose(0,3,1,2).copy()).float().to(device)/255
            layers=enc((x-mean)/std)[1:]
            fs.append(F.normalize(torch.cat([F.normalize(z.mean((2,3)),dim=1) for z in layers],dim=1),dim=1).cpu())
            if start%128==0:print('Encoded',min(start+16,len(aug)),'/',len(aug),flush=True)
    x=torch.cat(fs);np.savez_compressed(out/'teaching_features.npz',features=x.numpy())
    original=np.load(P/'04_Results/65_Mixed_Appearance_Evidence_Check/three_way_five_windows/frozen_features.npz')['teaching']
    assert np.allclose(x[::48][:5].numpy(),original,atol=1e-5)
    targets=np.array([[1,0],[1,1],[0,1]],dtype='float32')
    y=torch.from_numpy(np.repeat(targets[[r['class'] for r in selected]],48,axis=0))
    w=torch.tensor(np.repeat([r['total_field_weight']/48 for r in selected],48),dtype=torch.float32)
    assert abs(float(w.sum())-1)<1e-6
    torch.manual_seed(1617);head=torch.nn.Linear(1024,2)
    opt=torch.optim.AdamW(head.parameters(),lr=.02,weight_decay=.05);history=[]
    for step in range(1,501):
        opt.zero_grad();loss=(F.binary_cross_entropy_with_logits(head(x),y,reduction='none').mean(1)*w).sum();loss.backward();opt.step()
        if step==1 or step%100==0:history.append({'step':step,'loss_before':float(loss.detach())})
    head.eval().requires_grad_(False)
    with torch.inference_mode():
        final_loss=float((F.binary_cross_entropy_with_logits(head(x),y,reduction='none').mean(1)*w).sum())
        fitted=head(x[::48]).sigmoid().numpy()
    torch.save(head.state_dict(),out/'experimental_presence_head.pt')
    fitrows=[{'id':r['id'],'expected':targets[r['class']].tolist(),'scores':s.tolist(),'matched':bool(np.array_equal(s>=.5,targets[r['class']].astype(bool)))} for r,s in zip(selected,fitted)]
    save(out/'training.json',{'history':history,'final_loss':final_loss,'weight_norm':float(head.weight.norm()),'fields':fitrows,'augmented_images':len(aug),'unique_fields':11,'unique_slides':2,'not_independent_accuracy':True})
    cases={
      'COL0_root3':('04_Results/65_Mixed_Appearance_Evidence_Check/three_way_five_windows/frozen_features.npz','scan'),
      'A04_root3':('04_Results/67_Fixed_A04_Root3_Check/frozen_features_and_scores.npz','features'),
      '2_12_root1':('04_Results/68_Fixed_212_Root1_Check/frozen_features_and_scores.npz','features'),
      '20_29_root4':('04_Results/66_Fixed_20_29_Check/frozen_features_and_scores.npz','features')}
    checks=[]
    for name,(f,key) in cases.items():
        a=np.load(P/f)[key];profile=json.loads((B/'fixtures/profiles'/f'{name}.json').read_text());q=profile['prediction']
        distance=np.array(q.get('distance_along_local_path',q.get('distance_native_pixels')))
        with torch.inference_mode():scores=head(torch.from_numpy(a).float()).sigmoid().numpy()
        rule=continuity_rule(scores,distance);t=rule['selected']['T_distance_pixels'] if rule['selected'] else None
        save(out/f'{name}_prediction.json',{'distance':distance.tolist(),'scores':scores.tolist(),'rule':rule,'T':t,'manual_T_used_by_model':False})
        expected=profile['expected_T_path_pixels'];passed=t is None if expected is None else t is not None and abs(t-expected)<=640
        checks.append({'root':name,'reference_T':expected,'candidate_T':t,'passed':passed})
    gate={'passed':all(r['matched'] for r in fitrows) and all(r['passed'] for r in checks),'teaching_fields_all_matched':all(r['matched'] for r in fitrows),'checks':checks,'one_seven_features_read':False,'promoted':False,'independent_validation':False}
    save(out/'gate.json',gate);print(json.dumps(gate,indent=2),flush=True)
    print('STOP: candidate rejected; no 1_7 evaluation.' if not gate['passed'] else 'Gate passed; candidate still needs the fixed 1_7 development check.',flush=True)

if __name__=='__main__':main()
