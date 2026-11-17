"""Four-class PAT U-Net++ refinement on four teaching slides only.
20_29 human reference is a familiar development check, NEVER training input.
"""
from pathlib import Path
import argparse,json,time,hashlib,shutil
from datetime import datetime,timezone
import numpy as np
from PIL import Image,ImageFilter
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from check_reference import metrics
from preprocess import prepare
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'04_Results/59_Rule_Derived_Teaching_Data';REF=ROOT/'04_Results/52_Reference_Check_20261116'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=800);p.add_argument('--name',default='four_class_v1');p.add_argument('--initial',type=Path);p.add_argument('--contrast',choices=['none','grayscale_clahe'],default='none');p.add_argument('--train-all',action='store_true');p.add_argument('--update-bn',action='store_true');p.add_argument('--augment',choices=['varied','mild'],default='varied');p.add_argument('--lr',type=float,default=0.0001);a=p.parse_args();assert 1<=a.steps<=1500
 out=ROOT/'04_Results/54_Refinement_Experiments'/a.name;out.mkdir(parents=True,exist_ok=False)
 rng=np.random.default_rng(1611);torch.manual_seed(1611);torch.set_num_threads(4);device='mps' if torch.backends.mps.is_available() else 'cpu'
 manifest=json.loads((DATA/'manifest.json').read_text());assert {s['slide'] for s in manifest['sources']}=={'MT25_rep3','20#MT28_gfp','25717_gfp','COL0_Slide1'}
 pairs=[]
 for p in manifest['patches']:
  assert sha(DATA/p['image'])==p['image_sha256'] and sha(DATA/p['mask'])==p['mask_sha256'];pairs.append((prepare(np.array(Image.open(DATA/p['image'])),a.contrast),np.array(Image.open(DATA/p['mask']))))
 mixed=[i for i,p in enumerate(manifest['patches']) if p['counts']['1']>1000 and p['counts']['2']>1000]
 pools={c:[i for i,p in enumerate(manifest['patches']) if p['counts'][str(c)]>1000] for c in [1,2,3,4]};assert mixed and all(pools.values())
 new_mixed=[i for i in mixed if manifest['patches'][i]['slide']=='COL0_Slide1'];old_mixed=[i for i in mixed if manifest['patches'][i]['slide']!='COL0_Slide1'];old_pools={c:[i for i in pools[c] if manifest['patches'][i]['slide']!='COL0_Slide1'] for c in [1,2]};assert new_mixed and old_mixed and all(old_pools.values())
 initial=a.initial or ROOT/'00_Software/PAT_Reference/pat_state_dict.pt';state=torch.load(initial,map_location='cpu',weights_only=True)
 if a.initial is None:
  state['segmentation_head.0.weight']=state['segmentation_head.0.weight'][[1,2,0,3]].clone();state['segmentation_head.0.bias']=state['segmentation_head.0.bias'][[1,2,0,3]].clone()
 model=smp.UnetPlusPlus(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=4,activation=None);model.load_state_dict(state,strict=True);model.to(device)
 for p in model.encoder.parameters():p.requires_grad_(False)
 for layer in [model.encoder.layer2,model.encoder.layer3,model.encoder.layer4]:
  for p in layer.parameters():p.requires_grad_(True)
 if a.train_all:
  for p in model.encoder.parameters():p.requires_grad_(True)
 opt=torch.optim.Adam([{'params':list(model.decoder.parameters())+list(model.segmentation_head.parameters()),'lr':a.lr},{'params':[p for p in model.encoder.parameters() if p.requires_grad],'lr':a.lr/10}]);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=1e-6)
 mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
 devrgb=np.array(Image.open(REF/'image.png'));devmask=np.array(Image.open(REF/'reference_mask.png'));vx=torch.from_numpy(prepare(devrgb,a.contrast).transpose(2,0,1).copy()).float()[None].to(device)/255
 # Dev tensor is used only inside inference_mode; never appended to training pairs.
 plan={'created_utc':datetime.now(timezone.utc).isoformat(),'data_manifest':str(DATA/'manifest.json'),'manifest_sha256':sha(DATA/'manifest.json'),'initial_weights':str(initial),'initial_sha256':sha(initial),'role':'Development-guided refinement, not blind final testing','training_sources':[s['slide'] for s in manifest['sources']],'training_root_count':20,'development_image':str(REF/'image.png'),'reference_sha256':sha(REF/'reference_mask.png'),'dev_in_training':False,'reserved_test_used':False,'class_order':['Periderm','Endoderm','Background','Lateral_root'],'class_codes':[1,2,3,4],'unknown_and_conflicts_ignored':[0,255],'initial_ancestry':'Public PAT training-image membership is unknown. No local20_29-adapted model used.','augmentation':'Each training field: random512-768 crop resized to512, paired flips/90degree rotations; gain0.55-1.7, gamma0.75-1.3, occasional mild Gaussian blur. No geometric labels invented.','batch':'4 fields:2mixed-transition,1rotatingP/E/LR,1explicitBackground. Class-balanced CE+.5Dice. All batchnorm running stats frozen.','trainable':'decoder/head andencoderstages2-4','learning_rate_decoder':a.lr,'seed':1611,'updates_requested':a.steps,'selection':'Highest raw development mean Periderm/Endoderm IoU; score every50updates. Save full trace and never represent as an independent test. No forced left/right boundary.','target_for_success_on_this_crop':'Both tissue recalls>=.85, both tissue precisions>=.85, labelled Background recall>=.98, tissue meanIoU>=.75. Even passing is not validation on hundreds of images.'}
 plan['augmentation_mode']=a.augment
 plan['contrast_preprocessing']=a.contrast
 plan['trainable']='all encoder plus decoder/head' if a.train_all else plan['trainable']
 plan['batchnorm_running_statistics']='updated on teaching batches only' if a.update_bn else 'frozen'
 if a.augment=='mild':plan['augmentation']='Native512 crop, paired rotations/flips, gain0.8-1.2. No resizing/gamma/blur.'
 if a.initial is not None:plan['initial_ancestry']='Continues four-class expanded_transition_balancing_v6 best950. Original ancestry baseline39 with publishedPAT LRhead; no20_29gradient training. PublicPAT exposure unknown.'
 plan['class_support']='14originalroots savedP/E;6additionalCOL0 savedP and experimental rule-derivedE. Do not present derivedE as human-drawn truth.'
 plan['rule_manifest']=manifest['rule_regions'];plan['rule_manifest_sha256']=manifest['rule_regions_sha256']
 plan['rule_label_weight']=.5;plan['previous_A04_check_already_observed']=True;plan['reserved_2_12_used']=False
 plan['next_check_status']='20_29 remainsfamiliar development;A04wascheckedbeforethisexperimentandcannotbeclaimedfreshblindfinaltest.'
 plan['batch']='6fields:2COL0rulemixed,1oldmixed,1oldP/Ealternating,1LR,1explicitBG. Per-field class-balancedCE+.5Dice, weight.5forrule-labelledfields,1otherwise. Mixed crops retry12times toretain>=512pixels eachP/E.'
 save(out/'experiment_plan.json',plan);shutil.copy2(__file__,out/Path(__file__).name);shutil.copy2(Path(__file__).with_name('check_reference.py'),out/'check_reference.py');shutil.copy2(Path(__file__).with_name('preprocess.py'),out/'preprocess.py')
 start=time.monotonic();best=-1;checks=[];history=[]
 def crop(i):
  rgb,ma=pairs[i];n=512 if a.augment=='mild' else int(rng.choice([512,640,768]))
  for retry in range(12):
   x=int(rng.integers(769-n));y=int(rng.integers(769-n));sub=ma[y:y+n,x:x+n]
   if i not in mixed or all((sub==c).sum()>=512 for c in [1,2]):break
  rgb=rgb[y:y+n,x:x+n];ma=sub
  rgb=Image.fromarray(rgb).resize((512,512),Image.Resampling.BILINEAR);ma=np.array(Image.fromarray(ma).resize((512,512),Image.Resampling.NEAREST))
  if a.augment=='varied' and rng.random()<.25:rgb=rgb.filter(ImageFilter.GaussianBlur(float(rng.uniform(.1,.7))))
  im=np.array(rgb).astype('float32')/255
  im=np.clip(im*rng.uniform(.8,1.2),0,1) if a.augment=='mild' else np.clip(im**rng.uniform(.75,1.3)*rng.uniform(.55,1.7),0,1)
  k=int(rng.integers(4));im=np.rot90(im,k);ma=np.rot90(ma,k)
  if rng.random()<.5:im=im[:,::-1];ma=ma[:,::-1]
  target=np.full(ma.shape,255,dtype='int64');v=np.isin(ma,[1,2,3,4]);target[v]=ma[v]-1
  return torch.from_numpy(im.transpose(2,0,1).copy()),torch.from_numpy(target.copy())
 for step in range(a.steps+1):
  if step%50==0 or step==a.steps:
   model.eval()
   with torch.inference_mode():prob=model((vx-mean)/std).softmax(1)[0].cpu().numpy()
   assert np.isfinite(prob).all();pred=(prob.argmax(0)+1).astype('uint8');met=metrics(pred,devmask);score=met['tissue_mean_iou'];entry={'step':step,'metrics':met,'elapsed_seconds':time.monotonic()-start};checks.append(entry)
   print(f"{a.name} update{step}/{a.steps} dev P/E/B recall "+'/'.join(f"{met['per_class'][n]['recall']:.3f}" for n in ['Periderm','Endoderm','Background'])+f' tissueIoU{score:.3f} time{time.monotonic()-start:.0f}s',flush=True)
   if score>best:
    best=score;beststep=step;torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'best_weights.pt');Image.fromarray(pred).save(out/'best_prediction.png');np.savez_compressed(out/'best_probabilities.npz',probabilities=prob);save(out/'best_result.json',entry)
   save(out/'checks.json',checks);save(out/'training_history.json',history)
  if step==a.steps:break
  inds=[int(rng.choice(new_mixed)) for _ in range(2)]+[int(rng.choice(old_mixed)),int(rng.choice(old_pools[1+step%2])),int(rng.choice(pools[4])),int(rng.choice(pools[3]))]
  items=[crop(i) for i in inds];x=torch.stack([i[0] for i in items]).to(device);t=torch.stack([i[1] for i in items]).to(device)
  model.train()
  for m in model.modules():
   if isinstance(m,torch.nn.BatchNorm2d) and not a.update_bn:m.eval()
  opt.zero_grad(set_to_none=True);logits=model((x-mean)/std);ce=F.cross_entropy(logits,t,ignore_index=255,reduction='none');valid=t!=255;pr=logits.softmax(1);ces=[];ds=[]
  field_losses=[];field_weights=[]
  for bi in range(t.shape[0]):
   ces=[];ds=[]
   for c in range(4):
    hit=t[bi]==c
    if hit.any().item():
     ces.append(ce[bi][hit].mean());pc=pr[bi,c]*valid[bi];ds.append(1-(2*(pc*hit).sum()+1)/(pc.sum()+hit.sum()+1))
   if ces:
    wt=float(manifest['patches'][inds[bi]].get('loss_weight',1.));field_weights.append(wt);field_losses.append(wt*(torch.stack(ces).mean()+.5*torch.stack(ds).mean()))
  loss=torch.stack(field_losses).sum()/sum(field_weights);assert torch.isfinite(loss).item();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();sched.step();history.append({'update':step+1,'loss':float(loss.item()),'fields':[manifest['patches'][i]['id'] for i in inds]})
  if (step+1)%25==0:print(f'Training update{step+1}/{a.steps} loss{loss.item():.4f}',flush=True)
 torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'final_weights.pt')
 save(out/'result.json',{**plan,'completed_steps':a.steps,'best_step':beststep,'best_tissue_mean_iou':best,'model_sha256':sha(out/'best_weights.pt'),'elapsed_seconds':time.monotonic()-start});print('FINISHED '+str(out),flush=True)
if __name__=='__main__':main()
