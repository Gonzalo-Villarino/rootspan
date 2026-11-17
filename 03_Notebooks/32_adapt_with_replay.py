"""Bounded local adaptation with archived training examples replayed.

Assistant labels from two user-identified 20_29 regions are TRAINING data.
MT25 root4 remains a reused development check, not an independent test.
Fixed 400 updates; no whole-slide or test-reference input to training.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,shutil,time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'04_Results/35_Multiple_Slides_Data'
NEW=ROOT/'02_Data/05_Training_and_Evaluation/03_Assistant_Prepared_Adaptation'
INITIAL=ROOT/'04_Results/39_PAT_Transfer_Learning/run_20261115T140159Z/model_weights.pt'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=400);parser.add_argument('--retain-old-predictions',action='store_true');args=parser.parse_args();assert 1<=args.steps<=400
 spec=importlib.util.spec_from_file_location('helpers',ROOT/'03_Notebooks/20_train_multiple_roots.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 rng=np.random.default_rng(1511);torch.manual_seed(1511);device='mps' if torch.backends.mps.is_available() else 'cpu'
 old=json.loads((OLD/'manifest.json').read_text());new=json.loads((NEW/'manifest.json').read_text());before={str(p):sha(p) for p in [INITIAL,OLD/'manifest.json',NEW/'manifest.json']}
 samples=[]
 for p in old['patches']:
  assert sha(OLD/p['image'])==p['image_sha256'] and sha(OLD/p['mask'])==p['mask_sha256'];samples.append((np.array(Image.open(OLD/p['image'])),np.array(Image.open(OLD/p['mask']))))
 fresh=[]
 for p in new['samples']:
  assert sha(NEW/p['image'])==p['source_sha256'] and sha(NEW/p['mask'])==p['mask_sha256'];fresh.append((np.array(Image.open(NEW/p['image'])),np.array(Image.open(NEW/p['mask']))))
 mixed=[i for i,(_,m) in enumerate(samples) if (m==1).sum()>=512 and (m==2).sum()>=512]
 bg=[i for i,(_,m) in enumerate(samples) if (m==3).sum()>512]
 assert mixed and bg
 model=smp.UnetPlusPlus(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=3,activation=None)
 model.load_state_dict(torch.load(INITIAL,map_location='cpu',weights_only=True),strict=True);model.to(device)
 for p in model.encoder.parameters():p.requires_grad_(False)
 for layer in [model.encoder.layer3,model.encoder.layer4]:
  for p in layer.parameters():p.requires_grad_(True)
 teacher=None
 if args.retain_old_predictions:
  for p in model.encoder.parameters():p.requires_grad_(False)
  teacher=smp.UnetPlusPlus(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=3,activation=None)
  teacher.load_state_dict(torch.load(INITIAL,map_location='cpu',weights_only=True),strict=True);teacher.to(device).eval().requires_grad_(False)
  optimizer=torch.optim.Adam([{'params':model.decoder.parameters(),'lr':1e-5},{'params':model.segmentation_head.parameters(),'lr':1e-5}])
 else:optimizer=torch.optim.Adam([{'params':model.decoder.parameters(),'lr':3e-5},{'params':model.segmentation_head.parameters(),'lr':3e-5},{'params':list(model.encoder.layer3.parameters())+list(model.encoder.layer4.parameters()),'lr':3e-6}])
 sched=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.steps,eta_min=5e-7)
 mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
 def tensor(rgb):return torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None].to(device)/255
 def predict(rgb):
  model.eval()
  with torch.inference_mode():p=model((tensor(rgb)-mean)/std).softmax(1)[0].cpu().numpy()
  return p
 pair=json.loads(helper.PAIR.read_text());devrgb=np.array(Image.open(helper.PAIR.parent/pair['image_file']));devmask=np.array(Image.open(helper.PAIR.parent/pair['mask_file']))
 parent='46_Adaptation_With_Retention' if args.retain_old_predictions else '45_Adaptation_With_Replay'
 out=ROOT/'04_Results'/parent/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(parents=True);shutil.copy2(__file__,out/Path(__file__).name)
 plan={'initial_model_sha256':sha(INITIAL),'old_manifest_sha256':sha(OLD/'manifest.json'),'new_manifest_sha256':sha(NEW/'manifest.json'),'seed':1511,'updates':args.steps,'training_slides':4,'new_data_role':'Assistant-prepared sparse TRAINING labels based on user-confirmed tissue identities; not independent ground truth.','batch':'6 fields: 2 from clearer new crop (one centred on each tissue), 1 from confirmed Periderm crop, 2 archived mixed-tissue fields, 1 archived explicit Background field. 512 native pixels, random position, paired rotation/flip, gain0.8-1.2.','loss':'Equal average of class-balanced CE + 0.5 Dice on new and archived halves; ignore codes0/255.','trainable':'Decoder, segmentation head, encoder stages3/4; early encoder and all BN running statistics frozen.','learning_rates':{'decoder_head':3e-5,'encoder_stages3_4':3e-6},'selection':'Check fixed steps100,200,300,400. Eligible if minimum recall on present classes in both new sparse training masks >=0.95 and old development tissue meanIoU >=0.60. Choose eligible checkpoint with highest old development tissue meanIoU; if none, preserve initial model as default and report adaptation inconclusive.','development_role':'MT25 root4 reused for selection; not independent. Original PAT exposure unknown.','postprocessing':'None in adaptation or selection.','prediction_input':'Image only. Reference masks used for loss/metrics separately.'}
 if teacher:
  plan.update({'method':'Preserve old training-image predictions with frozen teacher; update decoder/head only.','loss':'New-label balanced CE+0.5Dice +0.25 archived-label balanced CE/Dice +2.0 class-balanced KL to frozen initial model on archived training fields only. No development-reference input to gradients.','trainable':'Decoder and segmentation head only; whole encoder and all BN running statistics frozen.','learning_rates':{'decoder_head':1e-5,'encoder':0}})
 write(out/'experiment_plan.json',plan);print('Output: '+str(out),flush=True);history=[];checks=[];start=time.monotonic()
 def crop_new(index,code):
  im,ma=fresh[index];ys,xs=np.where(ma==code);pick=int(rng.integers(len(xs)));x=int(np.clip(xs[pick]-rng.integers(100,412),0,512));y=int(np.clip(ys[pick]-rng.integers(100,412),0,512));return im[y:y+512,x:x+512],ma[y:y+512,x:x+512]
 def loss_fn(logits,t):
  ce=F.cross_entropy(logits,t,ignore_index=255,reduction='none');valid=t!=255;pr=logits.softmax(1);cs=[];ds=[]
  for c in range(3):
   hit=t==c
   if hit.any().item():cs.append(ce[hit].mean());pc=pr[:,c]*valid;ds.append(1-(2*(pc*hit).sum()+1)/(pc.sum()+hit.sum()+1))
  return torch.stack(cs).mean()+.5*torch.stack(ds).mean()
 for step in range(args.steps+1):
  if step==0 or step%100==0 or step==args.steps:
   dmet=helper.metrics((predict(devrgb).argmax(0)+1).astype('uint8'),devmask);fits=[]
   for im,ma in fresh:
    met=helper.metrics((predict(im).argmax(0)+1).astype('uint8'),ma);met['scope']='Only assistant-prepared sparse TRAINING labels';fits.append(met)
   rec=min(c['recall'] for fit in fits for c in fit['per_class'].values() if c['reference_pixels'])
   check={'step':step,'old_development':dmet,'new_training_fit':fits,'minimum_new_training_recall':rec,'eligible':step>0 and rec>=.95 and dmet['tissue_mean_iou']>=.60}
   checks.append(check);write(out/'checks.json',checks)
   print(f"Update {step}: new training minimum recall {rec:.3f}; old development tissue IoU {dmet['tissue_mean_iou']:.3f}; {time.monotonic()-start:.0f}s",flush=True)
   if step>0:torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/f'checkpoint_{step}.pt')
  if step==args.steps:break
  pairs=[crop_new(0,2),crop_new(0,1),crop_new(1,1),samples[int(rng.choice(mixed))],samples[int(rng.choice(mixed))],samples[int(rng.choice(bg))]]
  images=[];targets=[]
  for im,ma in pairs:
   rot=int(rng.integers(4));im=np.rot90(im,rot).astype('float32')/255;ma=np.rot90(ma,rot)
   if rng.random()<.5:im=im[:,::-1];ma=ma[:,::-1]
   im=np.clip(im*rng.uniform(.8,1.2),0,1);tar=np.full(ma.shape,255,dtype='int64');v=np.isin(ma,[1,2,3]);tar[v]=ma[v]-1
   images.append(torch.from_numpy(im.transpose(2,0,1).copy()));targets.append(torch.from_numpy(tar.copy()))
  x=torch.stack(images).to(device);t=torch.stack(targets).to(device);model.train()
  for m in model.modules():
   if isinstance(m,torch.nn.BatchNorm2d):m.eval()
  optimizer.zero_grad(set_to_none=True);logits=model((x-mean)/std)
  if teacher:
   with torch.no_grad():tp=teacher((x[3:]-mean)/std).softmax(1)
   kl=F.kl_div(logits[3:].log_softmax(1),tp,reduction='none').sum(1);tc=tp.argmax(1);pieces=[kl[tc==c].mean() for c in range(3) if (tc==c).any().item()]
   loss=loss_fn(logits[:3],t[:3])+.25*loss_fn(logits[3:],t[3:])+2*torch.stack(pieces).mean()
  else:loss=(loss_fn(logits[:3],t[:3])+loss_fn(logits[3:],t[3:]))/2
  assert torch.isfinite(loss).item();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();sched.step();history.append({'update':step+1,'loss':float(loss.item())})
  if (step+1)%50==0:print(f"Training {step+1}/{args.steps}, loss {loss.item():.5f}",flush=True);write(out/'training_history.json',history)
 eligible=[c for c in checks if c['eligible']]
 selected=max(eligible,key=lambda c:c['old_development']['tissue_mean_iou']) if eligible else None
 if selected:shutil.copy2(out/f"checkpoint_{selected['step']}.pt",out/'model_weights.pt')
 for p,h in before.items():assert sha(p)==h
 result={**plan,'elapsed_seconds':time.monotonic()-start,'selected_step':selected['step'] if selected else None,'selected_check':selected,'selected_weights_sha256':sha(out/'model_weights.pt') if selected else None,'original_files_unchanged':True}
 write(out/'result.json',result);write(out/'training_history.json',history);print('Selected step: '+str(result['selected_step']),flush=True)
if __name__=='__main__':main()
