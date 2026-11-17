"""Transfer PAT features to this three-category learning task.

Train on existing613 fields, never root4/6. Keep original PAT BatchNorm
statistics fixed. Initialise head from P/E/Background channels of published
four-class PAT. No masks enter inference. PAT's earlier image exposure is
unknown, so root4 is development validation, not an independent test.
"""
import argparse,copy,hashlib,importlib.util,json,shutil,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'04_Results/35_Multiple_Slides_Data';PARENT=ROOT/'04_Results/39_PAT_Transfer_Learning'
def load_module(p):
 s=importlib.util.spec_from_file_location('multiroot_helpers',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--steps',type=int,default=1500);a.add_argument('--focus-transitions',action='store_true');args=a.parse_args();assert 1<=args.steps<=3000
 helper=load_module(Path(__file__).with_name('20_train_multiple_roots.py'));manifest=json.loads((DATA/'manifest.json').read_text())
 for p,h in manifest['sources_sha256'].items():assert sha(p)==h
 rng=np.random.default_rng(34);torch.manual_seed(34);device='mps' if torch.backends.mps.is_available() else 'cpu'
 weights=ROOT/'00_Software/PAT_Reference/pat_state_dict.pt';state=torch.load(weights,map_location='cpu',weights_only=True)
 state['segmentation_head.0.weight']=state['segmentation_head.0.weight'][[1,2,0]].clone();state['segmentation_head.0.bias']=state['segmentation_head.0.bias'][[1,2,0]].clone()
 model=smp.UnetPlusPlus(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=3,activation=None);model.load_state_dict(state,strict=True);model.to(device)
 # Early encoder layers remain fixed; adapt higher features and the decoder.
 for p in model.encoder.parameters():p.requires_grad_(False)
 for layer in [model.encoder.layer3,model.encoder.layer4]:
  for p in layer.parameters():p.requires_grad_(True)
 optim=torch.optim.Adam([{'params':model.decoder.parameters(),'lr':.0001},{'params':model.segmentation_head.parameters(),'lr':.0001},{'params':list(model.encoder.layer3.parameters())+list(model.encoder.layer4.parameters()),'lr':.00001}])
 sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=args.steps,eta_min=.000001)
 samples=[]
 for p in manifest['patches']:
  assert sha(DATA/p['image'])==p['image_sha256'] and sha(DATA/p['mask'])==p['mask_sha256'];samples.append((np.array(Image.open(DATA/p['image'])),np.array(Image.open(DATA/p['mask']))))
 eligible={c:[i for i,(_,m) in enumerate(samples) if (m==c).sum()>512] for c in [1,2,3]}
 mixed=[i for i,(_,m) in enumerate(samples) if (m==1).sum()>=512 and (m==2).sum()>=512]
 if args.focus_transitions:assert mixed
 pair=json.loads(helper.PAIR.read_text());rgb=np.array(Image.open(helper.PAIR.parent/pair['image_file']));truth=np.array(Image.open(helper.PAIR.parent/pair['mask_file']));old=np.array(Image.open(helper.OLD));assert sha(helper.PAIR.parent/pair['mask_file'])==pair['mask_sha256']
 mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
 vx=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None].to(device)/255
 PARENT.mkdir(exist_ok=True);out=PARENT/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir();(out/'programs').mkdir();shutil.copy2(__file__,out/'programs'/Path(__file__).name);shutil.copy2(Path(__file__).with_name('20_train_multiple_roots.py'),out/'programs/20_train_multiple_roots.py')
 plan={'method':'Fine-tune published PAT UNet++/ResNet34, mapped head to Periderm/Endoderm/Background; freeze early encoder and all BatchNorm running stats; adapt decoder and encoder stages3/4.','data_manifest':str(DATA/'manifest.json'),'data_manifest_sha256':sha(DATA/'manifest.json'),'initial_weights_sha256':sha(weights),'training_roots':16,'training_slides':3,'current_training_excludes_roots':[4,6],'root4_role':'Development validation for checkpoint selection. PAT original training membership unknown.','steps_requested':args.steps,'seed':34,'batch_size':6,'training_field':'512x512 native pixels, paired flip/90degree rotations, image gain0.8-1.2; ImageNet RGB normalization. No resizing.','class_order':['Periderm','Endoderm','Background'],'mask_codes':[1,2,3],'ignore_codes':[0,255],'reference_sha256':pair['mask_sha256'],'postprocessing':'None','prediction_input':'RGB image only','baseline_metrics':helper.metrics(old,truth)}
 plan['transition_focused_sampling']=args.focus_transitions
 plan['mixed_tissue_fields_available']=len(mixed)
 plan['sampling']='Four mixed-tissue fields and two explicit-Background fields per update' if args.focus_transitions else 'Two fields centred on each of the three categories per update'
 helper.write_json(out/'experiment_plan.json',plan);best=-1;beststep=0;history=[];start=time.monotonic()
 for step in range(args.steps+1):
  if step==0 or step%100==0 or step==args.steps:
   model.eval()
   with torch.inference_mode():prediction=(model((vx-mean)/std).argmax(1)[0].cpu().numpy()+1).astype('uint8')
   met=helper.metrics(prediction,truth);score=met['tissue_mean_iou'];print(f'Update {step}/{args.steps}: root4 P/E/B '+ '/'.join(f'{100*met["per_class"][n]["recall"]:.1f}' for n in helper.LABELS)+f'; tissue IoU {score:.3f}; {time.monotonic()-start:.0f}s',flush=True)
   if score>best:
    best=score;beststep=step;torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'model_weights.pt');Image.fromarray(prediction).save(out/'predicted_mask.png');helper.write_json(out/'best_metrics.json',met)
   if history:history[-1]['validation_tissue_mean_iou']=score
   helper.write_json(out/'training_history.json',history)
  if step==args.steps:break
  images=[];targets=[]
  for j in range(6):
   c=(step+j)%3+1;pool=(mixed if j<4 else eligible[3]) if args.focus_transitions else eligible[c]
   im,ma=samples[int(rng.choice(pool))];im=im.astype('float32')/255;ma=ma.copy();rot=int(rng.integers(4));im=np.rot90(im,rot);ma=np.rot90(ma,rot)
   if rng.random()<.5:im=im[:,::-1];ma=ma[:,::-1]
   im=np.clip(im*rng.uniform(.8,1.2),0,1);tar=np.full(ma.shape,255,dtype='int64');v=np.isin(ma,[1,2,3]);tar[v]=ma[v]-1;images.append(torch.from_numpy(np.ascontiguousarray(im.transpose(2,0,1))));targets.append(torch.from_numpy(np.ascontiguousarray(tar)))
  x=torch.stack(images).to(device);t=torch.stack(targets).to(device);model.train()
  for m in model.modules():
   if isinstance(m,torch.nn.BatchNorm2d):m.eval()
  optim.zero_grad(set_to_none=True);logits=model((x-mean)/std);ce=F.cross_entropy(logits,t,ignore_index=255,reduction='none');valid=t!=255;prob=logits.softmax(1);ces=[];ds=[]
  for c in range(3):
   hit=t==c
   if hit.any().item():ces.append(ce[hit].mean());p=prob[:,c]*valid;ds.append(1-(2*(p*hit).sum()+1)/(p.sum()+hit.sum()+1))
  loss=torch.stack(ces).mean()+.5*torch.stack(ds).mean();assert torch.isfinite(loss).item();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optim.step();sched.step();history.append({'update':step+1,'loss':float(loss.item())})
 model.load_state_dict(torch.load(out/'model_weights.pt',map_location='cpu',weights_only=True));model.eval()
 with torch.inference_mode():pred=(model((vx-mean)/std).argmax(1)[0].cpu().numpy()+1).astype('uint8')
 assert np.array_equal(pred,np.array(Image.open(out/'predicted_mask.png')));met=helper.metrics(pred,truth);Image.fromarray(helper.PAL[pred]).save(out/'predicted_mask_preview.png');helper.comparison(rgb,truth,old,pred,met,out/'root4_before_after.png')
 # Correct the generic comparison's caption: prior PAT exposure is unknown.
 from PIL import ImageDraw,ImageFont
 fig=Image.open(out/'root4_before_after.png');dr=ImageDraw.Draw(fig);dr.rectangle((0,58,fig.width,104),fill='white');dr.text((24,65),'PAT transfer learning. Root 4 excluded from this training; original PAT image exposure is unknown.',font=ImageFont.load_default(size=21),fill='#374151');dr.rectangle((0,877,fig.width,929),fill='white');dr.text((24,883),'Development comparison only. Predictions use the image alone; no cleanup or copied reference labels.',font=ImageFont.load_default(size=20),fill='#374151');fig.save(out/'root4_before_after.png')
 for p,h in manifest['sources_sha256'].items():assert sha(p)==h
 result={**plan,'best_step':beststep,'completed_steps':args.steps,'elapsed_seconds':time.monotonic()-start,'validation_metrics':met,'model_weights_sha256':sha(out/'model_weights.pt'),'prediction_sha256':sha(out/'predicted_mask.png'),'source_files_unchanged':True};helper.write_json(out/'result.json',result);print('SAVED: '+str(out),flush=True)
if __name__=='__main__':main()
