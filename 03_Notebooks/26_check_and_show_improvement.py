"""Verify the saved transfer model, compare raw vs explicit continuity cleanup.

Reference mask is read ONLY after image-only inference and cleanup. This is
development validation with unknown original-PAT image exposure, not a test.
"""
import argparse,hashlib,importlib.util,json,shutil
from pathlib import Path
import numpy as np
import torch
import segmentation_models_pytorch as smp
from scipy import ndimage as ndi
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'04_Results/40_Current_Improved_Result'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def continuity(probabilities):
 """Single Endoderm-left/Periderm-right transition, using model scores only.

 This orientation/one-transition assumption is specific to these prepared
 primary-root images. Allow all-one-class. Not Ling's exact PAT algorithm.
 Keep Background and small disconnected predictions unchanged.
 """
 raw=(probabilities.argmax(0)+1).astype('uint8');tissue=raw!=3
 lab,n=ndi.label(ndi.binary_closing(tissue,iterations=4));counts=np.bincount(lab.ravel());counts[0]=0
 if not counts.max():return raw,None
 active=ndi.binary_dilation(lab==counts.argmax(),iterations=5)&tissue
 denominator=(probabilities[0]+probabilities[1]).clip(1e-8)
 ep=(np.log((probabilities[1]/denominator).clip(1e-6,1))*active).sum(0)
 pp=(np.log((probabilities[0]/denominator).clip(1e-6,1))*active).sum(0)
 score=np.r_[0,np.cumsum(ep)]+np.r_[np.cumsum(pp[::-1])[::-1],0]
 boundary=int(score.argmax());pred=raw.copy();columns=np.arange(raw.shape[1])[None,:];pred[active&(columns<boundary)]=2;pred[active&(columns>=boundary)]=1
 assert np.array_equal(pred==3,raw==3)
 return pred,boundary
def main():
 global OUT
 parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path);args=parser.parse_args()
 if args.output:OUT=args.output.resolve()
 run=args.run.resolve();record=json.loads((run/'result.json').read_text());weights=run/'model_weights.pt'
 assert record['transition_focused_sampling'] and sha(weights)==record['model_weights_sha256']
 helperfile=Path(__file__).with_name('20_train_multiple_roots.py');spec=importlib.util.spec_from_file_location('helpers',helperfile);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 pair=json.loads(helper.PAIR.read_text());ip=helper.PAIR.parent/pair['image_file'];mp=helper.PAIR.parent/pair['mask_file'];assert sha(ip)==pair['image_sha256'];rgb=np.array(Image.open(ip))
 model=smp.UnetPlusPlus(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=3,activation=None);model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False).to('mps');x=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None].to('mps')/255
 mean=torch.tensor([.485,.456,.406],device='mps')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='mps')[None,:,None,None]
 with torch.inference_mode():scores=model((x-mean)/std).softmax(1)[0].cpu().numpy()
 assert np.isfinite(scores).all();raw=(scores.argmax(0)+1).astype('uint8');clean,boundary=continuity(scores)
 assert np.array_equal(raw,np.array(Image.open(run/'predicted_mask.png')))
 # All model predictions and optional cleanup are finished before reading truth.
 assert sha(mp)==record['reference_sha256'];truth=np.array(Image.open(mp));old=np.array(Image.open(helper.OLD));before=helper.metrics(old,truth);rm=helper.metrics(raw,truth);cm=helper.metrics(clean,truth)
 use_clean=cm['tissue_mean_iou']>rm['tissue_mean_iou'] and all(cm['per_class'][c]['recall']>=rm['per_class'][c]['recall'] for c in ['Periderm','Endoderm'])
 selected=clean if use_clean else raw;sm=cm if use_clean else rm
 OUT.mkdir(exist_ok=False);(OUT/'programs').mkdir();shutil.copy2(__file__,OUT/'programs'/Path(__file__).name);shutil.copy2(helperfile,OUT/'programs'/helperfile.name)
 for name,p in [('raw_model_prediction',raw),('continuity_prediction',clean),('selected_prediction',selected)]:
  Image.fromarray(p).save(OUT/f'{name}.png');Image.fromarray(helper.PAL[p]).save(OUT/f'{name}_colours.png')
 np.save(OUT/'raw_model_probabilities.npy',scores)
 title='Improved model + continuity rule' if use_clean else 'Improved model: raw prediction'
 # Close-up comparison, same fixed window for all images.
 crop=(0,330,1024,700);width=500;gap=18;margin=22;canvas=Image.new('RGB',(4*width+3*gap+2*margin,535),'white');draw=ImageDraw.Draw(canvas);font=lambda n:ImageFont.load_default(size=n)
 draw.text((margin,18),'Root 4: original result and the improved version',font=font(32),fill='#17212b')
 draw.text((margin,65),'Same microscopy and unchanged reference. Transfer learning from PAT; development comparison only.',font=font(23),fill='#374151')
 panels=[Image.fromarray(rgb),Image.fromarray(helper.PAL[truth]),Image.fromarray(helper.PAL[old]),Image.fromarray(helper.PAL[selected])]
 for i,(p,name) in enumerate(zip(panels,['Microscopy image','Your reference labels','Before: one-patch model',title])):
  xx=margin+i*(width+gap);draw.text((xx,120),name,font=font(21),fill='#17212b');canvas.paste(p.crop(crop).resize((width,181),Image.Resampling.LANCZOS if i==0 else Image.Resampling.NEAREST),(xx,166))
 draw.text((margin,378),'Red = Periderm | Green = Endoderm | Yellow = Background | Grey = unlabelled (not scored)',font=font(24),fill='#374151')
 draw.text((margin,422),'Overlap score (IoU): '+ ' | '.join(f'{n}: {100*before["per_class"][n]["iou"]:.1f}% -> {100*sm["per_class"][n]["iou"]:.1f}%' for n in helper.LABELS),font=font(24),fill='#17212b')
 draw.text((margin,473),'No reference copied into predictions. Root 4 excluded from our training; original PAT exposure is unknown.',font=font(22),fill='#374151');canvas.save(OUT/'OPEN_ME_root4_before_after.png')
 helper.comparison(rgb,truth,raw,clean,cm,OUT/'raw_vs_continuity_full.png')
 # Replace generic titles so no reader mistakes postprocessing for training.
 fig=Image.open(OUT/'raw_vs_continuity_full.png');dd=ImageDraw.Draw(fig);dd.rectangle((0,0,fig.width,109),fill='white');dd.text((24,20),'Raw model prediction versus the separate continuity rule',font=font(30),fill='#17212b');dd.text((24,65),'The rule uses model scores and an Endoderm-left / Periderm-right assumption, never the reference.',font=font(20),fill='#374151')
 for i,txt in [(2,'Raw transfer-model prediction'),(3,'After continuity rule')]:
  xx=24+i*(430+16);dd.rectangle((xx,111,xx+430,144),fill='white');dd.text((xx,115),txt,font=font(20),fill='#17212b')
 dd.rectangle((0,837,fig.width,929),fill='white');dd.text((24,848),'After cleanup, overlap score (IoU): '+ ' | '.join(f'{n} {100*cm["per_class"][n]["iou"]:.1f}%' for n in helper.LABELS),font=font(21),fill='#17212b');dd.text((24,883),'Development comparison, not an independent test. The rule is an experiment, not the exact PAT postprocessor.',font=font(19),fill='#374151');fig.save(OUT/'raw_vs_continuity_full.png')
 result={'selected_run':str(run),'selected_checkpoint_update':record['best_step'],'completed_training_updates':record['completed_steps'],'before':before,'raw_model':rm,'continuity_rule':cm,'selected':sm,'selected_kind':'raw_model_plus_continuity' if use_clean else 'raw_model','transition_column':boundary,'transition_x_full_image':None if boundary is None else boundary+8928,'continuity_rule_used_reference':False,'postprocess_decision':'Use only if tissue mean IoU and both tissue recalls improve; otherwise retain raw prediction. This selection uses root4 validation, not an independent test.','reference_sha256':sha(mp),'image_sha256':sha(ip),'model_weights_sha256':sha(weights),'raw_prediction_sha256':sha(OUT/'raw_model_prediction.png'),'selected_prediction_sha256':sha(OUT/'selected_prediction.png'),'caveat':'PAT original training membership unknown; root4 used repeatedly for development selection. Needs an independent new slide. Unlabelled pixels not scored.'}
 helper.write_json(OUT/'comparison_results.json',result)
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
