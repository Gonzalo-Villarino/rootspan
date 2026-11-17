"""Save all planned comparisons; keep training fit apart from development and new-slide views."""
from pathlib import Path
import argparse,importlib.util,json,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
ROOT=Path(__file__).resolve().parents[1]
def module(name,file):
 s=importlib.util.spec_from_file_location(name,ROOT/'03_Notebooks'/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
api=module('predict_image','33_predict_image.py');cleanup=module('old_cleanup','26_check_and_show_improvement.py');helper=module('metrics','20_train_multiple_roots.py')
def font(n):return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',n)
def comparison(rgb,old,new,clean,path,title,scope,box=(0,250,1024,730)):
 w=512;h=240;out=Image.new('RGB',(4*(w+20)+20,h+230),'white');d=ImageDraw.Draw(out)
 d.text((20,12),title,font=font(31),fill='black');d.text((20,55),scope,font=font(24),fill='#374151')
 for j,(name,arr) in enumerate([('Microscopy',rgb),('Earlier model: raw',api.PAL[old]),('Adapted model: raw',api.PAL[new]),('Adapted + continuity rule',api.PAL[clean])]):
  x=20+j*(w+20);d.text((x,105),name,font=font(25),fill='black');im=Image.fromarray(arr).crop(box).resize((w,h),Image.Resampling.LANCZOS if j==0 else Image.Resampling.NEAREST);out.paste(im,(x,145))
 d.text((20,h+162),'Red = Periderm | Green = Endoderm | Yellow = Background. Right panel includes an explicit rule; it is not raw U-Net output.',font=font(24),fill='black')
 d.text((20,h+197),'Continuity assumes one left-Endoderm/right-Periderm primary root. Branches and uncertain transitions remain limitations.',font=font(24),fill='black');out.save(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args();result=json.loads((a.run/'result.json').read_text());checks=json.loads((a.run/'checks.json').read_text())
 eligible=result['selected_step'] is not None
 # Failed eligibility does not silently become a recommended replacement.
 chosen=next(c for c in checks if c['step']==result['selected_step']) if eligible else max([c for c in checks if c['step']>0],key=lambda c:c['old_development']['tissue_mean_iou'])
 weights=a.run/f"checkpoint_{chosen['step']}.pt";baseline=ROOT/'04_Results/39_PAT_Transfer_Learning/run_20261115T140159Z/model_weights.pt'
 out=ROOT/'04_Results/47_Final_Model_Comparison';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
 adaptation=ROOT/'02_Data/05_Training_and_Evaluation/03_Assistant_Prepared_Adaptation'
 cases={
 '20_29_clearer':{'image':adaptation/'20_29_clearer/image.png','mask':adaptation/'20_29_clearer/mask.png','role':'Sparse assistant labels USED FOR TRAINING; this is training fit.'},
 '20_29_periderm':{'image':adaptation/'20_29_periderm/image.png','mask':adaptation/'20_29_periderm/mask.png','role':'Sparse assistant labels USED FOR TRAINING; this is training fit.'},
 '20_29_other_root':{'image':ROOT/'04_Results/42_20_29_Transition_Prediction/view_20261115T153948Z_bmduly9w/20_29_x4864_y6144_image.png','mask':None,'role':'Not used in adaptation; same slide, poor-quality region, no pixel reference.'},
 'MT25_root4':{'image':ROOT/'04_Results/31_Root4_Background_Check_Patch/MT25_rep3_root4_x8928_y5264_image.png','mask':ROOT/'04_Results/31_Root4_Background_Check_Patch/MT25_rep3_root4_x8928_y5264_mask.png','role':'Reused DEVELOPMENT check; not independent test accuracy.'}}
 record={'candidate_weights':str(weights.resolve()),'candidate_sha256':api.sha(weights),'baseline_weights':str(baseline),'baseline_sha256':api.sha(baseline),'selected_step':chosen['step'],'passed_predeclared_replacement_criteria':eligible,'default_model':'adapted' if eligible else 'original retained; adapted candidate is exploratory only','cases':{},'cleanup':'Unchanged function from program26, applied equally before/after. It assumes one horizontally oriented Endoderm-left/Periderm-right primary root. No labels read by inference/cleanup.'}
 buffers={}
 for name,case in cases.items():
  folder=out/name;folder.mkdir();rgb=np.array(Image.open(case['image']));buffers[name]=rgb;Image.fromarray(rgb).save(folder/'image.png');record['cases'][name]={'image':str(case['image']),'image_sha256':api.sha(case['image']),'role':case['role']}
 # Two separate fixed model passes, then score references. No label data enters inference.
 for version,weight in [('before',baseline),('after',weights)]:
  model,device=api.load_model(weight)
  for name,rgb in buffers.items():
   prob=api.predict(model,device,rgb);pred=(prob.argmax(0)+1).astype('uint8');clean,bound=cleanup.continuity(prob);folder=out/name
   np.savez_compressed(folder/f'{version}_probabilities.npz',probabilities=prob);Image.fromarray(pred).save(folder/f'{version}_raw.png');Image.fromarray(api.PAL[pred]).save(folder/f'{version}_raw_colours.png');Image.fromarray(clean).save(folder/f'{version}_continuity.png');Image.fromarray(api.PAL[clean]).save(folder/f'{version}_continuity_colours.png');record['cases'][name][version]={'continuity_boundary_x_in_crop':int(bound)}
  del model
  if device=='mps':torch.mps.empty_cache()
 for name,case in cases.items():
  folder=out/name;rgb=buffers[name]
  old=np.array(Image.open(folder/'before_raw.png'));new=np.array(Image.open(folder/'after_raw.png'));clean=np.array(Image.open(folder/'after_continuity.png'))
  comparison(rgb,old,new,clean,folder/'OPEN_ME_comparison.png',name+': before and after adaptation',case['role'])
  api.figure(rgb,new,folder/'FULL_raw_prediction.png');api.figure(rgb,clean,folder/'FULL_continuity_prediction.png')
  # Correct title and explanatory footer of the processed view.
  fig=Image.open(folder/'FULL_continuity_prediction.png');d=ImageDraw.Draw(fig);d.rectangle((740,0,fig.width,56),fill='white');d.text((740,18),'Model + continuity rule',font=font(25),fill='black');d.rectangle((0,fig.height-42,fig.width,fig.height),fill='white');d.text((20,fig.height-35),'Processed view; assumes one left-Endoderm/right-Periderm primary root. Lateral branches are unreliable.',font=font(17),fill='black');fig.save(folder/'FULL_continuity_prediction.png')
  if case['mask']:
   truth=np.array(Image.open(case['mask']));record['cases'][name]['reference_sha256']=api.sha(case['mask'])
   for version in ['before','after']:
    for method in ['raw','continuity']:
     met=helper.metrics(np.array(Image.open(folder/f'{version}_{method}.png')),truth);met['scope']=case['role'];record['cases'][name][version][method]=met
 # All six source-selected 2_12 fields, AFTER selection. Do not tune against these.
 plan=json.loads((ROOT/'05_Notes/Reviews/20261115_Model_Repair/untuned_check_plan.json').read_text());source=ROOT/plan['source'];assert api.sha(source)==plan['source_sha256'];Image.MAX_IMAGE_PIXELS=200_000_000
 with Image.open(source) as im:fields=[np.array(im.crop(b)) for b in plan['boxes']]
 other=out/'2_12_not_used_in_fine_tuning';other.mkdir();preds={}
 for version,weight in [('before',baseline),('after',weights)]:
  model,device=api.load_model(weight);preds[version]=[]
  for i,rgb in enumerate(fields):
   pr=api.predict(model,device,rgb);pred=(pr.argmax(0)+1).astype('uint8');preds[version].append(pred);Image.fromarray(pred).save(other/f'root_{i+1}_{version}_raw.png')
   if version=='before':Image.fromarray(rgb).save(other/f'root_{i+1}_image.png')
  del model
  if device=='mps':torch.mps.empty_cache()
 w=512;h=230;sheet=Image.new('RGB',(1616,6*(h+50)+130),'white');d=ImageDraw.Draw(sheet);d.text((20,15),'2_12: six fields not used in this fine-tuning | no reference labels, no accuracy score',font=font(25),fill='black')
 for i,rgb in enumerate(fields):
  for j,(name,arr) in enumerate([('Microscopy',rgb),('Earlier raw',api.PAL[preds['before'][i]]),('Adapted raw',api.PAL[preds['after'][i]])]):
   x=20+j*532;y=65+i*(h+50);d.text((x,y),f'Field {i+1}: {name}',font=font(22),fill='black');sheet.paste(Image.fromarray(arr).crop((0,282,1024,742)).resize((w,h),Image.Resampling.LANCZOS if j==0 else Image.Resampling.NEAREST),(x,y+30))
 d.text((20,sheet.height-35),'All six planned views retained. Original PAT training exposure is unknown; these pictures alone cannot establish accuracy.',font=font(22),fill='black');sheet.save(other/'ALL_SIX_FIELDS.png')
 record['2_12_snapshot']={**plan,'checkpoint_selected_before_predictions':True,'used_for_further_tuning':False};(out/'comparison_details.json').write_text(json.dumps(record,indent=2)+'\n')
 shutil.copy2(out/'20_29_clearer/OPEN_ME_comparison.png',out/'OPEN_ME_NEW_RESULT.png')
 print('Saved '+str(out));print('Candidate passed replacement checks:',eligible);print('Step:',chosen['step'])
if __name__=='__main__':main()
