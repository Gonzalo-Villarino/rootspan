"""Show raw image-only network output separately from the anatomical rule; score after inference."""
from pathlib import Path
import json,argparse
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from predict import PAL
from main_root_rule import interpret,met
from check_reference import metrics
ROOT=Path(__file__).resolve().parents[1];REF=ROOT/'04_Results/52_Reference_Check_20261116'
def full_metrics(pred,truth):
 valid=np.isin(truth,[1,2,3,4]);out={}
 for c,name in enumerate(['Periderm','Endoderm','Background','Lateral_root'],1):
  tp=int(((pred==c)&(truth==c)).sum());support=int((truth==c).sum());pp=int(((pred==c)&valid).sum());union=support+pp-tp
  out[name]={'reference_pixels':support,'correct_pixels':tp,'recall':tp/support if support else None,'precision':tp/pp if pp else None,'iou':tp/union if union else None,'reference_pixels_left_for_review':int(((truth==c)&(pred==255)).sum())}
 return {'per_class':out,'tissue_mean_iou':float(np.mean([out[k]['iou'] for k in ['Periderm','Endoderm']])),'scope':'Familiar development crop. Only labelled pixels scored. Review255 counts as missed, not removed from recall/IoU.'}
def gate(m):
 p=m['per_class'];return bool(all(p[k]['recall']>=.85 and p[k]['precision'] is not None and p[k]['precision']>=.85 for k in ['Periderm','Endoderm']) and p['Background']['recall']>=.98 and m['tissue_mean_iou']>=.75)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--folder',type=Path,required=True);args=ap.parse_args();folder=args.folder;out=folder/'directional_rule_review';out.mkdir(exist_ok=False)
 p=np.load(folder/'best_probabilities.npz')['probabilities'];raw=(p.argmax(0)+1).astype('uint8');assert np.array_equal(raw,np.array(Image.open(folder/'best_prediction.png')));rule,details=interpret(p)
 # Reference loaded AFTER prediction and rule. Neither predictor takes the reference as input.
 truth=np.array(Image.open(REF/'reference_mask.png'));rgb=np.array(Image.open(REF/'image.png'));rm=full_metrics(raw,truth);pm=full_metrics(rule,truth);rr=json.loads((folder/'best_result.json').read_text());assert np.isclose(rm['tissue_mean_iou'],rr['metrics']['tissue_mean_iou'])
 pmin=int(np.where(truth==1)[1].min());emax=int(np.where(truth==2)[1].max());b=details.get('boundary_x');gap=sorted([pmin,emax]);err=max(gap[0]-b,b-gap[1],0) if b is not None else None
 report={'raw_model_metrics':rm,'directional_rule_metrics':pm,'raw_passes_development_gate':gate(rm),'rule_passes_development_gate':gate(pm),'rule_details':details,'manual_transition_x_range_px':gap,'rule_boundary_horizontal_distance_from_manual_range_px':err,'gate':'Tissue recalls andprecisions>=.85;BGrecall>=.98;meantissueIoU>=.75. Reviewpixelscountasmissed. Passing this familiarcrop isnot independentvalidation.','network_source':str(folder),'checkpoint_step':rr['step'],'new_training_label_provenance':'Includes half-weight rule-derived Endoderm teaching examples onCOL0, not independently hand-drawn labels.','status':'Prototype check; no unattended measurement approval.'};(out/'review_results.json').write_text(json.dumps(report,indent=2)+'\n');Image.fromarray(rule).save(out/'rule_prediction.png');pal=PAL.copy();pal[255]=[150,150,160];Image.fromarray(pal[rule]).save(out/'rule_colours.png')
 box=(0,300,1024,640);w=820;h=round((box[3]-box[1])*w/1024);canvas=Image.new('RGB',(1690,2*(h+55)+155),'white');dr=ImageDraw.Draw(canvas);font=lambda n:ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',n)
 dr.text((20,12),'20_29: network prediction and the main-root direction rule',font=font(29),fill='black')
 panels=[('1 Microscopy image',rgb),('2 Your hand-drawn reference',pal[truth]),('3 U-Net family model: raw prediction',pal[raw]),('4 Prediction + one-transition rule',pal[rule])]
 for i,(name,a) in enumerate(panels):
  x=20+(i%2)*840;y=64+(i//2)*(h+55);dr.text((x,y),name,font=font(23),fill='black');canvas.paste(Image.fromarray(a).crop(box).resize((w,h),Image.Resampling.LANCZOS if i==0 else Image.Resampling.NEAREST),(x,y+33))
 dr.text((20,canvas.height-80),'Red Periderm | Green Endoderm | Yellow Background | Purple predicted lateral root',font=font(21),fill='black');dr.text((20,canvas.height-54),'Dark grey: unlabelled reference. Light grey: rule leaves tissue for review; it is not confirmed Background.',font=font(21),fill='black');dr.text((20,canvas.height-28),'Both predictions use image information only. This familiar development example is not a blind test.',font=font(20),fill='black');canvas.save(out/'RESULT_TO_REVIEW.png')
 print(json.dumps({'raw_tissue_iou':rm['tissue_mean_iou'],'rule_tissue_iou':pm['tissue_mean_iou'],'rule_boundary':b,'reference_gap':gap,'gap_distance_px':err,'raw_pass':gate(rm),'rule_pass':gate(pm),'rule_recalls':{k:v['recall'] for k,v in pm['per_class'].items()}},indent=2),flush=True)
if __name__=='__main__':main()
