"""Learn from several roots instead of memorising one patch.

Learning experiment, 2026-11-15. Uses Gonzalo's existing polygons only.
Train roots 1,2,3,5; root4 is development validation; root6 remains reserved.
This is NOT a reproduction of PAT or an independent new-slide test.
Raw image-only predictions are saved before any optional post-processing.
All source annotations, images, masks and earlier models remain unchanged.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '02_Data/02_My_First_Annotations/MT25_rep3_practice.png'
ANN = ROOT / '02_Data/02_My_First_Annotations/MT25_rep3_root5_refined_GV.json'
ANN1 = ROOT / '02_Data/03_Background_Annotations/MT25_rep3_background_refined_GV.json'
DATA = ROOT / '04_Results/33_Multiple_Roots_Data'
RUNS = ROOT / '04_Results/34_Multiple_Roots_Training'
PAIR = ROOT / '04_Results/31_Root4_Background_Check_Patch/patch_details.json'
OLD = ROOT / '04_Results/32_Root4_Prediction/view_20261115T132047Z_70i13lvb/predicted_mask.png'
LABELS = ['Periderm', 'Endoderm', 'Background']
PAL = np.zeros((256,3), dtype=np.uint8)
PAL[0], PAL[1], PAL[2], PAL[3], PAL[255] = (48,52,58),(218,54,65),(30,170,91),(255,212,59),(209,77,199)

def module(name, path):
    s=importlib.util.spec_from_file_location(name, path); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

def overlap(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))

def bounds(shapes):
    pts=np.array([p for s in shapes for p in s['points']]);return [math.floor(pts[:,0].min()),math.floor(pts[:,1].min()),math.ceil(pts[:,0].max())+1,math.ceil(pts[:,1].max())+1]

def prepare():
    if DATA.exists():
        manifest=json.loads((DATA/'manifest.json').read_text())
        for p,h in manifest['sources_sha256'].items():assert sha(p)==h, f'Source changed: {p}'
        for p in manifest['patches']:
            assert sha(DATA/p['image'])==p['image_sha256'] and sha(DATA/p['mask'])==p['mask_sha256']
        return manifest
    sources={str(p):sha(p) for p in [SOURCE,ANN,ANN1,PAIR,OLD]}
    d=json.loads(ANN.read_text()); a=json.loads(ANN1.read_text())
    assert (d['imageWidth'],d['imageHeight'])==(15423,9543)
    groups={1:a['shapes'][:2],2:d['shapes'][2:4],3:d['shapes'][4:6],5:d['shapes'][8:10]}
    for ss in groups.values():assert {s['label'] for s in ss}=={'Periderm','Endoderm'}
    bg=d['shapes'][10:14]+[a['shapes'][i] for i in [12,13,14,16,17,18,19,20]]
    assert all(s['label']=='Background' for s in bg)
    forbidden={'root4_entire_bbox':bounds(d['shapes'][14:16]),'root6_entire_bbox':bounds(d['shapes'][6:8]),'root4_validation_crop':[8928,5264,9952,6288]}
    selected={k:v for k,v in d.items() if k!='imageData'}
    selected['shapes']=[s for ss in groups.values() for s in ss]+bg
    converter=module('mask_converter',Path(__file__).with_name('07_make_background_mask.py'))
    mask=converter.make_mask(selected)
    Image.MAX_IMAGE_PIXELS=200_000_000
    image=Image.open(SOURCE).convert('RGB')
    DATA.mkdir();(DATA/'images').mkdir();(DATA/'masks').mkdir();(DATA/'programs').mkdir()
    write_json(DATA/'selected_annotations_without_image_data.json',selected)
    candidates=[]
    # Fixed 512px fields following each labelled root; no image resizing in files.
    for root,ss in groups.items():
        region=Image.new('1',image.size);draw=ImageDraw.Draw(region)
        for s in ss:draw.polygon([tuple(p) for p in s['points']],fill=1)
        region=np.asarray(region)
        b=bounds(ss)
        for x in range(max(0,b[0]-64),min(image.width-512,b[2]-256)+1,320):
            local=np.where(region[:,x+192:x+320])[0]
            if not len(local):continue
            y=int(np.clip(np.median(local)-256,0,image.height-512))
            candidates.append((root,[x,y,x+512,y+512]))
        del region
    for s in bg:
        b=bounds([s]);x=int(np.clip((b[0]+b[2])/2-256,0,image.width-512));y=int(np.clip((b[1]+b[3])/2-256,0,image.height-512))
        candidates.append((0,[x,y,x+512,y+512]))
    patches=[];seen=set();skipped=0
    for root,b in candidates:
        if tuple(b) in seen:continue
        seen.add(tuple(b))
        if any(overlap(b,f) for f in forbidden.values()):skipped+=1;continue
        m=mask[b[1]:b[3],b[0]:b[2]].copy()
        counts={str(c):int((m==c).sum()) for c in [0,1,2,3,255]}
        if sum(counts[str(c)] for c in [1,2,3])<512:continue
        name=f'root{root}_x{b[0]}_y{b[1]}'
        ip=DATA/'images'/f'{name}.png';mp=DATA/'masks'/f'{name}.png'
        image.crop(b).save(ip);Image.fromarray(m).save(mp)
        patches.append({'id':name,'root':root,'box':b,'image':str(ip.relative_to(DATA)),'mask':str(mp.relative_to(DATA)),'image_sha256':sha(ip),'mask_sha256':sha(mp),'counts':counts})
    assert patches and {p['root'] for p in patches} >= {1,2,3,5}
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'sources_sha256':sources,'training_roots':[1,2,3,5],'development_validation_root':4,'reserved_root':6,'forbidden_boxes':forbidden,'skipped_for_overlap':skipped,'patch_size':[512,512],'patches':patches,'label_codes':dict(zip(LABELS,[1,2,3])),'ignored_codes':[0,255],'selection':'Existing user tissue polygons: root1 from its refined JSON; roots2/3/5 from latest working JSON. Explicit Background polygons only. No new annotations or automatic background. Overlapping training crops are not independent specimens.'}
    for p,h in sources.items():assert sha(p)==h
    write_json(DATA/'manifest.json',manifest)
    shutil.copy2(__file__,DATA/'programs'/Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('07_make_background_mask.py'),DATA/'programs/07_make_background_mask.py')
    # Small provenance contact sheet, not training data.
    chosen=[next(p for p in patches if p['root']==r and int(p['counts']['1'])>1000) for r in [1,2,3,5]]
    sheet=Image.new('RGB',(1050,590),'white');dr=ImageDraw.Draw(sheet)
    for i,p in enumerate(chosen):
        x=15+260*i;dr.text((x,8),f"Training root {p['root']}",fill='black',font=ImageFont.load_default(size=21))
        sheet.paste(Image.open(DATA/p['image']).resize((250,250)),(x,43))
        mm=np.asarray(Image.open(DATA/p['mask']));sheet.paste(Image.fromarray(PAL[mm]).resize((250,250),Image.Resampling.NEAREST),(x,310))
    sheet.save(DATA/'training_examples.png')
    print(f'Prepared {len(patches)} fields from four training roots; excluded {skipped} candidate fields touching reserved roots.',flush=True)
    return manifest

def metrics(pred,truth):
    valid=np.isin(truth,[1,2,3]);cm=np.bincount(3*(truth[valid].astype(int)-1)+pred[valid]-1,minlength=9).reshape(3,3)
    rows=cm.sum(1);cols=cm.sum(0);tp=np.diag(cm);iou=tp/np.maximum(rows+cols-tp,1)
    return {'confusion_rows_reference_columns_prediction':cm.tolist(),'per_class':{n:{'correct':int(tp[i]),'reference_pixels':int(rows[i]),'recall':float(tp[i]/max(rows[i],1)),'precision':float(tp[i]/max(cols[i],1)),'iou':float(iou[i]),'f1':float(2*tp[i]/max(rows[i]+cols[i],1))} for i,n in enumerate(LABELS)},'mean_iou':float(iou.mean()),'tissue_mean_iou':float(iou[:2].mean()),'unlabelled_pixels_ignored':int((~valid).sum()),'scope':'Only labelled pixels. Root4 same-slide development validation; not an independent test.'}

def tensor_image(a):return torch.from_numpy(np.ascontiguousarray(a)).permute(2,0,1).float()/255

@torch.inference_mode()
def predict(model,rgb,device):
    model.eval();x=tensor_image(rgb)[None].to(device)
    scores=model(x);assert torch.isfinite(scores).all().item()
    return (scores.argmax(1)[0].cpu().numpy()+1).astype('uint8')

def comparison(rgb,truth,old,new,stats,path):
    # Fixed display box from the established root4 comparison; no label editing.
    panels=[Image.fromarray(rgb),Image.fromarray(PAL[truth]),Image.fromarray(PAL[old]),Image.fromarray(PAL[new])]
    names=['Microscopy image','Your reference labels','Before: one training patch','After: multiple training roots']
    width=430;gap=16;margin=24;canvas=Image.new('RGB',(4*width+3*gap+2*margin,930),'white');d=ImageDraw.Draw(canvas)
    font=lambda n:ImageFont.load_default(size=n)
    d.text((margin,20),'Root 4: did learning from more examples help?',fill='#17212b',font=font(31))
    d.text((margin,65),'Root 4 excluded from training. Raw model predictions at every pixel; no cleanup or copied labels.',fill='#374151',font=font(21))
    crop=(150,360,950,680)
    for i,(name,panel) in enumerate(zip(names,panels)):
        x=margin+i*(width+gap);d.text((x,115),name,fill='#17212b',font=font(20));canvas.paste(panel.resize((width,width),Image.Resampling.NEAREST if i else Image.Resampling.LANCZOS),(x,150));canvas.paste(panel.crop(crop).resize((width,172),Image.Resampling.NEAREST if i else Image.Resampling.LANCZOS),(x,615))
    d.text((margin,586),'Same enlarged root region in each panel:',fill='#374151',font=font(20))
    d.text((margin,812),'Red = Periderm | Green = Endoderm | Yellow = Background | Grey = unlabelled (not scored)',fill='#374151',font=font(21))
    d.text((margin,848),'After, matching your labelled pixels: '+ ' | '.join(f'{n} {100*stats["per_class"][n]["recall"]:.1f}%' for n in LABELS),fill='#17212b',font=font(21))
    d.text((margin,883),'One slide only. Root 4 helps select this version; another slide is still needed for independent evaluation.',fill='#374151',font=font(20))
    canvas.save(path)

def train(manifest,args):
    rng=np.random.default_rng(args.seed);torch.manual_seed(args.seed)
    unet=module('multiroot_unet',Path(__file__).with_name('09_three_class_unet.py'))
    device=unet.choose_device('auto');model=unet.ThreeClassUNet().to(device)
    RUNS.mkdir(exist_ok=True);out=RUNS/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+f'_seed{args.seed}');out.mkdir();(out/'programs').mkdir()
    for name in [Path(__file__).name,'04_small_unet.py','09_three_class_unet.py']:
        shutil.copy2(Path(__file__).with_name(name),out/'programs'/name)
    samples=[]
    for p in manifest['patches']:
        rgb=np.asarray(Image.open(DATA/p['image'])).copy();mask=np.asarray(Image.open(DATA/p['mask'])).copy();samples.append((rgb,mask))
    eligible={c:[i for i,(_,m) in enumerate(samples) if (m==c).sum()>=512] for c in [1,2,3]}
    for c in eligible:assert eligible[c]
    pair=json.loads(PAIR.read_text());vp=PAIR.parent
    assert sha(vp/pair['image_file'])==pair['image_sha256'] and sha(vp/pair['mask_file'])==pair['mask_sha256']
    vrgb=np.asarray(Image.open(vp/pair['image_file']));vtruth=np.asarray(Image.open(vp/pair['mask_file']));old=np.asarray(Image.open(OLD))
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.steps,eta_min=args.lr*.1)
    plan={'purpose':'Multiple-root development experiment','seed':args.seed,'steps_requested':args.steps,'batch_size':6,'learning_rate_initial':args.lr,'loss':'Mean of per-class cross entropy (classes present) + 0.5 masked multiclass soft Dice. Mask 0/255 ignored.','augmentation':'Class-balanced field sampling; random 256/384/512 source crop resampled to256, paired rotations90/flips, image-only gain0.6-1.6 and gamma0.8-1.2. RGB/255. Source files unchanged.','architecture':'Existing 482067-parameter ThreeClassUNet unchanged','training_roots':[1,2,3,5],'development_validation_root':4,'reserved_root':6,'checkpoint_selection':'Highest mean tissue IoU on labelled root4 pixels; root4 is validation, not a final test.','data_manifest':str(DATA/'manifest.json'),'data_manifest_sha256':sha(DATA/'manifest.json'),'reference_sha256':pair['mask_sha256'],'baseline_metrics':metrics(old,vtruth),'PAT_reference':'https://pmc.ncbi.nlm.nih.gov/articles/PMC10981931/','postprocessing':'None. Raw image-only predictions.'}
    write_json(out/'experiment_plan.json',plan)
    print(f'Training on {len(samples)} fields from roots1/2/3/5; root4 used ONLY for validation. Device {device}.',flush=True)
    history=[];best=-1.;best_step=0;start=time.monotonic()
    for step in range(1,args.steps+1):
        ims=[];targets=[]
        for j in range(6):
            c=(step+j)%3+1;i=int(rng.choice(eligible[c]));rgb,mask=samples[i]
            yy,xx=np.where(mask==c);k=int(rng.integers(len(xx)));size=int(rng.choice([256,384,512]));ox=int(np.clip(xx[k]-rng.integers(size//4,3*size//4),0,512-size));oy=int(np.clip(yy[k]-rng.integers(size//4,3*size//4),0,512-size))
            im=Image.fromarray(rgb[oy:oy+size,ox:ox+size]).resize((256,256),Image.Resampling.BILINEAR);ma=Image.fromarray(mask[oy:oy+size,ox:ox+size]).resize((256,256),Image.Resampling.NEAREST)
            im=np.asarray(im).astype('float32')/255;ma=np.asarray(ma).copy();rot=int(rng.integers(4));im=np.rot90(im,rot);ma=np.rot90(ma,rot)
            if rng.random()<.5:im=im[:,::-1];ma=ma[:,::-1]
            im=np.clip(im**rng.uniform(.8,1.2)*rng.uniform(.6,1.6),0,1)
            tar=np.full(ma.shape,255,dtype='int64');valid=np.isin(ma,[1,2,3]);tar[valid]=ma[valid]-1
            ims.append(torch.from_numpy(np.ascontiguousarray(im)).permute(2,0,1));targets.append(torch.from_numpy(np.ascontiguousarray(tar)))
        x=torch.stack(ims).to(device);t=torch.stack(targets).to(device);model.train();optimizer.zero_grad(set_to_none=True);logits=model(x)
        ce=F.cross_entropy(logits,t,ignore_index=255,reduction='none');valid=t!=255;probs=logits.softmax(1);ces=[];dices=[]
        for c in range(3):
            hit=t==c
            if hit.any().item():
                ces.append(ce[hit].mean());pc=probs[:,c]*valid;dices.append(1-(2*(pc*hit).sum()+1)/(pc.sum()+hit.sum()+1))
        loss=torch.stack(ces).mean()+.5*torch.stack(dices).mean();assert torch.isfinite(loss).item();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();scheduler.step()
        history.append({'step':step,'loss':float(loss.item()),'learning_rate':optimizer.param_groups[0]['lr']})
        if step==1 or step%100==0 or step==args.steps:
            pred=predict(model,vrgb,device);met=metrics(pred,vtruth);score=met['tissue_mean_iou'];history[-1]['validation_tissue_mean_iou']=score
            print(f'Update {step}/{args.steps}, loss {loss.item():.4f}; root4 recall P/E/B '+ '/'.join(f'{100*met["per_class"][n]["recall"]:.1f}' for n in LABELS)+f'; tissue IoU {score:.3f}; {time.monotonic()-start:.0f}s',flush=True)
            if score>best:
                best=score;best_step=step;torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'model_weights.pt');Image.fromarray(pred).save(out/'predicted_mask.png');write_json(out/'best_validation_metrics.json',met)
            write_json(out/'training_history.json',history)
    model.load_state_dict(torch.load(out/'model_weights.pt',map_location='cpu',weights_only=True));pred=predict(model,vrgb,device)
    assert np.array_equal(pred,np.asarray(Image.open(out/'predicted_mask.png')))
    met=metrics(pred,vtruth);Image.fromarray(PAL[pred]).save(out/'predicted_mask_preview.png');comparison(vrgb,vtruth,old,pred,met,out/'root4_before_after.png')
    for p,h in manifest['sources_sha256'].items():assert sha(p)==h, f'Source changed during run: {p}'
    record={**plan,'best_step':best_step,'completed_steps':args.steps,'elapsed_seconds':time.monotonic()-start,'validation_metrics':met,'model_weights_sha256':sha(out/'model_weights.pt'),'prediction_sha256':sha(out/'predicted_mask.png'),'source_files_unchanged':True,'torch_version':torch.__version__,'device':str(device)}
    write_json(out/'result.json',record)
    (out/'READ_ME.txt').write_text('Multiple-root learning experiment. See experiment_plan.json for the changes, result.json for metrics, training_history.json for all checkpoints. Root4 was not trained on, but was used to select the best checkpoint, so it is development validation. Root6 is reserved. All results come from one slide; no independent-slide accuracy claim. No annotations changed. No postprocessing or reference-mask copying was used.\n')
    print('SAVED: '+str(out),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--steps',type=int,default=2000);parser.add_argument('--seed',type=int,default=34);parser.add_argument('--lr',type=float,default=.0003);args=parser.parse_args()
    assert 1<=args.steps<=10000 and 0<args.lr<=.01
    manifest=prepare()
    if not args.prepare_only:train(manifest,args)
    else:print(f'Dataset verified: {len(manifest["patches"])} fields. No training.',flush=True)

if __name__=='__main__':main()
