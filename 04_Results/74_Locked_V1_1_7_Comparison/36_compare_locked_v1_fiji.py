"""Selected-root pilot: unchanged locked v1 versus saved Fiji lines.

Prepare geometry from source fluorescence, supplying only the shared right-hand
endpoint from each ROI. Never supply the human transition to the detector.
Run --phase prepare, visually inspect routes, then --phase predict/compare.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import hashlib
import json
import math
import struct
import subprocess
from datetime import datetime, timezone
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
BUNDLE = BASE / '06_Models/Locked_Transition_v1_20261116'
sys.path.insert(0, str(BUNDLE / 'source_snapshot'))
from extend_root3_reference_route import graph_route
SOURCE = BASE / '02_Data/15_Additional_Slides_20261117/01_Current_Check/1_7.tif'
REF = BASE / '04_Results/73_V1_Manual_Comparison_Pilot/01_Manual_References'
OUT = BASE / '04_Results/74_Locked_V1_1_7_Comparison'
SCALE = .5299
Image.MAX_IMAGE_PIXELS = 300_000_000

# Coarse landmarks chosen only from the original slide overview for root identity.
# They define a broad corridor, not tissue classes or transition coordinates.
LANDMARKS = {
    1: [[200,820],[1600,500],[3500,200],[5000,100],[6500,300],
        [8500,240],[10000,650],[12000,1500],[13500,1530]],
    2: [[400,3300],[2000,2450],[4000,1850],[6000,2250],
        [8000,2680],[9300,2550],[11000,2900],[13000,3200]],
    6: [[300,6000],[1500,5940],[3500,6400],[5500,6250],
        [7500,6500],[9500,6140],[11500,6360],[12500,6360],[14000,6800]],
}

def read(p): return json.loads(p.read_text())
def save(p,o): p.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def verify():
    r=subprocess.run([sys.executable,str(BUNDLE/'verify_checkpoint.py'),'--require-read-only'],capture_output=True,text=True,check=True)
    return json.loads(r.stdout)

def roi_points(p):
    b=p.read_bytes(); u=lambda o:struct.unpack_from('>H',b,o)[0]
    if b[:4]!=b'Iout' or b[6]!=5 or u(50)!=0:
        raise ValueError('Expected saved integer ImageJ segmented line.')
    n=u(16);top,left,_,_=struct.unpack_from('>hhhh',b,8)
    x=struct.unpack_from('>'+str(n)+'H',b,64)
    y=struct.unpack_from('>'+str(n)+'H',b,64+2*n)
    return np.c_[np.array(x)+left,np.array(y)+top].astype(float)

def right_endpoint_only(p):
    # Only this endpoint is permitted into route preparation. Human T is not used.
    points=roi_points(p)
    return points[0].copy() if points[0,0]>points[-1,0] else points[-1].copy()

def make_route(im,root):
    roi=REF/f'1_7_root{root:02d}_Periderm.roi'
    J=right_endpoint_only(roi)
    landmarks=np.array(LANDMARKS[root]+[J.tolist()],float)
    if np.any(np.diff(landmarks[:,0])<=0): raise ValueError('Landmarks must increase in x.')
    box=[0,max(0,int(landmarks[:,1].min())-500),int(J[0])+450,min(im.height,int(landmarks[:,1].max())+550)]
    box[2]=min(im.width,box[2]);rgb=np.array(im.crop(box))
    h,w=rgb.shape[:2];pad=np.pad(rgb[:,:,1],((0,(-h)%4),(0,(-w)%4)))
    coarse=np.array(Image.fromarray(pad).resize((pad.shape[1]//4,pad.shape[0]//4),Image.Resampling.BOX),dtype=float)
    offset=np.array(box[:2])+1.5
    start=(landmarks[0]-offset)/4;stop=(J-offset)/4
    yy,xx=np.indices(coarse.shape);source_x=xx*4+offset[0]
    line_y=np.interp(source_x,landmarks[:,0],landmarks[:,1])
    corridor=(np.abs(yy*4+offset[1]-line_y)<350)&(source_x>=landmarks[0,0]-100)&(source_x<=J[0]+100)
    smooth=ndi.gaussian_filter(coarse,1)
    attempts=[];selected=None
    for threshold,closing in [(16,2),(12,2),(8,2),(5,2),(3,2),(3,3)]:
        fg=ndi.binary_fill_holes(ndi.binary_closing((smooth>threshold)&corridor,iterations=closing))&corridor
        labels,_=ndi.label(fg)
        ids=[]
        for point in (start,stop):
            candidates=np.argwhere(fg)
            if not len(candidates): ids.append(None);continue
            k=int(np.argmin(np.linalg.norm(candidates[:,::-1]-point,axis=1)))
            y,x=candidates[k];ids.append(int(labels[y,x]))
        attempt={'threshold':threshold,'closing':closing,'component_ids':ids}
        attempts.append(attempt)
        if ids[0] is None or ids[0]!=ids[1]:continue
        raw,ends=graph_route(labels==ids[0],start,stop)
        attempt['endpoint_snap_distances_native_pixels']=(np.array(ends)*4).tolist()
        # The left landmark is only an approximate scan start, not a length endpoint.
        # The shared J remains tightly constrained because it ends the measurement.
        if ends[0]*4>200 or ends[1]*4>110:continue
        raw=raw*4+offset
        a=np.r_[0,np.cumsum(np.linalg.norm(np.diff(raw,axis=0),axis=1))]
        positions=np.r_[np.arange(0,a[-1],4),a[-1]]
        path=np.c_[np.interp(positions,a,raw[:,0]),np.interp(positions,a,raw[:,1])]
        path=ndi.gaussian_filter1d(path,5,axis=0,mode='nearest')
        path=np.vstack([path,J])
        path=path[np.r_[True,np.linalg.norm(np.diff(path,axis=0),axis=1)>1e-8]]
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
        if np.max(np.linalg.norm(np.diff(path,axis=0),axis=1))>110:continue
        selected=(path,arc,threshold,closing);break
    if selected is None:raise ValueError(f'Root{root} no acceptable route: {attempts}')
    path,arc,threshold,closing=selected
    folder=OUT/f'root{root:02d}';folder.mkdir()
    Image.fromarray(rgb).save(folder/'image.png')
    record={'source_image':str(SOURCE),'source_sha256':sha(SOURCE),
        'source_crop_xyxy':box,'source_size_xy':[w,h],
        'path_xy_crop_pixels':(path-np.array(box[:2])).tolist(),
        'distance_along_path_pixels':arc.tolist(),
        'shared_J_source_xy':J.tolist(),'scan_start_source_xy':path[0].tolist(),
        'source_only_tissue_geometry':True,'manual_T_used':False,'tissue_labels_used':False,
        'shared_J_from_user_ROI':str(roi),'roi_sha256':sha(roi),
        'geometry_landmarks_source_xy':landmarks.tolist(),
        'settings':{'downsample':4,'green_threshold':threshold,'closing':closing,'sigma_coarse':1,'corridor_half_width_native_px':350,'resample_px':4,'smoothing_sigma_samples':5},
        'attempts':attempts,'limitations':'Assisted pilot: main-root identity/geometry checked during development; same user right endpoint J. Left scan start is not a measured cut end. Human T withheld.'}
    save(folder/'image_only_route.json',record)
    fig,axes=plt.subplots(2,1,figsize=(18,6),layout='constrained')
    for ax in axes:
        ax.imshow(rgb,extent=[box[0],box[2],box[3],box[1]]);ax.set_xlim(box[0],box[2]);ax.axis('off')
    axes[0].set_title(f'Root {root}: original source; no manual transition supplied',loc='left',fontsize=15)
    axes[1].plot(path[:,0],path[:,1],color='white',lw=.9)
    axes[1].plot(*J,'o',color='#ff9f1c',ms=5)
    axes[1].set_title('White: fluorescence-derived main-root path. Orange dot: shared right endpoint J.',loc='left',fontsize=13)
    fig.savefig(folder/'ROUTE_CHECK.png',dpi=120);plt.close(fig)
    print(json.dumps({'root':root,'route_points':len(path),'length_px':arc[-1],'threshold':threshold,'J':J.tolist(),'route_check':str(folder/'ROUTE_CHECK.png')}),flush=True)

def prepare():
    OUT.mkdir(exist_ok=False)
    assert sha(SOURCE)=='487d1086e4caaca419431a487beb131f050eece8ef8347283cabdf2f8a82a604'
    save(OUT/'plan.json',{'created_utc':now(),'roots':[1,2,6],'checkpoint_before':verify(),
        'method':'Fixed-v1 inference; manually chosen roots; source-fluorescence main-root route; shared human right endpoint J; no human transition input.',
        'comparison':'After predictions saved, compare T-to-J arc length to Fiji segmented line. Report abstentions. Root02 repeated trace is the primary reference; original scalar retained.',
        'scale_pixels_per_micrometre':SCALE,'training':False,'limits':'Selected three-root pilot on one slide. Historical PAT encoder training exposure unknown. Not fully automatic whole-slide validation.'})
    im=Image.open(SOURCE).convert('RGB')
    for root in (1,2,6):make_route(im,root)
    (OUT/Path(__file__).name).write_text(Path(__file__).read_text())

def predict():
    qa=read(OUT/'route_visual_check.json')
    if not qa['passed']:raise ValueError('Inspect main-root routes before prediction.')
    for root in (1,2,6):
        folder=OUT/f'root{root:02d}'
        assert sha(folder/'image_only_route.json')==qa['route_hashes'][str(root)]
        result=subprocess.run([sys.executable,str(BUNDLE/'run_checkpoint.py'),'--image',str(folder/'image.png'),'--route',str(folder/'image_only_route.json'),'--output',str(folder/'inference')],text=True,capture_output=True)
        (folder/'inference_stdout.txt').write_text(result.stdout+'\n'+result.stderr)
        if result.returncode:raise RuntimeError(result.stderr)
        print(f'Root {root}: '+result.stdout,flush=True)
    save(OUT/'checkpoint_after.json',verify())

def project_point(point,path,arc):
    delta=np.diff(path,axis=0);norm2=(delta*delta).sum(1)
    t=np.clip(((point-path[:-1])*delta).sum(1)/norm2,0,1)
    projected=path[:-1]+t[:,None]*delta
    j=int(np.argmin(np.linalg.norm(projected-point,axis=1)))
    return float(arc[j]+t[j]*math.sqrt(norm2[j])),float(np.linalg.norm(projected[j]-point)),projected[j]

def compare():
    # All predictions must exist before reference transition endpoints are opened here.
    prediction_files=[OUT/f'root{r:02d}/inference/prediction.json' for r in (1,2,6)]
    assert all(p.is_file() for p in prediction_files)
    hashes={str(r):sha(p) for r,p in zip((1,2,6),prediction_files)}
    results=[];image=Image.open(SOURCE).convert('RGB')
    blue='#27b8ff';manual_color='#ffffff';orange='#ffab3d'
    for root,predfile in zip((1,2,6),prediction_files):
        folder=OUT/f'root{root:02d}';pred=read(predfile);route=read(folder/'image_only_route.json')
        box=route['source_crop_xyxy'];path=np.array(route['path_xy_crop_pixels'])+np.array(box[:2]);arc=np.array(route['distance_along_path_pixels'])
        points=roi_points(REF/f'1_7_root{root:02d}_Periderm.roi')
        T=points[0] if points[0,0]<points[-1,0] else points[-1]
        J=points[-1] if points[0,0]<points[-1,0] else points[0]
        assert np.array_equal(path[-1],J)
        manual_px=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
        manual_arc,dist,projected=project_point(T,path,arc)
        model_length=None if pred['T_path_pixels'] is None else float((arc[-1]-pred['T_path_pixels'])/SCALE)
        manual_length=manual_px/SCALE
        difference=None if model_length is None else model_length-manual_length
        relative=None if difference is None else 100*difference/manual_length
        common_manual=(arc[-1]-manual_arc)/SCALE
        record={'root_id':root,'manual_length_um':manual_length,'model_assisted_length_um':model_length,
            'model_minus_manual_um':difference,'percent_difference_vs_manual':relative,
            'manual_T_source_xy':T.tolist(),'model_T_source_xy':pred['T_source_xy'],'shared_J_source_xy':J.tolist(),
            'manual_T_projected_arc_pixels':manual_arc,'manual_T_distance_to_computer_path_pixels':dist,
            'manual_T_to_J_on_computer_route_um':common_manual,
            'model_vs_manual_T_shift_um_along_common_route':None if model_length is None else (pred['T_path_pixels']-manual_arc)/SCALE,
            'trace_geometry_difference_um':common_manual-manual_length,
            'interpretation':'Difference from user reference, not independently established biological truth.',
            'roi_sha256':sha(REF/f'1_7_root{root:02d}_Periderm.roi'),'prediction_sha256':hashes[str(root)],
            'route_sha256':sha(folder/'image_only_route.json')}
        if root==2:record['earlier_unsaved_trace_measurement_um']=14014.733
        results.append(record);save(folder/'comparison.json',record)

        # Native source crops around each independently selected transition.
        fig,axes=plt.subplots(2,1,figsize=(14,8),layout='constrained')
        for ax,(label,point,color,style) in zip(axes,[('Your manual transition',T,manual_color,'--'),('Locked V1 transition',pred['T_source_xy'],blue,'-')]):
            if point is None:ax.text(.5,.5,'V1 returned no supported transition',ha='center');ax.axis('off');continue
            px,py=point;x0=int(np.clip(px-900,0,image.width-1800));y0=int(np.clip(py-280,0,image.height-560))
            crop=np.array(image.crop((x0,y0,x0+1800,y0+560)))
            ax.imshow(crop,extent=[x0,x0+1800,y0+560,y0]);ax.axvline(px,color=color,ls=style,lw=2)
            ax.set_title(f'Root {root} — {label} (source x = {px:,.0f} pixels)',loc='left',fontsize=15)
            ax.axis('off')
            ax.plot([x0+60,x0+60+500*SCALE],[y0+510]*2,color='white',lw=3)
            ax.text(x0+60,y0+480,'500 µm',color='white',fontsize=11)
        fig.suptitle('Same original image, two positions — no new training or manual movement of model T',fontsize=14)
        fig.savefig(folder/'TRANSITION_CLOSEUPS.png',dpi=130);plt.close(fig)

    summary={'created_utc':now(),'slide':'1_7','roots':[1,2,6],'results':results,
        'mean_absolute_difference_um':float(np.mean([abs(r['model_minus_manual_um']) for r in results if r['model_minus_manual_um'] is not None])),
        'scale_pixels_per_micrometre':SCALE,'model_unchanged':True,'new_training':False,
        'scope':'Three user-selected roots on one slide. V1 receives image and checked source-derived main-root route, sharing user right endpoint J. Human transitions withheld until predictions were saved.',
        'limits':['Assisted length comparison, not unattended whole-slide measurement.','Root selection excluded other roots the user judged unsuitable.','Published PAT encoder historical image exposure unknown.','Manual repeated root02 line used; original scalar remains archived.','The shared J and calibration are accepted user inputs, not model-detected or independently validated.','Main-root tracing and model appearance windows can still be affected by lateral branches.','No acceptance tolerance was pre-specified; numerical agreement should be reviewed biologically.']}
    save(OUT/'comparison_summary.json',summary)
    fig=plt.figure(figsize=(16,13),layout='constrained')
    gs=fig.add_gridspec(5,1,height_ratios=[.75,1.9,1.9,1.9,.9])
    title=fig.add_subplot(gs[0]);title.axis('off')
    title.text(0,.9,'Slide 1_7: locked V1 compared with your Fiji measurements',fontsize=22,weight='bold')
    title.text(0,.42,'Blue solid line = model T    |    White dashed line = your T    |    Orange dot = shared right endpoint J',fontsize=13)
    title.text(0,.05,'Periderm length follows the curved main root from T to J. Same V1; no new training.',fontsize=13)
    for row,r in enumerate(results,1):
        root=r['root_id'];ax=fig.add_subplot(gs[row]);route=read(OUT/f'root{root:02d}/image_only_route.json');b=route['source_crop_xyxy']
        rgb=np.array(Image.open(OUT/f'root{root:02d}/image.png'));path=np.array(route['path_xy_crop_pixels'])+np.array(b[:2]);arc=np.array(route['distance_along_path_pixels'])
        ax.imshow(rgb,extent=[b[0],b[2],b[3],b[1]])
        manual=r['manual_T_source_xy'];model=r['model_T_source_xy'];J=r['shared_J_source_xy']
        ax.axvline(manual[0],color=manual_color,ls='--',lw=1.7)
        if model is not None:
            ax.axvline(model[0],color=blue,lw=2)
            pred=read(OUT/f'root{root:02d}/inference/prediction.json')
            segment=path[arc>=pred['T_path_pixels']]
            ax.plot(segment[:,0],segment[:,1],color=blue,lw=.8,alpha=.85)
        ax.plot(*J,'o',color=orange,ms=5)
        ml='No T' if r['model_assisted_length_um'] is None else f'{r["model_assisted_length_um"]:,.0f} µm'
        diff='not measurable' if r['model_minus_manual_um'] is None else f'{r["model_minus_manual_um"]:+,.0f} µm ({r["percent_difference_vs_manual"]:+.1f}%)'
        ax.set_title(f'Root {root}   Your line: {r["manual_length_um"]:,.0f} µm   |   V1: {ml}   |   Difference: {diff}',loc='left',fontsize=14,pad=10)
        ax.set_xlim(0,b[2]);ax.axis('off')
    foot=fig.add_subplot(gs[4]);foot.axis('off')
    foot.text(0,.88,'What this shows: V1 returned a T for all three roots, but the positions differ from your references.',fontsize=14,weight='bold')
    foot.text(0,.52,'Assisted pilot: root paths were checked and your right endpoints were shared. Your T points were withheld.',fontsize=12)
    foot.text(0,.21,'Scale: 0.5299 px/µm. Root 2 uses the redrawn, saved line. Three selected roots on one slide; not general validation.',fontsize=12)
    fig.savefig(OUT/'V1_VS_YOUR_MEASUREMENTS.png',dpi=130);fig.savefig(OUT/'V1_VS_YOUR_MEASUREMENTS.pdf');plt.close(fig)
    lines=['LOCKED V1 — SLIDE1_7 MANUAL COMPARISON','',summary['scope'],'',
        'Root | Your saved Fiji line (um) | V1 assisted length (um) | V1 minus yours (um) | Relative difference']
    for r in results:
        lines.append(f'{r["root_id"]} | {r["manual_length_um"]:.3f} | {r["model_assisted_length_um"]:.3f} | {r["model_minus_manual_um"]:+.3f} | {r["percent_difference_vs_manual"]:+.2f}%')
    lines += ['', 'Positive difference = V1 length longer than yours; negative = shorter.',
        'Root02 repeated trace13710.563um is used; earlier14014.733um remains archived.',
        'Model positions have not been shifted to agree. No retraining or threshold changes.',
        'Lengths use native-coordinate curved arc distances divided by0.5299px/um.',
        'Differences combine transition placement and small tracing differences; per-root JSON separates these.',
        'No pre-agreed acceptable error threshold. These discrepancies need review before practical batch use.',
        '', 'LIMITS:',*summary['limits'], '', 'NEXT: review root6 largest disagreement before designing a separate v2; preserve this v1 result.']
    (OUT/'READ_ME_RESULTS.txt').write_text('\n'.join(lines)+'\n')
    assert all(sha(p)==hashes[str(r)] for r,p in zip((1,2,6),prediction_files))
    assert sha(SOURCE)==read(OUT/'root01/image_only_route.json')['source_sha256']
    save(OUT/'verification.json',{'created_utc':now(),'checkpoint':verify(),'source_unchanged':True,
        'prediction_hashes_unchanged_during_comparison':True,'prediction_hashes':hashes,
        'reference_transition_used_for_prediction':False,'shared_reference_J_used':True,
        'figure_visual_check':'Pending','no_neural_training':True})
    (OUT/Path(__file__).name).write_text(Path(__file__).read_text())
    print(json.dumps([{k:r[k] for k in ['root_id','manual_length_um','model_assisted_length_um','model_minus_manual_um','percent_difference_vs_manual','trace_geometry_difference_um','manual_T_distance_to_computer_path_pixels']} for r in results],indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','predict','compare'],required=True)
    args=parser.parse_args();{'prepare':prepare,'predict':predict,'compare':compare}[args.phase]()
