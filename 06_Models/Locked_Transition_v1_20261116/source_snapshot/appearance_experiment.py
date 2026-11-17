"""A bounded reviewed-window appearance experiment, not a validated length model.

The published PAT encoder is frozen. Only a linear appearance head is fitted.
Root 3 geometry is an existing human-assisted route. Its expert T is loaded
only after inference, for comparison. Nothing replaces the working model.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates, gaussian_filter1d
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '02_Data/10_Cell_Appearance_Review'
OUT = ROOT / '04_Results/63_Cell_Appearance_Experiment'
NAMES = ['Clear Endoderm', 'Mixed appearance', 'Clear Periderm']
COLORS = ['#1aa65e', '#b44dc5', '#db303f']
WIDTH, HEIGHT, STRIDE = 320, 256, 64
SEED = 1617


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def font(size):
    return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', size)


def standardize(rgb):
    """Native-size field, centered vertically; photometric scaling only."""
    h, w = rgb.shape[:2]
    row_energy = rgb[:, w//5:4*w//5, 1].astype(float).sum(1)
    yc = int(np.argmax(gaussian_filter1d(row_energy, 12)))
    xc = w // 2
    out = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    x0, y0 = xc-WIDTH//2, yc-HEIGHT//2
    sx0, sy0, sx1, sy1 = max(0,x0), max(0,y0), min(w,x0+WIDTH), min(h,y0+HEIGHT)
    out[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = rgb[sy0:sy1, sx0:sx1]
    denominator = max(16., float(np.percentile(out[:,:,1], 99.5)))
    out = np.clip(out.astype(float) * (160./denominator), 0, 255).astype('uint8')
    return out, {'native_size':[WIDTH,HEIGHT], 'input_center_xy':[xc,yc],
                 'intensity_multiplier':160./denominator,
                 'outside_reviewed_window':'zero padding; no extra source tissue added',
                 'resized':False}


def ordered_labels(prob, penalty=.25):
    """Nondecreasing E/M/P, allowing absent classes and all-one-class output."""
    logp = np.log(np.maximum(prob, 1e-8))
    n = len(prob)
    score = logp[0].copy()
    parent = np.zeros((n,3), dtype=int)
    for i in range(1,n):
        next_score = np.empty(3)
        for k in range(3):
            candidates = score[:k+1].copy()
            candidates[:k] -= penalty
            j = int(np.argmax(candidates))
            next_score[k] = candidates[j] + logp[i,k]
            parent[i,k] = j
        score = next_score
    labels = np.empty(n, dtype=np.int64)
    labels[-1] = int(np.argmax(score))
    for i in range(n-1,0,-1):
        labels[i-1] = parent[i,labels[i]]
    return labels


def verify_decoder():
    for k in range(3):
        p = np.full((20,3), .005); p[:,k] = .99
        assert np.all(ordered_labels(p)==k), 'Must not manufacture missing states.'
    labels = np.repeat(np.arange(3), 8)
    p = np.full((24,3), .005); p[np.arange(24),labels] = .99
    assert np.array_equal(ordered_labels(p), labels)


def teaching_figure(images, examples, path):
    rows=(len(images)+2)//3
    im = Image.new('RGB',(1000,140+rows*310),'white'); d=ImageDraw.Draw(im)
    d.text((18,12),f'{len(images)} reviewed examples used for this experiment',font=font(26),fill='black')
    d.text((18,49),'Same native pixel scale; centered and brightness-normalized for the classifier.',font=font(19),fill='#334155')
    for i,a in enumerate(images):
        ex=examples[i];k=ex['class_index'];x=10+(i%3)*330;y=86+(i//3)*310
        d.text((x,y),f'{ex["id"]}  {NAMES[k]}',font=font(22),fill=COLORS[k])
        im.paste(Image.fromarray(a),(x,y+36))
    d.text((18,106+rows*310),'All windows come from ONE slide. Augmented copies are not independent examples.',font=font(21),fill='black')
    im.save(path)


def score_figure(distance, prob, labels, t_candidate, t_reference, interval, output, corrected_root3=False):
    im=Image.new('RGB',(1450,650),'white'); d=ImageDraw.Draw(im)
    d.text((24,16),'Appearance experiment: scanning the reviewed root 3 path',font=font(29),fill='black')
    scope=('Root3 contributes Endoderm teaching windows. This is a correction check; route supplied by a human.' if corrected_root3 else
           'New appearance head trained on root 2 only. Familiar same-slide check; route supplied by a human.')
    d.text((24,58),scope,font=font(21),fill='#334155')
    x0,y0,w,h=90,125,1300,280
    xx=lambda v:x0+w*float(v)/float(distance[-1])
    yy=lambda v:y0+h*(1-float(v))
    for v in [0,.25,.5,.75,1]:
        y=yy(v);d.line((x0,y,x0+w,y),fill='#dddddd');d.text((35,y-10),str(v),font=font(18),fill='black')
    for k in range(3):
        d.line([(xx(s),yy(p)) for s,p in zip(distance,prob[:,k])],fill=COLORS[k],width=3)
        d.text((x0+k*390,420),NAMES[k],font=font(21),fill=COLORS[k])
    d.text((x0,99),'Appearance scores (not calibrated confidence)',font=font(19),fill='black')
    d.text((x0,455),'Ordered interpretation:',font=font(21),fill='black')
    for i,k in enumerate(labels):
        left=distance[i]-STRIDE/2;right=distance[i]+STRIDE/2
        d.rectangle((max(x0,xx(left)),490,min(x0+w,xx(right)),520),fill=COLORS[int(k)])
    # Expert information is a separate comparison overlay, not a classifier input.
    d.rectangle((xx(interval[0]),y0,xx(interval[1]),y0+h),outline='black',width=2)
    d.line((xx(t_reference),y0,xx(t_reference),525),fill='black',width=2)
    if t_candidate is not None:
        for y in range(y0,525,14):d.line((xx(t_candidate),y,xx(t_candidate),y+7),fill='#0078bc',width=3)
    d.text((x0,540),'Distance from photographed left cut end along the main root (native pixels)',font=font(21),fill='black')
    for v in np.arange(0,distance[-1]+1,4000):d.text((xx(v)-12,567),f'{int(v):,}',font=font(18),fill='black')
    msg='Black: your reviewed T. Blue dashed: model candidate.' if t_candidate is not None else 'Black: your reviewed T. No sustained E -> mixed -> P boundary was recovered.'
    d.text((x0,610),msg,font=font(21),fill='black');im.save(output)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-name',default='run_01',help='New folder name; existing results are never overwritten.')
    parser.add_argument('--labels',type=Path,default=DATA/'confirmed_appearance_labels_01.json')
    parser.add_argument('--output-parent',type=Path,default=OUT)
    args=parser.parse_args()
    if not args.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_name):
        parser.error('Use only letters, digits, underscores or hyphens for the run name.')
    start=time.monotonic(); verify_decoder(); torch.set_num_threads(4)
    torch.manual_seed(SEED); rng=np.random.default_rng(SEED)
    run=args.output_parent/args.run_name; run.mkdir(parents=True, exist_ok=False)
    review=json.loads(args.labels.read_text())
    assert review['confirmed'] and set(x['class_index'] for x in review['examples'])=={0,1,2}
    examples=review['examples'];class_counts=np.bincount([ex['class_index'] for ex in examples],minlength=3)
    roots={ex.get('root',review['root']) for ex in examples};corrected_root3=3 in roots
    comparison_scope=('Root3 correction/teaching check; expert-labelled root3 windows train this head. Not independent validation.' if corrected_root3 else
                      'Familiar same-slide development, not blind evaluation. No new-slide test.')
    fixed_weights=ROOT/'00_Software/PAT_Reference/pat_state_dict.pt'
    working=ROOT/'06_Models/working_model.pt'
    default_hash=sha(working)
    source=Path(review['source']); source_hash=sha(source)
    assert source_hash==review['source_sha256']
    # Save the planned design before loading any root3 image or reference.
    plan={'created_utc':datetime.now(timezone.utc).isoformat(), 'labels':str(args.labels.resolve()),
          'labels_sha256':sha(args.labels),
          'purpose':'Feasibility experiment; NOT a validated replacement U-Net or automatic length pipeline',
          'training_unique_windows':len(examples),'training_unique_roots':len(roots),'training_unique_slides':1,
          'encoder_weights':str(fixed_weights),'encoder_sha256':sha(fixed_weights),
          'encoder_training':False,'public_encoder_original_training_membership':'unknown',
          'model':'Frozen PAT ResNet34 multilevel pooled features plus linear 3-appearance head',
          'features':'L2-normalized global mean for encoder stages 1-5, concatenated and normalized',
          'preprocessing':'320x256 native pixels, vertical intensity-centering, p99.5 green scaled to160; no geometric resizing; zero-pad outside reviewed window',
          'augmentation':'48 photometric/flip versions per reviewed window; gain0.8-1.2; gamma0.9-1.1. Not independent examples.',
          'updates':500,'optimizer':'AdamW lr0.02 weight_decay0.05','seed':SEED,
          'selection':'Final head; no checkpoint selection. Expert error correction is part of development when root3 supplies a teaching field.',
          'class_balance':'Each appearance class contributes1/3of loss; windows within a class share its weight equally.',
          'class_window_counts':class_counts.tolist(),
          'scan_stride_native_pixels':STRIDE, 'sequence_rule':'Monotonic E/M/P with0.25 log-score transition penalty. Any state may be absent. Candidate requires>=3 successive samples in every state.',
          'reference_use':'Only after prediction. Route is human-assisted. No expert T used in features or decoding.',
          'comparison_status':comparison_scope,
          'limitations':['Very few fields; root thickness, intensity and context can confound appearance.',
              'Frozen features are not demonstrated explicit cell-shape measurements.',
              'Window labels do not establish a pixel-precise boundary. 320-pixel field limits localization.',
              'Main-root path is supplied, not independently detected. No automatic lengths reported.'],
          'default_model_sha256_before':default_hash}
    save(run/'experiment_plan.json',plan);shutil.copy2(__file__,run/Path(__file__).name)
    images=[]; transforms=[]
    for ex in review['examples']:
        p=Path(ex['native_crop']);assert sha(p)==ex['crop_sha256']
        rgb=np.array(Image.open(p).convert('RGB'))
        if ex['rotate_quarter_turns']:
            rgb=np.rot90(rgb,ex['rotate_quarter_turns']).copy()
        a,info=standardize(rgb);images.append(a);transforms.append(info)
        Image.fromarray(a).save(run/f'teaching_{ex["id"]}_processed.png')
    teaching_figure(images,examples,run/'TEACHING_INPUTS.png');save(run/'preprocessing.json',transforms)
    encoder=smp.encoders.get_encoder('resnet34',in_channels=3,depth=5,weights=None)
    pat=torch.load(fixed_weights,map_location='cpu',weights_only=True)
    state={k.removeprefix('encoder.'):v for k,v in pat.items() if k.startswith('encoder.')}
    encoder.load_state_dict(state,strict=True);del pat,state
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    encoder.to(device).eval().requires_grad_(False)
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    def features(arrays):
        result=[]
        for pos in range(0,len(arrays),16):
            x=torch.from_numpy(np.stack(arrays[pos:pos+16]).transpose(0,3,1,2).copy()).float().to(device)/255
            with torch.inference_mode():
                layers=encoder((x-mean)/std)[1:]
                v=torch.cat([F.normalize(z.mean((2,3)),dim=1) for z in layers],dim=1)
                result.append(F.normalize(v,dim=1).cpu())
        return torch.cat(result)
    augmented=[];targets=[]
    for ex,a in zip(examples,images):
        k=ex['class_index']
        for j in range(48):
            z=a.astype(float)/255
            if j:
                z=np.clip(z**rng.uniform(.9,1.1)*rng.uniform(.8,1.2),0,1)
                if j%2:z=z[:,::-1]
                if j%4>=2:z=z[::-1]
            augmented.append((z*255).astype('uint8').copy()); targets.append(k)
    print(f'Extracting frozen image features from {len(images)} reviewed windows...',flush=True)
    x=features(augmented).float();target=torch.tensor(targets)
    weights=torch.tensor(1/class_counts,dtype=torch.float32)
    head=torch.nn.Linear(x.shape[1],3)
    opt=torch.optim.AdamW(head.parameters(),lr=.02,weight_decay=.05)
    history=[]
    for update in range(1,501):
        opt.zero_grad();loss=F.cross_entropy(head(x),target,weight=weights);loss.backward();opt.step()
        if update==1 or update%50==0:
            history.append({'update':update,'loss_before_update':float(loss.detach())})
            print(f'Appearance-head update {update}/500; loss {loss.item():.4f}',flush=True)
    head.eval();torch.save(head.state_dict(),run/'appearance_head.pt')
    save(run/'training_history.json',history)
    with torch.inference_mode():
        teaching_features=features(images)
        teaching_prob=head(teaching_features).softmax(1).numpy()
    save(run/'teaching_fit.json',{'scores':teaching_prob.tolist(),'predicted_classes':teaching_prob.argmax(1).tolist(),
        'example_ids':[ex['id'] for ex in examples],'expected_classes':[ex['class_index'] for ex in examples],
        'meaning':'Fit to the same teaching windows, not validation. Scores are not calibrated confidence.'})
    # Take only path coordinates from the route record. No transition index or
    # expert tissue label is read into the classifier or sequence decoder.
    route_path=ROOT/'02_Data/09_Transition_Endpoint_Review/route_proposal_02/reference_route_record.json'
    route=json.loads(route_path.read_text()); path=np.array(route['path_xy_source_pixels'])
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    distance=np.arange(WIDTH/2,arc[-1]-WIDTH/2+1,STRIDE)
    centers=np.c_[np.interp(distance,arc,path[:,0]),np.interp(distance,arc,path[:,1])]
    before=np.c_[np.interp(distance-80,arc,path[:,0]),np.interp(distance-80,arc,path[:,1])]
    after=np.c_[np.interp(distance+80,arc,path[:,0]),np.interp(distance+80,arc,path[:,1])]
    direction=after-before;direction/=np.linalg.norm(direction,axis=1)[:,None]
    Image.MAX_IMAGE_PIXELS=300_000_000
    with Image.open(source) as im:rgb_source=np.array(im.convert('RGB'))
    u,v=np.meshgrid(np.arange(WIDTH)-WIDTH/2+.5,np.arange(HEIGHT)-HEIGHT/2+.5)
    scan=[];scan_preprocess=[]
    print(f'Scanning {len(distance)} fields along the previously reviewed root3 path...',flush=True)
    for xy,unit in zip(centers,direction):
        nx=xy[0]+u*unit[0]-v*unit[1];ny=xy[1]+u*unit[1]+v*unit[0]
        raw=np.stack([map_coordinates(rgb_source[:,:,c],[ny,nx],order=1,mode='constant',cval=0) for c in range(3)],axis=2)
        a,info=standardize(raw);scan.append(a);scan_preprocess.append(info)
    with torch.inference_mode():
        scan_features=features(scan)
        prob=head(scan_features).softmax(1).numpy()
    np.savez_compressed(run/'frozen_features.npz',teaching=teaching_features.numpy(),scan=scan_features.numpy())
    sequence=ordered_labels(prob)
    counts=np.bincount(sequence,minlength=3)
    supported=bool(np.all(counts>=3))
    candidate=None
    if supported:
        first_m=int(np.flatnonzero(sequence==1)[0])
        candidate=float((distance[first_m-1]+distance[first_m])/2)
    prediction={'input':'Source image windows along supplied main-root route. No reference masks or T provided.',
        'route':str(route_path),'route_sha256':sha(route_path),'human_assisted_geometry':True,
        'root3_contributes_a_teaching_window':corrected_root3,
        'class_order':NAMES,'distance_from_left_end_native_pixels':distance.tolist(),
        'source_centers_xy':centers.tolist(),'appearance_scores':prob.tolist(),
        'raw_classes':prob.argmax(1).tolist(),'ordered_classes':sequence.tolist(),
        'ordered_class_sample_counts':counts.tolist(),'sustained_all_three_states':supported,
        'T_candidate_distance_native_pixels':candidate,
        'no_T_was_forced':True,'head_sha256':sha(run/'appearance_head.pt')}
    save(run/'root3_prediction_before_reference_comparison.json',prediction)
    save(run/'scan_preprocessing.json',scan_preprocess)
    # Reference is consulted only now, after saving independent classifier output.
    reference_xy=np.array(route['working_transition_xy'])
    reference_index=int(np.argmin(np.linalg.norm(path-reference_xy,axis=1)))
    t_reference=float(arc[reference_index])
    correction=json.loads((ROOT/'04_Results/61_Reviewed_Root3_Teaching_Correction/root3_class_correction.json').read_text())
    reference_range=correction['reviewed_approximate_T_zone_source_x']
    # Local path around the expert T is x-monotonic; restrict nearest-x search
    # to this neighborhood so another bent portion cannot supply the comparison.
    local=np.flatnonzero(np.abs(arc-t_reference)<1500)
    interval=sorted(float(arc[local[np.argmin(abs(path[local,0]-xv))]]) for xv in reference_range)
    delta=None if candidate is None else candidate-t_reference
    in_range=False if candidate is None else interval[0]<=candidate<=interval[1]
    xy_candidate=None if candidate is None else [float(np.interp(candidate,arc,path[:,j])) for j in range(2)]
    result={**plan,'device':device,'completed_updates':500,'teaching_fit':teaching_prob.tolist(),
        'root3_T_candidate_distance_native_pixels':candidate,'root3_T_candidate_source_xy':xy_candidate,
        'expert_working_T_distance_native_pixels':t_reference,'expert_approximate_T_interval_distance_native_pixels':interval,
        'difference_from_working_T_native_pixels':delta,'candidate_within_expert_interval':bool(in_range),
        'sustained_all_three_states':supported,'ordered_class_sample_counts':counts.tolist(),
        'candidate_resolution_note':'320-pixel-wide windows,64-pixel stride. Do not interpret as an exact pixel boundary or statistical confidence interval.',
        'success_scope':comparison_scope+' Not generalization or automatic length validation.',
        'elapsed_seconds':time.monotonic()-start,'model_promoted':False,
        'source_unchanged':sha(source)==source_hash,'default_model_unchanged':sha(working)==default_hash,
        'external_review':'external reviewer unavailable: subscription login required; no review sent.'}
    assert result['source_unchanged'] and result['default_model_unchanged']
    save(run/'result.json',result)
    score_figure(distance,prob,sequence,candidate,t_reference,interval,run/'ROOT3_APPEARANCE_PROFILE.png',corrected_root3)
    # A local source-photo comparison centered on expert T, for human review.
    box=(int(reference_xy[0])-1000,int(reference_xy[1])-260,int(reference_xy[0])+1000,int(reference_xy[1])+260)
    photo=Image.fromarray(rgb_source).crop(box).resize((1400,364),Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(1440,640),'white');d=ImageDraw.Draw(canvas)
    d.text((20,14),'Root 3: source image and transition comparison',font=font(29),fill='black')
    d.text((20,54),'White = your working T and approximate review interval. Blue = appearance-model T, if recovered.',font=font(21),fill='#334155')
    canvas.paste(photo,(20,100));d=ImageDraw.Draw(canvas)
    xplot=lambda val:20+(float(val)-box[0])*.7
    for xv in reference_range:
        for y in range(100,464,16):d.line((xplot(xv),y,xplot(xv),y+7),fill='white',width=2)
    d.line((xplot(reference_xy[0]),100,xplot(reference_xy[0]),464),fill='white',width=2)
    if xy_candidate is not None and box[0]<=xy_candidate[0]<=box[2]:
        d.line((xplot(xy_candidate[0]),100,xplot(xy_candidate[0]),464),fill='#00aaff',width=3)
    message=('No reliable internal T recovered by the ordered appearance rule.' if candidate is None else
             f'Candidate is {abs(delta):,.0f} native pixels {"rightward" if delta>0 else "leftward"} along the path from your working T.')
    d.text((20,489),message,font=font(24),fill='black')
    d.text((20,529),'This is an exploratory result. The main-root route was supplied; no automatic length was calculated.',font=font(21),fill='black')
    note=(f'{len(examples)} teaching windows; root3 now supplies an Endoderm correction. No independent validation.' if corrected_root3 else
          'Three teaching windows from root 2. Same-slide root 3 check; no new-slide validation.')
    d.text((20,566),note,font=font(21),fill='black')
    canvas.save(run/'ROOT3_TRANSITION_COMPARISON.png')
    print(json.dumps({'result':str(run/'result.json'),'teaching_classes':teaching_prob.argmax(1).tolist(),
        'ordered_counts':counts.tolist(),'candidate_native_arc':candidate,'expert_native_arc':t_reference,
        'error_pixels':delta,'candidate_within_reviewed_interval':bool(in_range)},indent=2),flush=True)


if __name__=='__main__':
    main()
