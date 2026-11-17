# Uses original PAT methods/code: Copyright (c) 2023 Salk Harnessing Plants Initiative.
# Original MIT notice: 00_Software/PAT_Reference/LICENSE. See THIRD_PARTY_NOTICES.md.
"""Supplementary raw PAT-weight labels and original 2D PAT cleanup comparison.

These diagnostics NEVER replace an anatomical periderm estimate or its gates.
The original checkpoint uses adapted class-safe tiling followed by original
cleanup stages. These are mask/path statistics, not anatomical measurements.
"""
from pathlib import Path
import copy, csv, hashlib, importlib.util, json, math, subprocess
import numpy as np
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
VERSION='pat_path_diagnostics_1.1'
WORKFLOW='Original PAT weights; adapted class-safe tiling; original 2D cleanup; V2 sampled path, not FilFinder'
PALETTE=['#85908b','#dc3241','#19ac64','#a04bd7']
FIELDS=['track_id','diagnostic_status','path_start_type','path_end_type',
        'sampled_path_um','raw_P_labelled_path_um','raw_E_labelled_path_um',
        'raw_Background_path_um','raw_LR_path_um','background_fraction','P_run_count',
        'longest_P_run_um','first_P_path_um','last_P_path_um','PAT_cleanup_P_labelled_path_um',
        'P_detection_state','PAT_cleanup_detection_state','endpoint_anatomically_verified',
        'pipeline_version','diagnostic_version','diagnostic_workflow','raw_mask_sha256','source_sha256','pixels_per_micrometre','scope','diagnostic_reason']

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def sample_path(mask,path,pixels_per_um):
    """Midpoint quadrature, each Euclidean segment subdivided at <=1 pixel.

    Coordinates use nearest pixel centres. All class lengths sum to arc length.
    The path starts at the detected left end, not the predicted transition.
    """
    path=np.asarray(path,dtype=float);mask=np.asarray(mask)
    if mask.ndim!=2 or mask.dtype.kind not in 'ui':
        raise ValueError('Expected a 2D four-class integer mask.')
    if not math.isfinite(pixels_per_um) or pixels_per_um<=0:raise ValueError('Positive calibration required.')
    if path.ndim!=2 or path.shape[1]!=2 or len(path)<2 or not np.isfinite(path).all():
        raise ValueError('A finite polyline with at least two vertices is required.')
    if (path<0).any() or (path[:,0]>mask.shape[1]-1).any() or (path[:,1]>mask.shape[0]-1).any():
        raise ValueError('Path outside mask; sampling was not clipped.')
    delta=np.diff(path,axis=0);distance=np.linalg.norm(delta,axis=1)
    valid=distance>1e-9;start=path[:-1][valid];delta=delta[valid];distance=distance[valid]
    if len(distance)==0:raise ValueError('Zero-length path.')
    counts=np.ceil(distance).astype(int)
    origins=np.repeat(start,counts,axis=0);vectors=np.repeat(delta,counts,axis=0)
    fractions=np.concatenate([(np.arange(n)+.5)/n for n in counts])
    xy=origins+vectors*fractions[:,None];lengths=np.repeat(distance/counts,counts)
    ij=np.floor(xy+.5).astype(int);classes=mask[ij[:,1],ij[:,0]]
    if not np.isin(classes,[0,1,2,3]).all():raise ValueError('Invalid sampled class code.')
    edges=np.r_[0,np.cumsum(lengths)];perclass=np.bincount(classes,weights=lengths,minlength=4)
    assert np.isclose(perclass.sum(),distance.sum(),rtol=1e-10)
    p=classes==1;runs=boolean_runs(p);pruns=[r for r in runs if r[2]]
    stats={'sampled_path_um':float(distance.sum()/pixels_per_um),
           'raw_P_labelled_path_um':float(perclass[1]/pixels_per_um),
           'raw_E_labelled_path_um':float(perclass[2]/pixels_per_um),
           'raw_Background_path_um':float(perclass[0]/pixels_per_um),
           'raw_LR_path_um':float(perclass[3]/pixels_per_um),
           'background_fraction':float(perclass[0]/distance.sum()),
           'P_run_count':len(pruns),
           'longest_P_run_um':max([float((edges[b]-edges[a])/pixels_per_um) for a,b,_ in pruns],default=0.),
           'first_P_path_um':None if not pruns else float(edges[pruns[0][0]]/pixels_per_um),
           'last_P_path_um':None if not pruns else float(edges[pruns[-1][1]]/pixels_per_um),
           'max_sample_step_pixels':float(lengths.max()),'path_length_pixels':float(distance.sum())}
    return stats,{'xy':xy,'classes':classes,'lengths':lengths,'edges':edges}

def boolean_runs(flags):
    flags=np.asarray(flags,dtype=bool)
    if len(flags)==0:return []
    cuts=np.r_[0,np.flatnonzero(flags[1:]!=flags[:-1])+1,len(flags)]
    return [(int(a),int(b),bool(flags[a])) for a,b in zip(cuts[:-1],cuts[1:])]

def cleanup_candidate(samples,pixels_per_um,max_gap=150.,max_end_gap=250.):
    """Explicit experiment: enclosed-gap closing, rightmost P run.

    Unlike original 2D cleanup, cannot bridge between separate root tracks.
    No outside gaps filled, no class replaced in the saved raw mask.
    Not a validated transition/hypocotyl detector; never used for acceptance.
    """
    raw=samples['classes']==1;clean=raw.copy();edges=samples['edges']
    for a,b,value in boolean_runs(raw):
        if not value and a>0 and b<len(raw) and edges[b]-edges[a]<=max_gap:
            clean[a:b]=True
    runs=[r for r in boolean_runs(clean) if r[2]]
    added=float(samples['lengths'][clean&~raw].sum()/pixels_per_um)
    out={'cleanup_candidate_um':None,'cleanup_added_path_um':added,
         'cleanup_status':'no_P_detected_not_confirmed_absence','cleanup_start_path_pixels':None,
         'cleanup_gap_limit_pixels':max_gap,'cleanup_end_gap_limit_pixels':max_end_gap}
    if not runs:return out
    a,b,_=runs[-1];end_gap=float(edges[-1]-edges[b])
    if end_gap>max_end_gap:
        out['cleanup_status']='P_run_does_not_reach_endpoint';return out
    # Count only the retained predicted/closed run, without filling to the endpoint.
    out.update(cleanup_candidate_um=float((edges[b]-edges[a])/pixels_per_um),
               cleanup_status='experimental_only_requires_anatomical_review',
               cleanup_start_path_pixels=float(edges[a]),cleanup_stop_path_pixels=float(edges[b]),
               cleanup_end_gap_pixels=end_gap)
    return out

def diagnostic_for_row(row,mask,scale):
    out={'track_id':row['track_id'],'diagnostic_status':'not_computed','diagnostic_reason':'',
         'path_start_type':'detected_left_end','path_end_type':row.get('endpoint_method'),
         'endpoint_anatomically_verified':False}
    if row.get('status') in ('failed','connected_group','partial_root','low_image_quality'):
        out['diagnostic_reason']='Withheld: '+row['status'];return out,None
    try:stats,samples=sample_path(mask,row.get('path_xy'),scale)
    except (ValueError,TypeError) as e:
        out['diagnostic_reason']=str(e);return out,None
    out.update(stats)
    out['diagnostic_status']='computed_endpoint_uncertain' if row.get('endpoint_method')=='visible_right_end_only' else 'computed_unvalidated_endpoint'
    out['P_detection_state']='P_detected' if stats['P_run_count'] else 'no_sampled_P_not_confirmed_absence'
    out['diagnostic_reason']='Label statistics on the detected left-end-to-automatic-endpoint path; not anatomical periderm length.'
    if out['background_fraction']>.1:out['diagnostic_reason']+=' More than 10% Background: inspect track/mask alignment.'
    return out,samples

def draw_diagnostic(rgb,path,samples,destination):
    p=np.asarray(path);lo=np.maximum(0,np.floor(p.min(axis=0)-120).astype(int))
    hi=np.minimum(rgb.size,np.ceil(p.max(axis=0)+121).astype(int));photo=rgb.crop((*lo,*hi))
    scale=min(1,1600/photo.width);photo=photo.resize((round(photo.width*scale),round(photo.height*scale)))
    d=ImageDraw.Draw(photo);xy=(samples['xy']-lo)*scale
    for a,b in zip(range(len(xy)-1),range(1,len(xy))):
        d.line([tuple(xy[a]),tuple(xy[b])],fill=PALETTE[int(samples['classes'][a])],width=3)
    photo.save(destination)

def augment(source,output,summary,report=lambda message:None,mask_folder=None):
    """Keep all V2 values; add a separate diagnostic and CSV, never replace them."""
    summary=copy.deepcopy(summary)
    output=Path(output);source=Path(source)
    report('Checking raw PAT-weight labels and per-root post-processing diagnostics…')
    if mask_folder is None:
        prediction_script=PROJECT/'07_Periderm_Length_Proof/predict_slide.py'
        spec=importlib.util.spec_from_file_location('pat_full_diagnostic',prediction_script)
        prediction=importlib.util.module_from_spec(spec);spec.loader.exec_module(prediction)
        prediction.run(source,output/'pat_prediction','published4')
        mask_folder=output/'pat_prediction'
    mask_folder=Path(mask_folder)
    metadata=json.loads((mask_folder/'prediction_record.json').read_text())
    if metadata['source_sha256']!=summary['source_sha256']:raise ValueError('Diagnostic image hash mismatch.')
    if metadata['model_name']!='published4':raise ValueError('Wrong diagnostic model.')
    if metadata['weights_sha256']!='71ba82d7eab40c0b8489f47ee373fd0ea08f2e2e33457c62152a1f8240d68348':raise ValueError('Wrong diagnostic weights.')
    with (PROJECT/'00_Software/PAT_Reference/pat_state_dict.pt').open('rb') as weight_file:
        if hashlib.file_digest(weight_file,'sha256').hexdigest()!=metadata['weights_sha256']:raise ValueError('Diagnostic weights file changed.')
    mask=np.asarray(Image.open(mask_folder/'raw_classes.png'))
    if list(mask.shape[::-1])!=summary['source_dimensions_xy']:raise ValueError('Diagnostic dimensions mismatch.')
    if mask.ndim!=2 or not np.isin(mask,[0,1,2,3]).all():raise ValueError('Invalid diagnostic class mask.')
    folder=output/'model_diagnostics';folder.mkdir()
    report('Comparing original PAT cleanup with the raw mask; no training…')
    with (folder/'postprocessing.log').open('w') as log:
        subprocess.run([str(PROJECT/'00_Software/pat-audit-env/bin/python'),str(HERE/'pat_postprocessing.py'),
                        '--mask',str(mask_folder/'raw_classes.png'),'--output',str(folder/'PAT_cleanup')],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    clean=(np.asarray(Image.open(folder/'PAT_cleanup/measurement/slide.png'))>127).astype(np.uint8)
    rgb=Image.open(source).convert('RGB');rows=[]
    raw_hash=hashlib.sha256((mask_folder/'raw_classes.png').read_bytes()).hexdigest()
    for row in summary['rows']:
        saved=output/f"track_{row['track_id']:02d}/result.json"
        full=json.loads(saved.read_text());diag,samples=diagnostic_for_row(full,mask,summary['pixels_per_micrometre'])
        diag.update(pipeline_version=summary['pipeline_version'],source_sha256=summary['source_sha256'],
                    diagnostic_version=VERSION,diagnostic_workflow=WORKFLOW,raw_mask_sha256=raw_hash,
                    pixels_per_micrometre=summary['pixels_per_micrometre'],PAT_cleanup_detection_state='not_computed',
                    scope='Not anatomical periderm length. Sum of model-labelled portions of a detected path. Zero is not confirmed biological absence; the automatic endpoint can include hypocotyl.')
        if samples is not None:
            name=f"track_{row['track_id']:02d}_labelled_path.png"
            draw_diagnostic(rgb,full['path_xy'],samples,folder/name)
            diag['image']='model_diagnostics/'+name
            np.savez_compressed(folder/f"track_{row['track_id']:02d}_samples.npz",**samples)
            clean_stats,clean_samples=sample_path(clean,full['path_xy'],summary['pixels_per_micrometre'])
            diag['PAT_cleanup_P_labelled_path_um']=clean_stats['raw_P_labelled_path_um']
            diag['PAT_cleanup_detection_state']='P_retained' if clean_stats['P_run_count'] else 'no_P_retained_not_confirmed_absence'
            diag['P_sample_membership_unchanged_by_PAT_cleanup']=bool(np.array_equal(samples['classes']==1,clean_samples['classes']==1))
            clean_name=f"track_{row['track_id']:02d}_PAT_cleanup_path.png"
            draw_diagnostic(rgb,full['path_xy'],clean_samples,folder/clean_name)
            diag['PAT_cleanup_image']='model_diagnostics/'+clean_name
            np.savez_compressed(folder/f"track_{row['track_id']:02d}_PAT_cleanup_samples.npz",**clean_samples)
        rows.append(diag)
    diagnostics={'version':VERSION,'source_sha256':summary['source_sha256'],
                 'model':'Original PAT weights with adapted class-safe tiling; raw argmax and original 2D PAT cleanup. Not original stitching or FilFinder measurement.',
                 'mask_source':str(mask_folder.resolve()),'raw_mask_sha256':hashlib.sha256((mask_folder/'raw_classes.png').read_bytes()).hexdigest(),
                 'pixels_per_micrometre':summary['pixels_per_micrometre'],
                 'postprocessing_record':json.loads((folder/'PAT_cleanup/record.json').read_text()),
                 'scope':'Not anatomical periderm lengths. Raw and 2D-cleaned P-labelled path sums are not used by the main measurement table.',
                 'training_occurred':False,'rows':rows}
    write(folder/'diagnostics.json',diagnostics)
    with (folder/'model_label_diagnostics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    summary['model_diagnostics']=diagnostics
    write(output/'results.json',summary)
    report('Analysis complete. Main measurements and separate model-label diagnostics saved.')
    return summary
