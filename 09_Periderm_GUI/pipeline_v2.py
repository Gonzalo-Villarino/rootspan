"""Automatic image-to-results prototype. No ROI, mask or human T inputs.

V2: inspect the complete track, independently of the measuring endpoint.
The locked transition detector is unchanged. Foreground tracking and endpoint
estimation below are NEW experimental geometry, not learned biological models.
Connected groups and unsupported transitions are retained as failed rows.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import importlib.util
import json
import csv
import math
import hashlib
import traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from root_tracks import fill_small_holes, split_component, VERSION as TRACK_VERSION
VERSION = "automatic_pipeline_v2.0_context"
from image_quality import assess_quality

APP = Path(__file__).resolve().parent
PROJECT = APP.parent
BUNDLE = PROJECT / '06_Models/Locked_Transition_v1_20261116'
sys.path.insert(0, str(BUNDLE))
from run_checkpoint import predict
from verify_checkpoint import verify

spec = importlib.util.spec_from_file_location('existing_geometry', PROJECT / '07_Periderm_Length_Proof/measure_roots.py')
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)
Image.MAX_IMAGE_PIXELS = 200_000_000
DS = 4
BLUE = '#27b8ff'
ORANGE = '#ffa735'

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def arc(path):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]

def validate_scale(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0 or value > 10000:
        raise ValueError('Enter a positive image scale in pixels per micrometre.')
    return value

def connected_group_fraction(component):
    """Persistent parallel foreground intervals suggest connected main roots."""
    h, w = component.shape
    counts = []
    for x in np.linspace(int(w*.3), int(w*.7), 31).astype(int):
        column = component[:, max(0, min(w-1, x))]
        lab, n = ndi.label(column)
        lengths = np.bincount(lab)[1:]
        counts.append(int((lengths >= 4).sum()))
    return float(np.mean(np.array(counts) >= 2))

def detect_tracks(rgb):
    w, h = rgb.size
    a = np.array(rgb.getchannel('G'))
    padded = np.pad(a, ((0, (-h) % DS), (0, (-w) % DS)))
    small = np.array(Image.fromarray(padded).resize((padded.shape[1]//DS, padded.shape[0]//DS), Image.Resampling.BOX), dtype=float)
    fg = fill_small_holes(ndi.binary_closing(ndi.gaussian_filter(small, 1) > 16, iterations=2))
    fg = ndi.binary_opening(fg, iterations=1)
    labels, _ = ndi.label(fg)
    candidates = []
    for i, box in enumerate(ndi.find_objects(labels), 1):
        if box is None:
            continue
        width = (box[1].stop-box[1].start)*DS
        local = labels[box] == i
        if int(local.sum()) < 1000:
            continue
        if width < max(2000, .45*w):
            touches_edge=(box[0].start<=3 or box[1].start<=3 or
                          box[0].stop>=fg.shape[0]-3 or box[1].stop>=fg.shape[1]-3)
            if touches_edge and width>=max(1000,.15*w) and local.shape[1]>2*local.shape[0]:
                candidates.append((box,local,{'partial_at_image_edge':True,'split_applied':False}))
            continue
        candidates.extend(split_component(box, local))
    # A fixed source-image ordering, not a claim that detected count equals true count.
    candidates.sort(key=lambda c: (c[0][0].start+c[0][0].stop)/2)
    return candidates

def route_from_component(box, component):
    local = np.pad(component, 3)
    xy, detail = geometry.skeleton_path(local)
    offset = np.array([box[1].start-3, box[0].start-3], float)
    xy = (xy+offset)*DS + (DS-1)/2
    a = arc(xy)
    positions = np.r_[np.arange(0, a[-1], 4), a[-1]]
    path = np.c_[np.interp(positions, a, xy[:,0]), np.interp(positions, a, xy[:,1])]
    path = ndi.gaussian_filter1d(path, 5, axis=0, mode='nearest')
    path = path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-7]]
    radii = ndi.distance_transform_edt(local)
    q = (path-1.5)/DS-offset
    rad = ndi.map_coordinates(radii, [q[:,1],q[:,0]], order=1, mode='nearest')*DS
    a = arc(path)
    # Detect sustained widening near the right side. This is only a junction candidate.
    smooth = ndi.median_filter(rad, size=51, mode='nearest')
    baseline_values = smooth[(a>.45*a[-1]) & (a<.7*a[-1])]
    baseline = max(3., float(np.median(baseline_values)))
    wide = (smooth > 1.8*baseline) & (a>.72*a[-1])
    labs, n = ndi.label(wide)
    junction = None
    for k in range(1,n+1):
        ids = np.flatnonzero(labs==k)
        if a[ids[-1]]-a[ids[0]] >= 200 and a[-1]-a[ids[0]] >= 160:
            junction = int(ids[0])
            break
    if junction is not None:
        endpoint = 'width_based_junction_candidate'
    else:
        endpoint = 'visible_right_end_only'
    measure_index = junction if junction is not None else len(path)-1
    return path, {'measurement_path_index':measure_index,
                  'measurement_distance_pixels':float(a[measure_index]),
                  'inspection_distance_pixels':float(a[-1]),
                  'endpoint_method':endpoint, 'width_reference_radius_px':baseline,
                  'junction_multiplier':1.8, 'minimum_widened_span_px':200,
                  'anatomical_endpoint_verified':False, **detail}

def font(size):
    return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', size)

def plot_result(rgb, row, path, bounds, destination, max_width=1600):
    photo = rgb.crop(bounds)
    ratio = min(1, max_width/photo.width)
    photo = photo.resize((round(photo.width*ratio),round(photo.height*ratio)), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(photo)
    xy = [((x-bounds[0])*ratio,(y-bounds[1])*ratio) for x,y in path]
    d.line(xy, fill='white', width=1)
    if row.get('transition_xy') is not None:
        tx,ty = row['transition_xy'];x=(tx-bounds[0])*ratio
        d.line((x,0,x,photo.height), fill=BLUE, width=3)
        t=row['T_path_pixels'];a=arc(path)
        segment=[((tx-bounds[0])*ratio,(ty-bounds[1])*ratio)]+[xy[i] for i in np.flatnonzero(a>t)]
        if len(segment)>1 and row.get('length_pixels') is not None:
            d.line(segment, fill=BLUE, width=3)
    ex,ey=xy[-1];d.ellipse((ex-5,ey-5,ex+5,ey+5),fill=ORANGE)
    photo.save(destination)

def run(source, output, pixels_per_um, report=lambda text: None):
    pixels_per_um = validate_scale(pixels_per_um)
    source, output = Path(source), Path(output)
    output.mkdir(exist_ok=False)
    report('Checking the saved model and opening the image…')
    integrity = verify(BUNDLE)
    source_hash = sha(source)
    with Image.open(source) as opened:
        if getattr(opened,'n_frames',1) != 1:
            raise ValueError('Please use a single-plane image; stacks need a separate projection workflow.')
        if opened.width*opened.height > 200_000_000:
            raise ValueError('This prototype accepts images up to 200 million pixels.')
        if opened.mode not in ('RGB','RGBA'):
            raise ValueError('This V1 prototype expects RGB fluorescence images. Convert channels deliberately before analysis.')
        rgb = opened.convert('RGB')
    if rgb.width < 800 or rgb.height < 256:
        raise ValueError('The image is too small for this native-pixel model.')
    preview=rgb.copy();preview.thumbnail((1800,1200));preview.save(output/'source_preview.jpg',quality=92)
    report('Finding long root tracks automatically…')
    components=detect_tracks(rgb)
    if len(components)>30:
        raise ValueError('More than 30 long tracks detected; image layout needs inspection before this prototype can run.')
    rows=[]
    for i,(box,component,separation) in enumerate(components,1):
        row={'track_id':i,'periderm_length_um':None,'segment_to_visible_end_um':None,
             'length_pixels':None,'transition_xy':None,'endpoint_xy':None,
             'status':'needs_review','notes':[],'endpoint_method':None,
             'separation':separation,'detection_bbox_xyxy':
             [box[1].start*DS,box[0].start*DS,box[1].stop*DS,box[0].stop*DS]}
        fraction=connected_group_fraction(component)
        row['parallel_intervals_fraction']=fraction
        folder=output/f'track_{i:02d}';folder.mkdir()
        report(f'Analysing track {i} of {len(components)}…')
        if separation.get('partial_at_image_edge'):
            row['status']='partial_root'
            row['notes']=['Short root-like fragment touches the image edge. Full root and transition context unavailable; no length reported.']
            rows.append(row);write_json(folder/'result.json',row);continue
        if fraction>.45 or separation.get('separation_failed'):
            row['status']='connected_group'
            row['notes']=['Several parallel roots appear connected. No length reported for this group.']
            rows.append(row);write_json(folder/'result.json',row);continue
        try:
            path, endpoint=route_from_component(box,component)
            if np.min(path)<0 or path[:,0].max()>=rgb.width or path[:,1].max()>=rgb.height:
                raise ValueError('Track reaches outside the image.')
            a=arc(path)
            if a[-1]<1000:raise ValueError('Track too short for the fixed transition rule.')
            if np.sum(np.diff(path[:,0]) < -4) > .03*len(path):
                raise ValueError('Track doubles back; a main-root route cannot be established reliably.')
            bounds=[max(0,int(path[:,0].min())-200),max(0,int(path[:,1].min())-200),
                    min(rgb.width,int(path[:,0].max())+201),min(rgb.height,int(path[:,1].max())+201)]
            crop=rgb.crop(bounds);crop.save(folder/'image.png')
            route={'source_crop_xyxy':bounds,'source_size_xy':list(crop.size),
                   'path_xy_crop_pixels':(path-np.array(bounds[:2])).tolist(),
                   'distance_along_path_pixels':a.tolist(),'automatic_geometry':True,
                   'human_ROI_used':False,'manual_transition_used':False,
                   'endpoint':endpoint,'source_sha256':source_hash}
            write_json(folder/'route.json',route)
            # Call the exact locked image-to-T function, with unchanged weights/rule.
            prediction=predict(folder/'image.png',folder/'route.json')
            prediction['pipeline_version']=VERSION
            prediction['full_visible_track_context_used']=True
            write_json(folder/'prediction.json',prediction)
            full_path=path
            path=full_path[:endpoint['measurement_path_index']+1]
            a=arc(path)
            row['inspection_extension_pixels']=float(endpoint['inspection_distance_pixels']-a[-1])
            row.update({'endpoint_xy':path[-1].tolist(),'endpoint_method':endpoint['endpoint_method'],
                        'transition_xy':prediction['T_source_xy'],'T_path_pixels':prediction['T_path_pixels'],
                        'path_xy':path.tolist(),'image':f'track_{i:02d}/overlay.png'})
            if prediction['T_path_pixels'] is None:
                row['status']='no_transition'
                scores=np.asarray(prediction['presence_scores'])
                row['notes']=['No supported transition on the complete inspection path. No length reported.']
                if not (scores[:,1]>=.5).any():
                    row['notes'].append('No Periderm-like window detected. This does not establish a zero periderm length.')
            elif prediction['T_path_pixels']>a[-1]:
                row['status']='transition_beyond_endpoint'
                row['notes']=['A transition candidate lies beyond the estimated root endpoint. Root versus hypocotyl needs review; no periderm length reported.']
            else:
                length=float(a[-1]-prediction['T_path_pixels'])
                if length<0:raise ValueError('Transition is beyond the measuring endpoint.')
                row['length_pixels']=length
                if endpoint['endpoint_method']=='width_based_junction_candidate':
                    row['periderm_length_um']=length/pixels_per_um
                    row['status']='provisional'
                    row['notes']=['Automatic widening-based endpoint; anatomical junction and transition need validation.']
                else:
                    row['segment_to_visible_end_um']=length/pixels_per_um
                    row['status']='endpoint_uncertain'
                    row['notes']=['No root–hypocotyl junction identified. Segment length ends at the visible right edge, not a confirmed biological junction.']
            quality=assess_quality(rgb,path,prediction['T_path_pixels'] if row['status']!='transition_beyond_endpoint' else None)
            write_json(folder/'image_quality.json',quality)
            row['image_quality']={k:v for k,v in quality.items() if k!='windows'}
            if quality['withhold_measurement']:
                row['status']='low_image_quality'
                row['notes']=['Possible blur / low image detail. '+quality['reason']+' No measurement reported.']
                row['periderm_length_um']=None;row['segment_to_visible_end_um']=None;row['length_pixels']=None
            plot_result(rgb,row,path,bounds,folder/'overlay.png')
            if row['transition_xy'] is not None:
                tx,ty=row['transition_xy'];x0=int(np.clip(tx-700,0,rgb.width-1400));y0=int(np.clip(ty-280,0,rgb.height-560))
                plot_result(rgb,row,path,[x0,y0,x0+1400,y0+560],folder/'transition.png')
                row['closeup']=f'track_{i:02d}/transition.png'
        except Exception as error:
            row['status']='failed';row['notes']=[str(error)]
            row['periderm_length_um']=None;row['segment_to_visible_end_um']=None;row['length_pixels']=None
            (folder/'error.txt').write_text(traceback.format_exc())
        rows.append(row);write_json(folder/'result.json',row)
    clean=[{k:v for k,v in row.items() if k!='path_xy'} for row in rows]
    summary={'source_name':source.name,'source_sha256':source_hash,'source_dimensions_xy':list(rgb.size),
        'pixels_per_micrometre':pixels_per_um,'checkpoint':'Transition_v1_20261116',
        'geometry_version':TRACK_VERSION,'pipeline_version':VERSION,'rows':clean,
        'model_version':'V1 weights and rule; V2 full-track inspection pipeline',
        'detected_groups':len(rows),'individual_tracks':sum(r['status'] not in ('connected_group','failed','partial_root') for r in rows),
        'provisional_lengths':sum(r['periderm_length_um'] is not None for r in rows),
        'training_occurred':False,'human_paths_or_transition_points_used':False,
        'results_are_provisional':True,'scope':'Automatic processing of horizontal RGB fluorescence roots, hypocotyl to right. Track count is not a verified root count.',
        'integrity_before':integrity,'integrity_after':verify(BUNDLE)}
    write_json(output/'results.json',summary)
    fields=['track_id','periderm_length_um','segment_to_visible_end_um','status','endpoint_method','notes']
    with (output/'measurements.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in clean:
            r={k:row.get(k) for k in fields};r['notes']='; '.join(r['notes']);writer.writerow(r)
    # Full source preview with automatic track overlays; no invented tissue mask.
    preview=rgb.copy();preview.thumbnail((1800,1200));sx=preview.width/rgb.width;sy=preview.height/rgb.height
    draw=ImageDraw.Draw(preview)
    for row in rows:
        if 'path_xy' not in row:
            x0,y0,x1,y1=row['detection_bbox_xyxy']
            draw.rectangle((x0*sx,y0*sy,x1*sx,y1*sy),outline=ORANGE,width=2)
            draw.text((x0*sx+4,max(0,y0*sy)),str(row['track_id'])+' · flagged',
                      fill=ORANGE,font=font(20),stroke_width=2,stroke_fill='black')
            continue
        path=np.array(row['path_xy']);scaled=[(x*sx,y*sy) for x,y in path]
        draw.line(scaled,fill='white',width=1)
        xx,yy=scaled[len(scaled)//2];draw.text((xx,yy-24),str(row['track_id']),fill='white',font=font(24),stroke_width=2,stroke_fill='black')
        if row['transition_xy'] is not None:
            tx,ty=row['transition_xy'];draw.line((tx*sx,ty*sy-35,tx*sx,ty*sy+35),fill=BLUE,width=3)
        ex,ey=scaled[-1];draw.ellipse((ex-5,ey-5,ex+5,ey+5),fill=ORANGE)
    preview.save(output/'all_tracks.png')
    assert sha(source)==source_hash
    report('Analysis complete. Measurements and review flags are saved.')
    return summary
