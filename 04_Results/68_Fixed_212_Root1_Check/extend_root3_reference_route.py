"""Extend a human-guided reference route to the confirmed left cut end.

Uses fluorescence geometry between human-selected landmarks, not ML inference.
Retains the previously reviewed T-J route exactly. New L-T section needs review.
Native-source coordinates are retained; display resizes are not measurement data.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / '02_Data/09_Transition_Endpoint_Review'
OUT = REVIEW / 'route_proposal_02'
Image.MAX_IMAGE_PIXELS = 300_000_000


def graph_route(component, start, stop):
    sk = skeletonize(component)
    yx = np.argwhere(sk)
    indices = np.full(sk.shape, -1, dtype=np.int32)
    indices[sk] = np.arange(len(yx))
    rows, cols, data = [], [], []
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        yy, xx = yx[:, 0] + dy, yx[:, 1] + dx
        valid = (yy >= 0) & (yy < sk.shape[0]) & (xx >= 0) & (xx < sk.shape[1])
        a = np.flatnonzero(valid)
        b = indices[yy[valid], xx[valid]]
        keep = b >= 0
        a, b = a[keep], b[keep]
        rows.extend([a, b]); cols.extend([b, a])
        data.extend([np.full(len(a), math.hypot(dx, dy))] * 2)
    graph = csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(len(yx), len(yx)))
    xy = yx[:, ::-1].astype(float)
    left = int(np.argmin(np.linalg.norm(xy - start, axis=1)))
    right = int(np.argmin(np.linalg.norm(xy - stop, axis=1)))
    distances, previous = dijkstra(graph, directed=False, indices=left, return_predecessors=True)
    if not np.isfinite(distances[right]):
        raise ValueError('No continuous source-fluorescence route between the reviewed landmarks.')
    path = [right]
    while path[-1] != left:
        node = int(previous[path[-1]])
        if node < 0 or len(path) > len(yx):
            raise ValueError('Invalid skeleton route.')
        path.append(node)
    return xy[path[::-1]], [float(np.linalg.norm(xy[left]-start)), float(np.linalg.norm(xy[right]-stop))]


def main():
    if OUT.exists():
        raise FileExistsError('Preserve earlier review: output folder already exists.')
    meta = json.loads((REVIEW / 'COL0_root3_review_details.json').read_text())
    old = json.loads((REVIEW / 'route_proposal_01/reference_route_record.json').read_text())
    left_review = json.loads((REVIEW / 'COL0_root3_left_end_review_02.json').read_text())
    calibration = json.loads((REVIEW / 'COL0_root3_calibration_review_02.json').read_text())
    assert left_review['endpoint_confirmed'] and calibration['confirmed_for_COL0']
    source = Path(meta['source'])
    with source.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == meta['source_sha256']
    im = Image.open(source).convert('RGB')
    approved = np.array(old['path_xy_source_pixels'], dtype=float)
    L = np.array(left_review['working_left_cut_end_source_xy'], dtype=float)
    T, J = approved[0], approved[-1]
    box = np.array([1700, 2780, 12900, 6600])
    downsample = 4
    green = np.asarray(im.crop(box.tolist()).getchannel('G').resize(
        (int((box[2]-box[0])//downsample), int((box[3]-box[1])//downsample)),
        Image.Resampling.BOX), dtype=float)
    # BOX-resampled pixel centres are offset by 1.5 original pixels.
    offset = box[:2] + (downsample-1)/2
    start, stop = (L-offset)/downsample, (T-offset)/downsample
    smooth = ndi.gaussian_filter(green, 1)
    attempts = []
    selected = None
    for threshold, closing in [(16, 2), (12, 2), (8, 2), (5, 2), (3, 2), (3, 3)]:
        foreground = ndi.binary_closing(smooth > threshold, iterations=closing)
        foreground = ndi.binary_fill_holes(foreground)
        labels, count = ndi.label(foreground)
        points = []
        for p in [start, stop]:
            px, py = np.rint(p).astype(int)
            local = foreground[max(0,py-25):py+26, max(0,px-25):px+26]
            yx = np.argwhere(local) + [max(0,py-25), max(0,px-25)]
            if not len(yx):
                points.append(None)
            else:
                y, x = yx[np.argmin(np.linalg.norm(yx[:, ::-1]-p, axis=1))]
                points.append(int(labels[y,x]))
        attempts.append({'threshold':threshold, 'closing_iterations':closing, 'endpoint_components':points})
        if points[0] is None or points[0] != points[1]:
            continue
        component = labels == points[0]
        candidate, endpoint_distances = graph_route(component, start, stop)
        if max(endpoint_distances)*downsample > 100:
            continue
        selected = (candidate*downsample+offset, threshold, closing, endpoint_distances)
        break
    if selected is None:
        raise ValueError(f'No satisfactory connected route: {attempts}')
    raw, threshold, closing, distances = selected
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
    positions = np.r_[np.arange(0, arc[-1], 4.), arc[-1]]
    path = np.column_stack([np.interp(positions, arc, raw[:,j]) for j in [0,1]])
    path = ndi.gaussian_filter1d(path, 5, axis=0, mode='nearest')
    # Bridge the skeleton's rounded ends to the user's working landmarks.
    extension = np.vstack([L, path, T])
    full = np.vstack([extension[:-1], approved])
    assert np.array_equal(full[-len(approved):], approved)
    assert np.array_equal(full[0], L) and np.array_equal(full[-1], J)
    assert np.isfinite(full).all()
    if np.max(np.linalg.norm(np.diff(full,axis=0),axis=1)) > 100:
        raise ValueError('Large route gap; inspect before creating a review figure.')
    OUT.mkdir()
    font = lambda n: ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', n)

    def panel(bounds, width, line=True, scale_bar=False):
        photo = im.crop(bounds)
        s = width/photo.width
        photo = photo.resize((width, round(photo.height*s)), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(photo)
        def point(p): return ((p[0]-bounds[0])*s, (p[1]-bounds[1])*s)
        if line:
            draw.line([point(p) for p in full], fill='#00e3ff', width=2)
        for label, p in [('L',L), ('T',T), ('J',J)]:
            x,y = point(p)
            if 0 <= x < photo.width and 0 <= y < photo.height:
                draw.ellipse((x-6,y-6,x+6,y+6),fill='#ff4fc1')
                draw.text((x+10,y-35),label,font=font(26),fill='white',stroke_width=2,stroke_fill='black')
        if scale_bar:
            bar = 1000*calibration['pixels_per_micrometre']*s
            x,y = 20, photo.height-25
            draw.line((x,y,x+bar,y),fill='white',width=4)
            draw.text((x,y-30),'1,000 µm',font=font(22),fill='white',stroke_width=1,stroke_fill='black')
        return photo

    whole_box = [1700,2780,20100,6500]
    original = panel(whole_box,1800,line=False)
    traced = panel(whole_box,1800,line=True)
    closeups = [panel([1660,2810,2660,3310],580,scale_bar=True),
                panel([12120,5800,13120,6300],580,scale_bar=True),
                panel([19000,4240,20000,4740],580,scale_bar=True)]
    height = 120+original.height+55+traced.height+70+closeups[0].height+85
    canvas = Image.new('RGB',(1840,height),'white')
    d=ImageDraw.Draw(canvas)
    d.text((20,12),'COL0 root 3 — check the complete measuring line',font=font(32),fill='black')
    d.text((20,58),'Your selected landmarks + computer tracing. This is a reference measurement, not a new model prediction.',font=font(24),fill='#273444')
    d.text((20,94),'Original image with landmark markers',font=font(22),fill='black')
    canvas.paste(original,(20,120))
    y=120+original.height+20
    d.text((20,y),'Cyan line: main root only, from L through T to J',font=font(24),fill='black')
    y+=35;canvas.paste(traced,(20,y));y+=traced.height+28
    for x,label,photo in zip([20,630,1240],['L: left cut end','T: working transition','J: root–hypocotyl junction'],closeups):
        d.text((x,y),label,font=font(24),fill='black');canvas.paste(photo,(x,y+36))
    y+=36+closeups[0].height+18
    d.text((20,y),'Left of T: Endoderm. T to J: Periderm including transition, using your agreed measurement convention.',font=font(23),fill='black')
    d.text((20,y+32),'Scale: 0.5299 pixels/µm, confirmed by you. New L–T line awaits your review; no final lengths reported.',font=font(23),fill='black')
    figure=OUT/'COL0_root3_complete_route_to_review.png'
    canvas.save(figure)
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'source':str(source),
        'source_sha256':meta['source_sha256'],'method':'Source-green-channel thresholding, closing, hole filling, skeleton shortest path between human-selected L and T; smoothed route joined to the previously approved T-J route unchanged.',
        'not_a_model_prediction':True,'not_an_independent_manual_trace':True,
        'previous_route':str(REVIEW/'route_proposal_01/reference_route_record.json'),
        'source_crop_xyxy':box.tolist(),'topology_downsample':downsample,
        'source_coordinate_offset':offset.tolist(),'green_threshold':threshold,
        'closing_iterations_downsampled':closing,'gaussian_sigma_downsampled':1,
        'candidate_attempts':attempts,'skeleton_endpoint_distances_native_pixels':(np.array(distances)*downsample).tolist(),
        'smoothing':'Resample skeleton every 4 native pixels; Gaussian sigma 5 samples; explicit bridges to working L and T.',
        'working_left_cut_end_xy':L.tolist(),'working_transition_xy':T.tolist(),'working_junction_xy':J.tolist(),
        'path_xy_source_pixels':full.tolist(),'transition_path_index':len(extension)-1,
        'previous_T_J_route_unchanged':True,'new_L_T_route_biological_review_pending':True,
        'source_image_unchanged':True,'calibration_record':str(REVIEW/'COL0_root3_calibration_review_02.json'),
        'pixels_per_micrometre':calibration['pixels_per_micrometre'],
        'length_reported':False,'new_training':False,'figure':str(figure)}
    (OUT/'reference_route_record.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    (OUT/Path(__file__).name).write_text(Path(__file__).read_text())
    print(json.dumps({'figure':str(figure),'threshold':threshold,'closing':closing,'endpoint_distances_px':record['skeleton_endpoint_distances_native_pixels'],'path_points':len(full)},indent=2))


if __name__ == '__main__':
    main()
