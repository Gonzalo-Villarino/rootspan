"""Propose an annotation-assisted measurement route for expert review.

This is NOT model inference, a new segmentation, or a finalized measurement.
The expert selected the two endpoint regions; an archived outline guides the
main-root route and source fluorescence extends it towards the new transition.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '07_Periderm_Length_Proof'))
from measure_roots import skeleton_path

Image.MAX_IMAGE_PIXELS = 300_000_000
REVIEW = ROOT / '02_Data/09_Transition_Endpoint_Review'
SOURCE = ROOT / '02_Data/07_Additional_Archived_Annotations_20261116'
OUT = REVIEW / 'route_proposal_01'
OUT.mkdir(exist_ok=False)
meta = json.loads((REVIEW / 'COL0_root3_review_details.json').read_text())
src = Path(meta['source'])
with src.open('rb') as f:
    assert hashlib.file_digest(f, 'sha256').hexdigest() == meta['source_sha256']
im = Image.open(src).convert('RGB')
ann = json.loads((SOURCE / 'COL0_Slide1.json').read_text())
shape = sorted(ann['shapes'], key=lambda s: np.mean(np.array(s['points'])[:, 1]))[2]
pts = np.array(shape['points'])
origin = np.floor(pts.min(0) - 8).astype(int)
end = np.ceil(pts.max(0) + 8).astype(int)
poly = Image.new('L', tuple((end - origin).tolist()), 0)
ImageDraw.Draw(poly).polygon([tuple(p - origin) for p in pts], fill=1)
component = np.pad(np.array(poly)[::4, ::4].astype(bool), 3)
archived_path, _ = skeleton_path(component)
archived_path = (archived_path - 3) * 4 + origin
order = np.argsort(archived_path[:, 0])
archived_path = archived_path[order]

# Extend the route towards the newly identified transition without using a model.
box = [12350, 5750, 13550, 6400]
green = np.array(im.crop(box))[:, :, 1].astype('float32')
green = ndi.gaussian_filter(green, [4, 7])
anchor_x = 13400
cy = float(np.interp(anchor_x, archived_path[:, 0], archived_path[:, 1]))
extension = []
for x in range(anchor_x, 12471, -8):
    profile = np.median(green[:, x-box[0]-10:x-box[0]+11], axis=1)
    lo = max(0, round(cy-box[1])-105)
    hi = min(len(profile), round(cy-box[1])+106)
    threshold = max(3., .17 * float(profile[lo:hi].max()))
    foreground = ndi.binary_closing(profile[lo:hi] > threshold, iterations=14)
    labels, n = ndi.label(foreground)
    candidates = []
    for k in range(1, n+1):
        rows = np.where(labels == k)[0] + lo
        if len(rows) < 14:
            continue
        centre = (rows[0]+rows[-1])/2 + box[1]
        candidates.append((abs(centre-cy), centre))
    if candidates:
        detected = min(candidates)[1]
        cy = .6*detected + .4*cy
    extension.append([x, cy])
extension = np.array(extension[::-1])
xs = np.arange(12480, 19601, 4.)
ys = np.where(xs < anchor_x,
              np.interp(xs, extension[:, 0], extension[:, 1]),
              np.interp(xs, archived_path[:, 0], archived_path[:, 1]))
ys = ndi.gaussian_filter1d(ys, 8)
transition_x = 12621.
junction_x = 19500.  # Explicit preview midpoint of expert's interval C-D.
route_x = np.r_[transition_x, xs[(xs > transition_x) & (xs < junction_x)], junction_x]
route = np.column_stack([route_x, np.interp(route_x, xs, ys)])
assert np.isfinite(route).all() and np.all(np.diff(route[:, 0]) > 0)
T, J = route[0], route[-1]
font = lambda n: ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', n)

def draw_panel(box, width, show_zone=False):
    photo = im.crop(box)
    scale = width / photo.width
    photo = photo.resize((width, round(photo.height*scale)), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(photo)
    def point(p):
        return ((p[0]-box[0])*scale, (p[1]-box[1])*scale)
    d.line([point(p) for p in route], fill='#00e3ff', width=3)
    for label, p in [('T', T), ('J', J)]:
        x, y = point(p)
        if -10 <= x <= photo.width+10 and -10 <= y <= photo.height+10:
            d.ellipse((x-7, y-7, x+7, y+7), fill='#ff4fc1')
            d.text((x+11, y-31), label, fill='white', font=font(26), stroke_width=2, stroke_fill='black')
    if show_zone:
        for bounds in [(12521, 12721), (19400, 19600)]:
            for bx in bounds:
                by = float(np.interp(bx, xs, ys))
                x, y = point([bx, by])
                if 0 <= x < photo.width:
                    d.line((x, y-85*scale, x, y+85*scale), fill='#ffc85a', width=2)
    return photo

whole = draw_panel([12400, 4180, 20100, 6340], 1660)
left = draw_panel([12350, 5810, 13150, 6260], 800, True)
right = draw_panel([19100, 4250, 19900, 4700], 800, True)
can = Image.new('RGB', (1700, 100+whole.height+60+left.height+100), 'white')
d = ImageDraw.Draw(can)
d.text((20, 12), 'COL0 root 3 — proposed reference route for your review', font=font(30), fill='black')
d.text((20, 55), 'Human-selected endpoint regions. Computer-proposed centreline. NOT a model prediction.', font=font(24), fill='black')
can.paste(whole, (20, 100))
yy = 100+whole.height+17
d.text((20, yy), 'T: working transition at marker 2', font=font(25), fill='black')
d.text((880, yy), 'J: junction preview at the middle of C-D', font=font(25), fill='black')
can.paste(left, (20, yy+40)); can.paste(right, (880, yy+40))
d.text((20, can.height-60), 'Cyan line = proposed route along the main root; pink dots = working ends.', font=font(25), fill='black')
d.text((20, can.height-29), 'Orange marks show the reviewed endpoint ranges. No final length or new training yet.', font=font(23), fill='black')
can.save(OUT/'COL0_root3_proposed_route.png')
record = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'source': str(src), 'source_sha256': meta['source_sha256'],
    'role': 'Human-guided reference route awaiting review; not a model prediction or independent manual trace.',
    'route_method': 'Smoothed centreline from archived Periderm outline, extended left with local fluorescence-band tracking. User-defined endpoint regions trim the route.',
    'archived_annotation': str(SOURCE/'COL0_Slide1.json'),
    'working_transition_xy': T.tolist(),
    'working_junction_xy': J.tolist(),
    'junction_point_selection': 'Midpoint x19500 of the expert interval C-D x19400-19600, for preview only; exact point not user-confirmed.',
    'transition_source_x_review_range': [12521, 12721],
    'junction_source_x_review_range': [19400, 19600],
    'path_xy_source_pixels': route.tolist(),
    'route_biological_review_pending': True,
    'length_reported': False, 'new_training': False,
    'calibration': 'Not confirmed for COL0; do not borrow calibration from 20_29 or 2_12.',
}
(OUT/'reference_route_record.json').write_text(json.dumps(record, indent=2)+'\n')
print(OUT/'COL0_root3_proposed_route.png')
