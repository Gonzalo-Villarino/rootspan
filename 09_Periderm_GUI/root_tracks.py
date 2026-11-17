"""Separate horizontal root lanes without changing the tissue model.

Only small holes are filled; enclosing two roots must not fill the space between
them. Persistent parallel runs seed background seams that split touching groups.
All parameters are image-independent. No expected count or hand coordinates.
"""
import numpy as np
from scipy import ndimage as ndi

VERSION = 'automatic_geometry_prototype_0.2.1'

def fill_small_holes(mask, maximum=255):
    labels, _ = ndi.label(~mask)
    sizes = np.bincount(labels.ravel())
    fill = sizes <= maximum
    fill[np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])] = False
    fill[0] = False
    return mask | fill[labels]

def column_runs(mask, x, minimum=3):
    labels, count = ndi.label(mask[:, x])
    runs = []
    for i in range(1, count + 1):
        yy = np.flatnonzero(labels == i)
        if len(yy) >= minimum:
            runs.append((int(yy[0]), int(yy[-1])))
    return runs

def background_seam(mask, seed_x, seed_y, gap_anchors=()):
    """A monotone seam avoids foreground, preferring smooth paths in empty space."""
    height, width = mask.shape
    clearance = ndi.distance_transform_edt(~mask)
    cost = .05 + np.exp(-clearance / 8) + 100 * mask
    # Repeated column observations keep a seam BETWEEN the same ordered roots;
    # a globally short seam could otherwise cut a curved root and escape outside.
    for x, lower, upper in gap_anchors:
        cost[:lower+1, x] += 1_000_000
        cost[upper:, x] += 1_000_000
    result = np.full(width, seed_y, dtype=int)
    shifts = np.arange(-4, 5)
    for direction in (-1, 1):
        columns = list(range(seed_x + direction, width if direction == 1 else -1, direction))
        previous = np.full(height, np.inf)
        previous[seed_y] = 0
        parents = np.zeros((len(columns), height), dtype=np.int8)
        for index, x in enumerate(columns):
            options = np.full((len(shifts), height), np.inf)
            for k, shift in enumerate(shifts):
                a, b = max(0, -int(shift)), min(height, height-int(shift))
                options[k, a:b] = previous[a+shift:b+shift] + .12*shift*shift
            choice = options.argmin(axis=0)
            previous = options[choice, np.arange(height)] + cost[:, x]
            parents[index] = shifts[choice]
        y = int(previous.argmin())
        for index in range(len(columns)-1, -1, -1):
            result[columns[index]] = y
            y += int(parents[index, y])
    return result

def split_component(box, mask):
    """Return lanes plus an explicit record of the automatic separation."""
    height, width = mask.shape
    samples = np.linspace(int(.3*width), int(.7*width), 31).astype(int)
    observations = [(int(x), column_runs(mask, int(x))) for x in samples]
    values, frequencies = np.unique([len(runs) for _, runs in observations], return_counts=True)
    count = int(values[frequencies.argmax()])
    support = float(frequencies.max()/len(samples))
    metadata = {'parallel_lane_count': count, 'lane_count_support': support,
                'split_method': 'background_seams', 'source_group_bbox_coarse':
                [box[1].start,box[0].start,box[1].stop,box[0].stop]}
    if count <= 1 or support < .45 or count > 12:
        return [(box, mask, {**metadata, 'split_applied': False})]
    candidates = [(x, runs) for x, runs in observations if len(runs) == count]
    # Prefer a clean middle column with large gaps, never a hand-picked position.
    def seed_score(item):
        x, runs = item
        gaps = [b[0]-a[1] for a,b in zip(runs, runs[1:])]
        return min(gaps)-.1*abs(x-width/2)
    x, runs = max(candidates, key=seed_score)
    anchor_observations = [(int(xx), column_runs(mask, int(xx))) for xx in
                           np.linspace(int(.08*width), int(.92*width), 121).astype(int)]
    anchor_observations = [(xx, rr) for xx,rr in anchor_observations if len(rr)==count]
    seams = []
    for i,(a,b) in enumerate(zip(runs,runs[1:])):
        anchors=[(xx,rr[i][1],rr[i+1][0]) for xx,rr in anchor_observations
                 if rr[i+1][0]-rr[i][1]>=4]
        seams.append(background_seam(mask,x,(a[1]+b[0])//2,anchors))
    # Ordered seams partition the image exactly even where roots touch.
    seams = np.sort(np.array(seams), axis=0)
    yy = np.arange(height)[:, None]
    lanes = []
    for i in range(count):
        region = mask.copy()
        if i: region &= yy > seams[i-1][None, :]
        if i < len(seams): region &= yy <= seams[i][None, :]
        labels, n = ndi.label(region)
        seed_y = (runs[i][0]+runs[i][1])//2
        label = int(labels[seed_y, x])
        if not label:
            continue
        lane = labels == label
        ys, xs = np.nonzero(lane)
        if len(xs) < 1000 or xs.max()-xs.min() < .45*width:
            continue
        y0,y1,x0,x1=int(ys.min()),int(ys.max()+1),int(xs.min()),int(xs.max()+1)
        local_box = (slice(box[0].start+y0,box[0].start+y1),slice(box[1].start+x0,box[1].start+x1))
        lanes.append((local_box,lane[y0:y1,x0:x1],{**metadata,'split_applied':True,'lane_in_group':i+1}))
    # Do not quietly lose a lane when separation fails; retain a flagged group.
    if len(lanes) != count:
        return [(box,mask,{**metadata,'split_applied':False,'separation_failed':True})]
    return lanes
