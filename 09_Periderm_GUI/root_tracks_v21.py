"""Separate horizontal root lanes without changing the tissue model.

Only small holes are filled; enclosing two roots must not fill the space between
them. Persistent parallel runs seed background seams that split touching groups.
All parameters are image-independent. No expected count or hand coordinates.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

VERSION = 'automatic_geometry_candidate_0.2.2_tail_guidance'

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

def background_seam(mask, seed_x, seed_y, gap_anchors=(), intensity=None):
    """A monotone seam avoids foreground, preferring smooth paths in empty space."""
    height, width = mask.shape
    clearance = ndi.distance_transform_edt(~mask)
    tissue_cost = 100*mask if intensity is None else mask*(5+20*(np.asarray(intensity,float)/32)**2)
    cost = .05 + np.exp(-clearance / 8) + tissue_cost
    # Repeated column observations keep a seam BETWEEN the same ordered roots;
    # a globally short seam could otherwise cut a curved root and escape outside.
    for x, lower, upper in gap_anchors:
        cost[:lower+1, x] += 1_000_000
        cost[upper:, x] += 1_000_000
    # The old unconstrained far-right endpoint could escape across a root and
    # give its hypocotyl to the adjacent lane. Extrapolate the observed gap,
    # independently of tissue scores or requested measurements.
    if len(gap_anchors)>=6:
        tail=sorted(gap_anchors)[-9:]
        xx=np.array([r[0] for r in tail],float)
        mid=np.array([(r[1]+r[2])/2 for r in tail],float)
        slopes=[(mid[j]-mid[i])/(xx[j]-xx[i]) for i in range(len(xx)) for j in range(i+1,len(xx)) if xx[j]>xx[i]]
        slope=float(np.clip(np.median(slopes),-.8,.8))
        intercept=float(np.median(mid-slope*xx))
        predicted=float(np.clip(intercept+slope*(width-1),0,height-1))
        half=max(8.,float(np.median([r[2]-r[1] for r in tail]))/2)
        yy=np.arange(height)
        cost[np.abs(yy-predicted)>half,-1]+=1_000_000
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

def tracked_gap_anchors(mask, seed_x, seed_runs):
    """Follow root identities; an artifact cannot replace a missing root by rank."""
    count=len(seed_runs);anchors=[[] for _ in range(count-1)]
    for direction in [-1,1]:
        centres=np.array([(a+b)/2 for a,b in seed_runs],float)
        widths=np.array([b-a+1 for a,b in seed_runs],float)
        velocity=np.zeros(count);previous_x=seed_x
        columns=list(range(seed_x,mask.shape[1] if direction==1 else -1,direction*24))
        for xx in columns:
            rr=column_runs(mask,xx);step=xx-previous_x
            expected=centres+velocity*step
            matched={}
            if rr:
                cc=np.array([(a+b)/2 for a,b in rr]);ww=np.array([b-a+1 for a,b in rr])
                cost=np.abs(expected[:,None]-cc[None,:])/np.maximum(8,widths[:,None])
                cost+=.7*np.abs(np.log(ww[None,:]/np.maximum(widths[:,None],1)))
                for i,j in zip(*linear_sum_assignment(cost)):
                    # A merged cross section must not reset either root centre.
                    covers=sum(rr[j][0]-4<=c<=rr[j][1]+4 for c in expected)
                    gap=abs(expected[i]-cc[j])
                    if covers>1 or gap>max(30,abs(step)*1.2) or ww[j]>max(60,6*widths[i]):continue
                    matched[i]=rr[j]
                    if step:
                        observed=(cc[j]-centres[i])/step
                        velocity[i]=np.clip(.7*velocity[i]+.3*observed,-1.5,1.5)
                    expected[i]=cc[j];widths[i]=.85*widths[i]+.15*ww[j]
            centres=expected;previous_x=xx
            for i in range(count-1):
                if i in matched and i+1 in matched:
                    a,b=matched[i],matched[i+1]
                    if b[0]-a[1]>=4:anchors[i].append((xx,a[1],b[0]))
    return [sorted(set(a)) for a in anchors]

def split_component(box, mask, intensity=None):
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
    anchors_by_gap = tracked_gap_anchors(mask,x,runs)
    # Preserve established distal separation. New identity tracking addresses
    # the proximal ends, where roots disappear or widen into shared groups.
    prior=[(int(xx),column_runs(mask,int(xx))) for xx in np.linspace(.08*width,x,61).astype(int)]
    for i in range(count-1):
        anchors_by_gap[i]=[(xx,a,b) for xx,a,b in anchors_by_gap[i] if xx>=x]
        anchors_by_gap[i]+=[(xx,rr[i][1],rr[i+1][0]) for xx,rr in prior if len(rr)==count and rr[i+1][0]-rr[i][1]>=4]
    seams = []
    for i,(a,b) in enumerate(zip(runs,runs[1:])):
        seams.append(background_seam(mask,x,(a[1]+b[0])//2,anchors_by_gap[i],intensity))
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
