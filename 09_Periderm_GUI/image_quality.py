"""Conservative low-detail guard; not a trained or validated blur classifier.

Native-scale windows measure fine structure relative to image contrast. The
fixed threshold is a severe-detail-loss screen checked with synthetic blur;
passing it does not establish biological image adequacy or model accuracy.
"""
import numpy as np
from scipy import ndimage as ndi

VERSION = 'low_detail_screen_0.1'
DETAIL_LIMIT = .001
CONTRAST_LIMIT = 4.0

def window_detail(green):
    green = np.asarray(green, dtype=float)
    smooth = ndi.gaussian_filter(green, 1)
    broad = ndi.gaussian_filter(green, 3)
    ratio = float(np.mean((smooth-broad)**2)/(np.var(smooth)+1e-8))
    contrast = float(np.percentile(green, 95)-np.percentile(green, 5))
    return ratio, contrast

def assess_quality(rgb, path, transition_distance=None):
    green = np.asarray(rgb.getchannel('G'),dtype=float)
    arc = np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    distances = np.arange(160,arc[-1]-160,256.)
    if transition_distance is not None:
        distances=np.unique(np.r_[distances,transition_distance])
    u,v=np.meshgrid(np.arange(320)-159.5,np.arange(96)-47.5)
    records=[]
    for distance in distances:
        centre=np.array([np.interp(distance,arc,path[:,i]) for i in range(2)])
        before=np.array([np.interp(max(0,distance-80),arc,path[:,i]) for i in range(2)])
        after=np.array([np.interp(min(arc[-1],distance+80),arc,path[:,i]) for i in range(2)])
        tangent=after-before;tangent/=max(1e-8,np.linalg.norm(tangent))
        xx=centre[0]+u*tangent[0]-v*tangent[1]
        yy=centre[1]+u*tangent[1]+v*tangent[0]
        valid=(xx>=0)&(yy>=0)&(xx<green.shape[1])&(yy<green.shape[0])
        if valid.mean()<.95:continue
        patch=ndi.map_coordinates(green,[yy,xx],order=1,mode='nearest')
        detail,contrast=window_detail(patch)
        records.append({'distance_pixels':float(distance),'detail_ratio':detail,
                        'contrast_95_5':contrast,'low_detail':bool(detail<DETAIL_LIMIT or contrast<CONTRAST_LIMIT)})
    all_bad=float(np.mean([r['low_detail'] for r in records])) if records else None
    nearby=[] if transition_distance is None else [r for r in records if abs(r['distance_pixels']-transition_distance)<=512]
    near_bad=float(np.mean([r['low_detail'] for r in nearby])) if nearby else None
    flag=not records or (all_bad is not None and all_bad>=.6) or (len(nearby)>=3 and near_bad>=.6)
    reason=('Insufficient image detail to assess quality.' if not records else
            'Low contrast or severe loss of fine detail around predicted T.' if len(nearby)>=3 and near_bad>=.6 else
            'Low contrast or severe loss of fine detail along most of the root.' if flag else
            'No severe low-detail flag; image adequacy is not established by this check.')
    return {'version':VERSION,'status':'possible_blur_or_low_detail' if flag else 'no_low_detail_flag',
            'withhold_measurement':bool(flag),'reason':reason,'detail_limit':DETAIL_LIMIT,
            'contrast_limit':CONTRAST_LIMIT,'low_detail_fraction':all_bad,
            'near_transition_low_detail_fraction':near_bad,'windows':records,
            'validated_blur_classifier':False}
