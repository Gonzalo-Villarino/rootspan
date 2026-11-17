"""Provisional proximal endpoint from a sustained fluorescence boundary.

This is an image heuristic, not a trained hypocotyl classifier. It does not read
T, a manual coordinate, or a desired length. Observed lane samples only: no extrapolation past the trace. This still does
not establish an anatomical junction and supplies review candidates only. Lack of support leaves V2's flag intact.
"""
import numpy as np
from scipy import ndimage as ndi

def proximal_endpoint(green,path,arc,previous,lane,box,downsample=4):
    lane_bytes=np.asarray(lane,dtype=np.uint8)
    distance=np.arange(max(80.,.6*arc[-1]),arc[-1],16.)
    centres=np.c_[np.interp(distance,arc,path[:,0]),np.interp(distance,arc,path[:,1])]
    before=np.c_[np.interp(distance-64,arc,path[:,0]),np.interp(distance-64,arc,path[:,1])]
    after=np.c_[np.interp(distance+64,arc,path[:,0]),np.interp(distance+64,arc,path[:,1])]
    tangent=after-before
    tangent/=np.maximum(1e-8,np.linalg.norm(tangent,axis=1))[:,None]
    uu,vv=np.meshgrid(np.arange(-24,25,4),np.arange(-48,49,4))
    peak=[];inside=[]
    for c,v in zip(centres,tangent):
        x=c[0]+uu*v[0]-vv*v[1];y=c[1]+uu*v[1]+vv*v[0]
        valid=(x>=0)&(y>=0)&(x<green.shape[1])&(y<green.shape[0])
        a=ndi.map_coordinates(green,[y,x],order=1,mode='nearest')
        # Exclude fluorescence belonging to a neighbouring lane.
        lane_y=(y-(downsample-1)/2)/downsample-box[0].start
        lane_x=(x-(downsample-1)/2)/downsample-box[1].start
        allowed=ndi.map_coordinates(lane_bytes,[lane_y,lane_x],order=0,mode='constant',cval=0)>0
        valid &= allowed
        peak.append(float(np.percentile(a[valid],90)) if valid.sum()>=5 else 0.)
        inside.append(float(valid.sum()>=5))
    peak=ndi.median_filter(peak,size=7,mode='nearest')
    candidates=[]
    lower=.72*arc[-1]
    if previous['endpoint_method']=='width_based_junction_candidate':
        lower=max(lower,previous['measurement_distance_pixels']-128)
    for i,s in enumerate(distance):
        if s<lower or s+384+24>arc[-1]:continue
        a=(distance>=s-384)&(distance<=s-64)
        b=(distance>=s+64)&(distance<=s+384)
        if a.sum()<12 or b.sum()<12:continue
        if min(np.asarray(inside)[a|b])<.9:continue
        pre=float(np.median(peak[a]));post=float(np.median(peak[b]))
        if pre>=15 and post<=.65*pre and pre-post>=8:
            candidates.append({'distance_pixels':float(s),'before_signal':pre,
                               'after_signal':post,'signal_ratio':post/pre})
    # The first sustained proximal signal loss is the boundary candidate.
    # Do not continue to the darkest/furthest location in the hypocotyl.
    selected=candidates[0] if candidates else None
    result=dict(previous)
    audit={'method':'observed_lane_fluorescence_candidate_v0.2','selected':selected,
           'search_minimum_distance_pixels':float(lower),
           'distance_pixels':distance.tolist(),'core_signal':list(map(float,peak)),
           'manual_reference_used':False,'transition_used':False,
           'anatomy_validated':False,'observed_lane_samples_only':True,
           'full_observed_after_window_required':True}
    if selected is not None:
        index=int(np.argmin(abs(arc-selected['distance_pixels'])))
        result.update(measurement_path_index=index,measurement_distance_pixels=float(arc[index]),
                      endpoint_method='fluorescence_boundary_candidate',
                      anatomical_endpoint_verified=False)
    return result,audit
