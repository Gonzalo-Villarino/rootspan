"""Explicit V2.1 short-transition fallback; locked V1 weights stay unchanged.

Only consulted when V1 abstains. Two consecutive mixed windows need immediately
adjacent runs of at least three clear E and three clear P windows. These are
overlapping fields, not independent evidence. The result is a review candidate.
"""
import copy
import numpy as np

def with_short_transition(prediction, path, distance):
    result=copy.deepcopy(prediction)
    result['transition_method']='locked_v1_rule'
    result['brief_transition_review_required']=False
    if prediction['T_path_pixels'] is not None:
        return result
    scores=np.asarray(prediction['presence_scores'],float)
    locations=np.asarray(prediction['distance_along_local_path'],float)
    if scores.shape!=(len(locations),2) or not np.isfinite(scores).all():
        raise ValueError('Invalid appearance scores for transition review.')
    e=scores[:,0]>=.5;p=scores[:,1]>=.5
    states=np.full(len(scores),3);states[e&~p]=0;states[e&p]=1;states[~e&p]=2
    cuts=np.r_[0,np.flatnonzero(np.diff(states))+1,len(states)]
    runs=[(int(states[a]),int(a),int(b)) for a,b in zip(cuts[:-1],cuts[1:])]
    candidates=[]
    for i,(state,a,b) in enumerate(runs):
        if state!=1 or b-a!=2 or i==0 or i==len(runs)-1:continue
        before=runs[i-1];after=runs[i+1]
        if before[0]!=0 or before[2]-before[1]<3:continue
        if after[0]!=2 or after[2]-after[1]<3:continue
        candidates.append({'mixed_indices':[a,b],'preceding_E_indices':list(before[1:]),
                           'following_P_indices':list(after[1:]),
                           'T_distance_pixels':float((locations[a-1]+locations[a])/2)})
    result['short_transition_audit']={'eligible_candidates':candidates,
        'rule':'Exactly two mixed fields directly between >=3 E-only and >=3 P-only fields.',
        'not_independent_fields':True,'manual_position_used':False}
    if candidates:
        t=candidates[0]['T_distance_pixels']
        result.update(T_path_pixels=t,T_source_xy=[float(np.interp(t,distance,path[:,j])) for j in range(2)],
                      transition_method='two_mixed_fields_candidate',brief_transition_review_required=True)
        # Crop location follows the original image-to-source offset.
        centres=np.asarray(prediction['centres_xy_crop_pixels'])
        result['T_crop_xy']=[float(np.interp(t,locations,centres[:,j])) for j in range(2)]
    return result
