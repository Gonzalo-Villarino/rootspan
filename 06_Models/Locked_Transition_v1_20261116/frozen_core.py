"""Exact approved image standardization and decoder functions, bundled locally."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
WIDTH, HEIGHT = 320, 256
WINDOW_WIDTH = 320.0
REQUIRED_SPACED_E_FIELDS = 3

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

def boundary_rule(scores,distance):
    """Require both predicted appearances; do not force a mixed segment."""
    e=scores[:,0]>=.5;p=scores[:,1]>=.5
    state=np.full(len(scores),3,dtype=int)
    state[e&~p]=0;state[e&p]=1;state[~e&p]=2
    cut=np.r_[0,np.flatnonzero(np.diff(state))+1,len(state)]
    runs=[{'state':int(state[a]),'start':int(a),'stop_exclusive':int(b),'samples':int(b-a)} for a,b in zip(cut[:-1],cut[1:])]
    candidates=[]
    for i,r in enumerate(runs):
        if r['state']!=1 or r['samples']<3:continue
        prior=[s for s in runs[:i] if s['state']==0 and s['samples']>=3]
        later=[s for s in runs[i+1:] if s['state']==2 and s['samples']>=3]
        if not prior or not later:continue
        a=r['start']
        if not a:continue
        candidates.append({'mixed_run':r,'preceding_clear_Endoderm':prior[-1],
                           'following_clear_Periderm':later[0],
                           'T_distance_pixels':float((distance[a-1]+distance[a])/2)})
    return {'threshold':.5,'state_order':['Endoderm_only','both_mixed','Periderm_only','neither'],
            'window_states':state.tolist(),'runs':runs,'eligible_candidates':candidates,
            'selected':candidates[0] if candidates else None,
            'policy':'First >=3-window mixed run with an earlier >=3-window E-only run and later >=3-window P-only run. No manual T or offset.'}

def spaced_indices(distance,start,stop,spacing=WINDOW_WIDTH):
    selected=[]
    for i in range(start,stop):
        if not selected or distance[i]-distance[selected[-1]]>=spacing-1e-8:
            selected.append(i)
    return selected

def continuity_rule(scores,distance):
    scores=np.asarray(scores,dtype=float);distance=np.asarray(distance,dtype=float)
    if scores.ndim!=2 or scores.shape!=(len(distance),2):raise ValueError('Expected two appearance scores per path position.')
    if not np.isfinite(scores).all() or not np.isfinite(distance).all():raise ValueError('Non-finite inputs.')
    if np.any((scores<0)|(scores>1)) or np.any(np.diff(distance)<=0):raise ValueError('Invalid scores or unordered path.')
    original=boundary_rule(scores,distance);audits=[];retained=[]
    for candidate in original['eligible_candidates']:
        stop=candidate['following_clear_Periderm']['start']
        start=candidate['mixed_run']['stop_exclusive'];contradictions=[]
        for run in original['runs']:
            if run['state']!=0:continue
            a=max(start,run['start']);b=min(stop,run['stop_exclusive'])
            ids=spaced_indices(distance,a,b) if a<b else []
            if len(ids)>=REQUIRED_SPACED_E_FIELDS:
                contradictions.append({'run':run,'spaced_field_indices':ids,
                    'spaced_centres_path_pixels':distance[ids].tolist(),
                    'clear_E_centre_span_pixels':float(distance[b-1]-distance[a])})
        accepted=not contradictions
        audits.append({'original_candidate':candidate,'retained':accepted,
                       'contradictory_E_only_runs':contradictions})
        if accepted:retained.append(candidate)
    return {'rule_version':'continuity_guard_v1','presence_threshold':.5,
            'window_width_native_path_pixels':WINDOW_WIDTH,
            'required_spaced_E_fields':REQUIRED_SPACED_E_FIELDS,
            'original_rule':original,'candidate_audits':audits,
            'selected':retained[0] if retained else None,
            'policy':'Keep both-appearance criterion and original support requirements; reject candidates followed by an E-only run with at least3 centres spaced by one window width before their supporting Periderm run. No new candidate coordinates or offsets invented.',
            'assumption':'Long E-only returns weigh against an early mixed burst; short E-only fields can occur within mixed tissue.',
            'new_neural_training':False}

def run_checks():
    E=[.99,.01];M=[.99,.99];P=[.01,.99];N=[.01,.01]
    def check(seq):
        s=np.array(seq);d=np.arange(len(seq))*64.;return continuity_rule(s,d)
    # No mixed state may be manufactured from one tissue or missing support.
    for seq in [[E]*40,[P]*40,[M]*40,[N]*40,
                [E]*12+[P]*12,[M]*12+[P]*12,[E]*12+[M]*12]:
        assert check(seq)['selected'] is None
    clean=check([E]*20+[M]*6+[P]*20)
    assert clean['selected']==clean['original_rule']['selected']
    # Deliberate early mixed burst, then a long E return, then supported mixture.
    noisy=check([E]*20+[M]*3+[E]*30+[M]*6+[P]*20)
    assert noisy['original_rule']['selected']['mixed_run']['start']==20
    assert noisy['selected']['mixed_run']['start']==53
    assert not noisy['candidate_audits'][0]['retained']
    # Mixed-zone heterogeneity: brief E-only fields must not by themselves veto T.
    short=check([E]*20+[M]*4+[E]*6+[M]*4+[P]*20)
    assert short['selected']==short['original_rule']['selected']
    # If all candidates are contradicted, abstain rather than force a T.
    assert check([E]*20+[M]*3+[E]*30+[P]*20)['selected'] is None
    # Verify actual spatial separation rather than counting overlapping neighbours.
    assert len(spaced_indices(np.arange(11)*64.,0,10))==2
    assert len(spaced_indices(np.arange(11)*64.,0,11))==3
    assert len(spaced_indices(np.arange(10)*80.,0,9))==3
    print('Passed continuity checks: no forced mixed state/T, sustained-return rejection, short-return tolerance, spatial support.')

