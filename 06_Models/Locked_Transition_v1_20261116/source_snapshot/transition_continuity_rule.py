"""Reject an early mixed candidate contradicted by a long E-only return.

This is a versioned development decoder; it never changes neural scores.
Both predicted appearances are still required for a mixed window. Spacing is
measured along the main-root path, not claimed to be independent biology.
"""
import numpy as np
from test_both_appearances import boundary_rule

WINDOW_WIDTH=320.0
REQUIRED_SPACED_E_FIELDS=3

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

if __name__=='__main__':run_checks()
