"""Safety and regression checks for candidate geometry; not biological validation."""
import json
from pathlib import Path
import unittest
import numpy as np
from endpoints_v21 import proximal_endpoint
from transition_v21 import with_short_transition
from root_tracks_v21 import split_component
from pipeline_v21 import BUNDLE
from frozen_core import continuity_rule

class CandidateChecks(unittest.TestCase):
    def apply_rule(self,seq):
        d=np.arange(len(seq))*64.+160
        q={'presence_scores':seq,'distance_along_local_path':d.tolist(),
           'centres_xy_crop_pixels':np.c_[d,np.zeros(len(d))].tolist(),
           'T_path_pixels':None,'T_source_xy':None,'T_crop_xy':None}
        return with_short_transition(q,np.c_[d,np.zeros(len(d))],d)

    def test_fallback_needs_both_tissues_and_mixture(self):
        E=[.99,.01];M=[.99,.99];P=[.01,.99]
        for seq in ([E]*30,[P]*30,[M]*30,[E]*10+[P]*10,
                    [E]*10+[M]+[P]*10,[M]*2+[P]*10,[E]*10+[M]*2):
            self.assertIsNone(self.apply_rule(seq)['T_path_pixels'])
        q=self.apply_rule([E]*10+[M]*2+[P]*10)
        self.assertIsNotNone(q['T_path_pixels'])
        self.assertTrue(q['brief_transition_review_required'])

    def test_approved_profiles_keep_locked_transition(self):
        checked=0
        for f in (BUNDLE/'fixtures/profiles').glob('*.json'):
            r=json.loads(f.read_text());q=r['prediction']
            scores=q.get('presence_scores',q.get('predicted_presence_scores'))
            d=q.get('distance_along_local_path',q.get('distance_native_pixels'))
            rule=continuity_rule(scores,d)
            t=None if rule['selected'] is None else rule['selected']['T_distance_pixels']
            self.assertEqual(t,r['expected_T_path_pixels'])
            if t is not None:
                result=with_short_transition({'T_path_pixels':t},None,None)
                self.assertEqual(result['T_path_pixels'],t)
                self.assertFalse(result['brief_transition_review_required'])
            checked+=1
        self.assertGreaterEqual(checked,3)

    def test_uniform_signal_does_not_invent_endpoint(self):
        path=np.c_[np.arange(100,2101,4),np.full(501,100.)]
        d=path[:,0]-100
        previous={'endpoint_method':'visible_right_end_only','measurement_path_index':500,
                  'measurement_distance_pixels':2000.}
        for value in (0,100):
            e,a=proximal_endpoint(np.full((256,2700),value,np.uint8),path,d,previous)
            self.assertEqual(e,previous)
            self.assertIsNone(a['selected'])

    def test_image_border_is_not_anatomical_endpoint(self):
        path=np.c_[np.arange(100,2101,4),np.full(501,100.)]
        d=path[:,0]-100
        previous={'endpoint_method':'visible_right_end_only','measurement_path_index':500,
                  'measurement_distance_pixels':2000.}
        e,a=proximal_endpoint(np.full((256,2102),100,np.uint8),path,d,previous)
        self.assertEqual(e,previous)
        self.assertIsNone(a['selected'])

    def test_curved_touching_roots_remain_separate(self):
        m=np.zeros((350,1000),bool);x=np.arange(1000)
        top=(65+25*np.sin(x/130)).astype(int);lower=(210+60*np.sin(x/220)).astype(int)
        for xx in x:
            m[top[xx]-5:top[xx]+6,xx]=True
            m[lower[xx]-5:lower[xx]+6,xx]=True
        m[top[10]:lower[10]+1,8:12]=True;m[top[990]:lower[990]+1,988:992]=True
        lanes=split_component((slice(0,350),slice(0,1000)),m,100*m)
        self.assertEqual(len(lanes),2)
        for box,mask,meta in lanes:
            self.assertTrue(meta['split_applied'])
            self.assertGreater(box[1].stop-box[1].start,900)
        # Shared endpoints must not cause the two lane masks to overlap.
        occupied=np.zeros_like(m)
        for box,mask,_ in lanes:
            self.assertFalse((occupied[box]&mask).any())
            occupied[box]|=mask

if __name__=='__main__':unittest.main()
