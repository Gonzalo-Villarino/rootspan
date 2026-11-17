"""Regression for the endpoint failure identified in independent review."""
import unittest
import numpy as np
from endpoints_v211 import proximal_endpoint

class EndpointReviewChecks(unittest.TestCase):
    def setUp(self):
        self.path=np.c_[np.arange(100.,2101.,4),np.full(501,100.)]
        self.arc=self.path[:,0]-100
        self.previous={'endpoint_method':'visible_right_end_only','measurement_path_index':500,
                       'measurement_distance_pixels':2000.}
        self.box=(slice(0,256),slice(0,2700))

    def test_root_ending_inside_dark_image_does_not_become_junction(self):
        green=np.zeros((256,2700),np.uint8);green[90:111,100:2102]=100
        lane=green>0
        e,a=proximal_endpoint(green,self.path,self.arc,self.previous,lane,self.box,1)
        self.assertEqual(e,self.previous)
        self.assertIsNone(a['selected'])

    def test_neighbour_signal_cannot_supply_endpoint(self):
        green=np.zeros((256,2700),np.uint8);green[95:106,100:2102]=100
        lane=green>0
        # Strong neighbouring fluorescence drops near the right end; excluded.
        green[110:145,100:1800]=255
        e,a=proximal_endpoint(green,self.path,self.arc,self.previous,lane,self.box,1)
        self.assertEqual(e,self.previous)
        self.assertIsNone(a['selected'])

    def test_any_candidate_has_complete_observed_after_window(self):
        green=np.zeros((256,2700),np.uint8);green[90:111,100:1600]=150
        green[90:111,1600:2102]=30
        e,a=proximal_endpoint(green,self.path,self.arc,self.previous,green>0,self.box,1)
        self.assertIsNotNone(a['selected'])
        self.assertLessEqual(a['selected']['distance_pixels']+384+24,self.arc[-1])
        self.assertFalse(e['anatomical_endpoint_verified'])

if __name__=='__main__':unittest.main()
