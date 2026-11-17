"""Regression: measuring a shorter root must not crop its inspection context."""
import unittest
import numpy as np
import pipeline
import pipeline_v2

class InspectionContextTests(unittest.TestCase):
    def test_widening_keeps_same_measurement_but_preserves_inspection_tail(self):
        mask=np.zeros((150,1600),bool)
        mask[65:85,5:1595]=True
        # A sustained wider right-hand segment provides a junction candidate.
        mask[45:105,1320:1595]=True
        box=(slice(200,350),slice(50,1650))
        old,old_detail=pipeline.route_from_component(box,mask)
        full,detail=pipeline_v2.route_from_component(box,mask)
        self.assertEqual(detail['endpoint_method'],'width_based_junction_candidate')
        self.assertEqual(old_detail['endpoint_method'],detail['endpoint_method'])
        np.testing.assert_allclose(old,full[:detail['measurement_path_index']+1])
        self.assertGreater(detail['inspection_distance_pixels']-detail['measurement_distance_pixels'],160)

    def test_no_junction_does_not_invent_a_measuring_endpoint(self):
        mask=np.zeros((150,1600),bool);mask[65:85,5:1595]=True
        box=(slice(200,350),slice(50,1650))
        old,_=pipeline.route_from_component(box,mask)
        full,detail=pipeline_v2.route_from_component(box,mask)
        np.testing.assert_allclose(old,full)
        self.assertEqual(detail['endpoint_method'],'visible_right_end_only')
        self.assertEqual(detail['measurement_path_index'],len(full)-1)

if __name__=='__main__':unittest.main()
