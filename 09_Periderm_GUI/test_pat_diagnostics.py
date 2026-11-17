import unittest
import numpy as np
from pat_diagnostics import sample_path,cleanup_candidate,diagnostic_for_row

class PathMeasurements(unittest.TestCase):
    def test_line_units_and_partition(self):
        m=np.full((5,11),2,np.uint8);m[:,5:]=1
        s,q=sample_path(m,[[0,2],[10,2]],.5)
        self.assertAlmostEqual(s['sampled_path_um'],20)
        self.assertAlmostEqual(s['raw_P_labelled_path_um'],12)
        self.assertAlmostEqual(s['raw_E_labelled_path_um'],8)
        self.assertLessEqual(s['max_sample_step_pixels'],1)
    def test_diagonal_not_skeleton_pixel_count(self):
        m=np.ones((11,11),np.uint8);s,q=sample_path(m,[[0,0],[10,10]],1)
        self.assertAlmostEqual(s['raw_P_labelled_path_um'],np.sqrt(200))
    def test_polyline_vertices_not_double_counted(self):
        m=np.ones((11,11),np.uint8);s,q=sample_path(m,[[0,0],[0,5],[0,5],[10,5]],1)
        self.assertAlmostEqual(s['raw_P_labelled_path_um'],15)
    def test_no_P_is_numeric_zero_not_absence_claim(self):
        m=np.full((10,10),2,np.uint8)
        row={'track_id':1,'path_xy':[[1,2],[7,2]],'status':'no_transition','endpoint_method':'visible_right_end_only'}
        d,q=diagnostic_for_row(row,m,1)
        self.assertEqual(d['raw_P_labelled_path_um'],0)
        self.assertEqual(d['P_detection_state'],'no_sampled_P_not_confirmed_absence')
        self.assertNotIn('periderm_length_um',d)
    def test_multiple_islands_and_only_enclosed_gaps(self):
        m=np.zeros((3,501),np.uint8);m[:,10:100]=1;m[:,150:300]=1
        s,q=sample_path(m,[[0,1],[500,1]],1)
        c=cleanup_candidate(q,1,max_gap=60,max_end_gap=250)
        self.assertEqual(s['P_run_count'],2)
        self.assertEqual(c['cleanup_added_path_um'],50)
        self.assertEqual(c['cleanup_candidate_um'],290)
        self.assertTrue((q['classes'][100:149]==0).all())
    def test_no_extending_far_to_endpoint(self):
        m=np.zeros((3,1001),np.uint8);m[:,10:100]=1
        s,q=sample_path(m,[[0,1],[1000,1]],1)
        self.assertIsNone(cleanup_candidate(q,1)['cleanup_candidate_um'])
    def test_bad_path_and_scale_refused(self):
        m=np.ones((5,5),np.uint8)
        for p,scale in [([[-1,0],[4,0]],1),([[0,0],[4.8,0]],1),([[1,1],[1,1]],1),([[0,0],[4,0]],0)]:
            with self.assertRaises(ValueError):sample_path(m,p,scale)
    def test_low_quality_not_overridden(self):
        d,q=diagnostic_for_row({'track_id':1,'status':'low_image_quality','path_xy':[[1,1],[3,1]]},np.ones((5,5),np.uint8),1)
        self.assertEqual(d['diagnostic_status'],'not_computed');self.assertIsNone(q)
    def test_invalid_label_refused(self):
        with self.assertRaises(ValueError):sample_path(np.full((3,3),255,np.uint8),[[0,1],[2,1]],1)

if __name__=='__main__':unittest.main()
