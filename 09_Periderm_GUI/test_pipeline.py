"""Checks for unit conversion, invalid scales and merged-track detection."""
import unittest
import numpy as np
from pipeline import validate_scale, connected_group_fraction, arc, detect_tracks
from PIL import Image,ImageDraw
from root_tracks import fill_small_holes, split_component
from image_quality import window_detail, DETAIL_LIMIT
from scipy import ndimage as ndi

class PipelineChecks(unittest.TestCase):
    def test_scale(self):
        self.assertAlmostEqual(5299/validate_scale(.5299),10000)
        for bad in (0,-1,float('nan'),float('inf'),10001):
            with self.assertRaises(ValueError):validate_scale(bad)

    def test_curved_length(self):
        # A bent line must be measured along its path, not the shorter chord.
        path=np.array([[0.,0.],[3.,4.],[6.,0.]])
        self.assertAlmostEqual(arc(path)[-1],10)
        self.assertGreater(arc(path)[-1],np.linalg.norm(path[-1]-path[0]))

    def test_connected_parallel_roots(self):
        single=np.zeros((100,200),bool);single[20:30,:]=True
        self.assertEqual(connected_group_fraction(single),0)
        joined=single.copy();joined[70:80,:]=True;joined[20:80,100:103]=True
        self.assertGreater(connected_group_fraction(joined),.9)

    def test_large_space_between_joined_roots_is_not_filled(self):
        m=np.zeros((250,1000),bool)
        m[25:33,10:990]=True;m[200:208,10:990]=True
        m[25:208,10:18]=True;m[25:208,982:990]=True
        m[27:29,500:502]=False
        filled=fill_small_holes(m)
        self.assertTrue(filled[27:29,500:502].all())
        self.assertFalse(filled[100:150,100:900].any())

    def test_curved_touching_roots_separate_into_two_long_lanes(self):
        m=np.zeros((350,1000),bool)
        x=np.arange(1000);top=(65+25*np.sin(x/130)).astype(int)
        lower=(210+60*np.sin(x/220)).astype(int)
        for xx in x:
            m[top[xx]-5:top[xx]+6,xx]=True
            m[lower[xx]-5:lower[xx]+6,xx]=True
        m[top[10]:lower[10]+1,8:12]=True
        m[top[990]:lower[990]+1,988:992]=True
        lanes=split_component((slice(0,350),slice(0,1000)),m)
        self.assertEqual(len(lanes),2)
        for box,mask,meta in lanes:
            self.assertTrue(meta['split_applied'])
            self.assertGreater(box[1].stop-box[1].start,900)

    def test_severe_blur_reduces_detail_below_guard(self):
        y,x=np.mgrid[:128,:320]
        cells=np.where((y>=24)&(y<104),
                       30+120*((x%48<3)|(y%28<3)).astype(float),0)
        sharp,_=window_detail(cells)
        blurred,_=window_detail(ndi.gaussian_filter(cells,12))
        self.assertGreater(sharp,DETAIL_LIMIT)
        self.assertLess(blurred,DETAIL_LIMIT)

    def test_partial_root_at_border_is_retained_as_a_flag(self):
        im=Image.new('RGB',(6400,1400))
        draw=ImageDraw.Draw(im)
        draw.rectangle((4000,0,6200,60),fill=(0,100,0))
        draw.rectangle((100,600,6200,660),fill=(0,100,0))
        found=detect_tracks(im)
        self.assertEqual(len(found),2)
        self.assertTrue(found[0][2]['partial_at_image_edge'])
        self.assertFalse(found[1][2].get('partial_at_image_edge',False))

if __name__=='__main__':unittest.main()
