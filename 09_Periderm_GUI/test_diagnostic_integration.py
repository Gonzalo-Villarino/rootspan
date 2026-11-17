import copy,hashlib,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import pat_diagnostics as pd
import server

class IntegrationChecks(unittest.TestCase):
    def test_augment_separate_metrics_and_nonmutating_write_failure(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t);out=base/'results';out.mkdir();masks=base/'masks';masks.mkdir()
            source=base/'image.png';Image.new('RGB',(20,20)).save(source)
            mask=np.full((20,20),2,np.uint8);mask[:,10:]=1;Image.fromarray(mask).save(masks/'raw_classes.png')
            meta={'source_sha256':'fixture','model_name':'published4','weights_sha256':'71ba82d7eab40c0b8489f47ee373fd0ea08f2e2e33457c62152a1f8240d68348'}
            (masks/'prediction_record.json').write_text(json.dumps(meta))
            row={'track_id':1,'path_xy':[[1,10],[18,10]],'status':'endpoint_uncertain','endpoint_method':'visible_right_end_only','periderm_length_um':None}
            (out/'track_01').mkdir();(out/'track_01/result.json').write_text(json.dumps(row))
            summary={'rows':[row],'source_sha256':'fixture','source_dimensions_xy':[20,20],'pixels_per_micrometre':.5,'pipeline_version':'test'}
            original=copy.deepcopy(summary)
            def cleanup_stub(args,**kwargs):
                dest=Path(args[-1]);(dest/'measurement').mkdir(parents=True)
                Image.fromarray((mask==1).astype(np.uint8)*255).save(dest/'measurement/slide.png')
                (dest/'record.json').write_text('{}')
            with patch.object(pd.subprocess,'run',side_effect=cleanup_stub):
                result=pd.augment(source,out,summary,mask_folder=masks)
            self.assertEqual(summary,original);self.assertIsNone(result['rows'][0]['periderm_length_um'])
            d=result['model_diagnostics']['rows'][0]
            self.assertEqual(d['raw_P_labelled_path_um'],d['PAT_cleanup_P_labelled_path_um'])
            self.assertEqual(d['PAT_cleanup_detection_state'],'P_retained')
            self.assertEqual(d['pipeline_version'],'test')
            self.assertEqual(d['diagnostic_version'],pd.VERSION)
            self.assertEqual(d['diagnostic_workflow'],pd.WORKFLOW)
            self.assertEqual(len(d['raw_mask_sha256']),64)
            self.assertIn('not confirmed biological absence',d['scope'])
            # Failure at the final write cannot mutate the caller's original summary.
            other=base/'other';other.mkdir();(other/'track_01').mkdir();(other/'track_01/result.json').write_text(json.dumps(row))
            original_write=pd.write
            def failing_write(path,value):
                if path.name=='results.json':raise OSError('simulated disk error')
                original_write(path,value)
            with patch.object(pd.subprocess,'run',side_effect=cleanup_stub),patch.object(pd,'write',side_effect=failing_write):
                with self.assertRaises(OSError):pd.augment(source,other,summary,mask_folder=masks)
            self.assertEqual(summary,original)

    def test_server_diagnostic_failure_preserves_main_result(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'run').mkdir()
            original={'rows':[{'track_id':1,'periderm_length_um':None}],'provisional_lengths':0}
            def pipeline_stub(source,out,scale,report):out.mkdir();return copy.deepcopy(original)
            def fail(source,out,result,report):
                (out/'model_diagnostics').mkdir()
                result['rows'][0]['periderm_length_um']=999
                raise RuntimeError('comparison intentionally failed')
            job={'id':'run','pipeline':'v2','include_diagnostics':True,'scale':.5299}
            with patch.object(server,'RUNS',root),patch.dict('sys.modules',{'pipeline_v2':types.SimpleNamespace(run=pipeline_stub)}),patch('version_registry.verify_v2'),patch('pat_diagnostics.augment',side_effect=fail):
                server.task(job,root/'source.tif')
            self.assertEqual(job['status'],'complete')
            self.assertIsNone(job['summary']['rows'][0]['periderm_length_um'])
            self.assertIn('comparison intentionally failed',job['summary']['diagnostic_error'])
            self.assertEqual(job['summary']['provisional_lengths'],0)
            self.assertTrue((root/'run/results/model_diagnostics/INVALID_INCOMPLETE.txt').exists())

if __name__=='__main__':unittest.main()
