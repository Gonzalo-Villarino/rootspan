import copy
import csv
import io
import unittest
from per_root_estimates import attach_report, report_csv


class PerRootReportChecks(unittest.TestCase):
    def setUp(self):
        self.job = {'summary': {'source_sha256': 'image', 'pixels_per_micrometre': .5,
            'rows': [{'track_id': i, 'status': status, 'periderm_length_um': None, 'notes': []}
                     for i, status in enumerate(['transition_beyond_endpoint', 'endpoint_uncertain',
                                                  'no_transition', 'low_image_quality'], 1)],
            'model_diagnostics': {'source_sha256': 'image', 'pixels_per_micrometre': .5,
                'version': 'test', 'raw_mask_sha256': 'mask', 'rows': [
                    {'track_id': 1, 'diagnostic_status': 'computed', 'PAT_cleanup_P_labelled_path_um': 8},
                    {'track_id': 2, 'diagnostic_status': 'computed', 'PAT_cleanup_P_labelled_path_um': 1234.56,
                     'path_end_type': 'visible_right_end_only'},
                    {'track_id': 3, 'diagnostic_status': 'computed', 'PAT_cleanup_P_labelled_path_um': 0},
                    {'track_id': 4, 'diagnostic_status': 'withheld', 'diagnostic_reason': 'Low detail'}]}}}
        self.job['summary']['rows'][0]['expert_reference'] = {
            'id': 'bound_reference', 'periderm_length_um': 0, 'finding': 'Confirmed absence'}

    def test_uncertain_T_and_J_do_not_suppress_available_numbers(self):
        report = attach_report(self.job)['summary']['per_root_report']
        self.assertEqual([r['periderm_length_um'] for r in report['rows']], [0,1234.56,0,None])
        self.assertEqual(report['numeric_count'],3)

    def test_expert_correction_retains_automatic_value_and_source(self):
        r=attach_report(self.job)['summary']['per_root_report']['rows'][0]
        self.assertEqual(r['periderm_length_um'],0)
        self.assertEqual(r['automatic_mask_estimate_um'],8)
        self.assertEqual(r['value_source'],'Expert correction')

    def test_csv_values_and_basis_match_main_report(self):
        j=attach_report(self.job)
        csv_rows=list(csv.DictReader(io.StringIO(report_csv(j))))
        for record, saved in zip(j['summary']['per_root_report']['rows'],csv_rows):
            value=record['periderm_length_um']
            self.assertEqual(saved['periderm_length_um'],'' if value is None else str(value))
            self.assertEqual(saved['value_source'],record['value_source'])
            self.assertIn('not validated T-to-J',saved['measurement_definition'])

    def test_no_mutation_of_original_measurements(self):
        original=copy.deepcopy(self.job);attach_report(self.job);self.assertEqual(self.job,original)

    def test_missing_data_never_fabricates_zero(self):
        r=attach_report(self.job)['summary']['per_root_report']['rows'][3]
        self.assertIsNone(r['periderm_length_um']);self.assertEqual(r['status'],'unmeasurable')

    def test_wrong_source_and_wrong_scale_rejected(self):
        for key,value in [('source_sha256','different'),('pixels_per_micrometre',1)]:
            j=copy.deepcopy(self.job);j['summary']['model_diagnostics'][key]=value
            with self.assertRaises(ValueError):attach_report(j)

    def test_bad_numeric_or_duplicate_root_rejected(self):
        for value in [float('nan'),float('inf'),-1,True]:
            j=copy.deepcopy(self.job);j['summary']['model_diagnostics']['rows'][0]['PAT_cleanup_P_labelled_path_um']=value
            with self.assertRaises(ValueError):attach_report(j)
        self.job['summary']['model_diagnostics']['rows'].append(self.job['summary']['model_diagnostics']['rows'][0])
        with self.assertRaises(ValueError):attach_report(self.job)

    def test_failed_comparison_never_shows_stale_report(self):
        j=attach_report(self.job);j['summary']['diagnostic_error']='failed'
        self.assertNotIn('per_root_report',attach_report(j)['summary'])

    def test_missing_bindings_and_invalid_expert_value_rejected(self):
        for key in ['source_sha256','pixels_per_micrometre']:
            j=copy.deepcopy(self.job)
            j['summary'].pop(key);j['summary']['model_diagnostics'].pop(key)
            with self.assertRaises(ValueError):attach_report(j)
        self.job['summary']['rows'][0]['expert_reference']['periderm_length_um']=float('nan')
        with self.assertRaises(ValueError):attach_report(self.job)

    def test_reporting_error_preserves_successful_pipeline(self):
        import tempfile,types,json
        from pathlib import Path
        from unittest.mock import patch
        import server
        with tempfile.TemporaryDirectory() as t:
            base=Path(t);(base/'run').mkdir();(base/'refs').mkdir()
            original=copy.deepcopy(self.job['summary'])
            def pipeline(source,out,scale,report):out.mkdir();return copy.deepcopy(original)
            job={'id':'run','pipeline':'v2','scale':.5}
            with patch.object(server,'RUNS',base),patch.object(server,'REFERENCES',base/'refs'),patch.dict('sys.modules',{'pipeline_v2':types.SimpleNamespace(run=pipeline)}),patch('version_registry.verify_v2'),patch('pat_diagnostics.augment',side_effect=lambda source,out,result,report:result),patch.object(server,'attach_report',side_effect=ValueError('simulated report failure')):
                server.task(job,base/'source.tif')
            self.assertEqual(job['status'],'complete')
            self.assertEqual(job['summary']['rows'],original['rows'])
            self.assertIn('simulated report failure',job['summary']['per_root_report_error'])
            self.assertEqual(json.loads((base/'run/job.json').read_text())['status'],'complete')


if __name__=='__main__':unittest.main()
