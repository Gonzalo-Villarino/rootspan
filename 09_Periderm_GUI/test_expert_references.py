import csv,io,json,tempfile,unittest
from pathlib import Path
from expert_references import attach_references,measurements_download,track_signature,ExpertReferenceError

class ExpertReferenceChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.directory=Path(self.tmp.name)
        self.row={'track_id':1,'detection_bbox_xyxy':[0,0,100,20],'transition_xy':[90,10],'endpoint_xy':[80,10],'periderm_length_um':None,'status':'transition_beyond_endpoint'}
        self.job={'summary':{'source_sha256':'source-a','pipeline_version':'v2-test','provisional_lengths':0,'rows':[self.row]}}
        self.ref={'id':'human-1','source_sha256':'source-a','pipeline_version':'v2-test','track_id':1,'track_signature':track_signature(self.row),'periderm_length_um':0.0,'finding':'Periderm absent (expert reference)','expert':'Example Expert','expert_words':'0 periderm'}
        (self.directory/'ref.json').write_text(json.dumps(self.ref))
    def tearDown(self):self.tmp.cleanup()
    def test_zero_is_retained_separately_without_changing_automatic_result(self):
        annotated=attach_references(self.job,self.directory);r=annotated['summary']['rows'][0]
        self.assertEqual(r['expert_reference']['periderm_length_um'],0)
        self.assertIsNone(r['periderm_length_um']);self.assertEqual(r['status'],'transition_beyond_endpoint')
        self.assertNotIn('expert_reference',self.row);self.assertEqual(annotated['summary']['provisional_lengths'],0)
    def test_different_source_version_or_geometry_never_gets_reference(self):
        for key,value in [('source_sha256','different'),('pipeline_version','v3')]:
            job=json.loads(json.dumps(self.job));job['summary'][key]=value
            self.assertEqual(attach_references(job,self.directory)['summary']['expert_reference_count'],0)
        job=json.loads(json.dumps(self.job));job['summary']['rows'][0]['endpoint_xy']=[85,10]
        self.assertEqual(attach_references(job,self.directory)['summary']['expert_reference_count'],0)
    def test_csv_keeps_automatic_blank_and_expert_zero(self):
        job=attach_references(self.job,self.directory);text='track_id,periderm_length_um,status\n1,,transition_beyond_endpoint\n'
        row=list(csv.DictReader(io.StringIO(measurements_download(text,job))))[0]
        self.assertEqual(row['periderm_length_um'],'');self.assertEqual(row['expert_periderm_length_um'],'0.0');self.assertEqual(row['expert_reference_id'],'human-1')
        self.assertEqual(row['expert_reference_author'],'Example Expert')
    def test_withdrawn_exports_keep_warning(self):
        row=list(csv.DictReader(io.StringIO(measurements_download('track_id,periderm_length_um\n1,100\n',self.job,True))))[0]
        self.assertEqual(row['periderm_length_um'],'100');self.assertEqual(row['record_status'],'withdrawn')
    def test_duplicate_reference_does_not_silently_choose(self):
        (self.directory/'ref2.json').write_text(json.dumps(self.ref))
        with self.assertRaisesRegex(ExpertReferenceError,'conflict: ref2.json'):attach_references(self.job,self.directory)
    def test_reference_is_only_attached_to_matching_row(self):
        other=dict(self.row,track_id=2);self.job['summary']['rows'].append(other)
        result=attach_references(self.job,self.directory)
        self.assertEqual(result['summary']['expert_reference_count'],1)
        self.assertNotIn('expert_reference',result['summary']['rows'][1])
    def test_malformed_reference_names_the_unreadable_file(self):
        (self.directory/'bad.json').write_text('{')
        with self.assertRaisesRegex(ExpertReferenceError,'unreadable: bad.json'):attach_references(self.job,self.directory)
    def test_invalid_reference_fields_do_not_become_missing_measurements(self):
        for key,value in [('periderm_length_um',None),('periderm_length_um',-1),('periderm_length_um',float('nan')),('periderm_length_um',True),('finding',None),('track_id',True),('id','')]:
            with self.subTest(key=key,value=value):
                ref=dict(self.ref);ref[key]=value
                (self.directory/'ref.json').write_text(json.dumps(ref))
                with self.assertRaisesRegex(ExpertReferenceError,'unreadable: ref.json'):attach_references(self.job,self.directory)

if __name__=='__main__':unittest.main()
