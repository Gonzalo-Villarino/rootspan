from pathlib import Path
import hashlib,json,sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'09_Periderm_GUI'))
from approved_release import verify_release

class ReleaseLockChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
        self.b=self.p/'bundle';self.b.mkdir();(self.b/'snapshot').mkdir()
        (self.p/'app.py').write_text('approved')
        (self.b/'snapshot/app.py').write_text('approved')
        (self.p/'refs').mkdir();(self.p/'refs/zero.json').write_text('{}')
        self.m={'release_id':'test','app_version':'test','runtime_files':{'app.py':{'sha256':hashlib.sha256(b'approved').hexdigest()}},'guarded_inventories':{'refs':['zero.json']},'environments':{'python':{'python':'test'}}}
        (self.b/'MANIFEST.json').write_text(json.dumps(self.m))
        self.pointer=self.p/'pointer.json'
        self.pointer.write_text(json.dumps({'release_id':'test','bundle_relative':'bundle','manifest_sha256':hashlib.sha256((self.b/'MANIFEST.json').read_bytes()).hexdigest()}))
    def tearDown(self):self.tmp.cleanup()
    def verify(self,**kwargs):return verify_release(self.p,self.pointer,check_environment=False,**kwargs)
    def test_active_and_backup_match(self):self.assertTrue(self.verify(check_snapshot=True)['passed'])
    def test_changed_code_rejected(self):
        (self.p/'app.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'version changed'):self.verify()
    def test_changed_backup_rejected(self):
        (self.b/'snapshot/app.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'backup changed'):self.verify(check_snapshot=True)
    def test_new_expert_reference_rejected(self):
        (self.p/'refs/extra.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'inventory changed'):self.verify()
    def test_missing_pointer_rejected(self):
        self.pointer.unlink()
        with self.assertRaisesRegex(ValueError,'record is missing'):self.verify()
    def test_changed_manifest_rejected(self):
        (self.b/'MANIFEST.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'manifest changed'):self.verify()
    def test_environment_version_change_rejected(self):
        with patch('approved_release.environment',return_value={'python':'different'}):
            with self.assertRaisesRegex(ValueError,'package versions changed'):
                verify_release(self.p,self.pointer)

if __name__=='__main__':unittest.main()
