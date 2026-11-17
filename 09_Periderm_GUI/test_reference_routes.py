import json,tempfile,threading,unittest,urllib.request,urllib.error
from pathlib import Path
from unittest.mock import patch
from http.server import ThreadingHTTPServer
import server

class ReferenceRouteChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        self.refs=self.base/'refs';self.refs.mkdir();(self.refs/'broken.json').write_text('{')
        results=self.base/'runs/example/results';results.mkdir(parents=True)
        (results/'measurements.csv').write_text('track_id,periderm_length_um\n1,\n')
        self.job={'id':'example','created_at':'2026-11-17','summary':{'rows':[]}}
        (results.parent/'job.json').write_text(json.dumps(self.job))
        self.refs_patch=patch.object(server,'REFERENCES',self.refs);self.runs_patch=patch.object(server,'RUNS',self.base/'runs')
        self.refs_patch.start();self.runs_patch.start()
        self.http=ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
        self.thread=threading.Thread(target=self.http.serve_forever,daemon=True);self.thread.start()
        self.url='http://127.0.0.1:'+str(self.http.server_port)
    def tearDown(self):
        self.http.shutdown();self.http.server_close();self.thread.join()
        self.refs_patch.stop();self.runs_patch.stop();self.tmp.cleanup()
    def assert_reference_error(self,path):
        with self.assertRaises(urllib.error.HTTPError) as caught:urllib.request.urlopen(self.url+path)
        self.assertEqual(caught.exception.code,409)
        self.assertEqual(json.load(caught.exception)['error'],'Expert reference unreadable: broken.json')
    def test_api_identifies_reference_problem(self):self.assert_reference_error('/api/state?job=example')
    def test_csv_identifies_reference_problem(self):self.assert_reference_error('/files/example/results/measurements.csv')

if __name__=='__main__':unittest.main()
