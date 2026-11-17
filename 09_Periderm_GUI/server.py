"""Local-only browser GUI. Images and results remain on this Mac."""
from pathlib import Path
import argparse
import copy
import datetime
import json
import mimetypes
import os
import secrets
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote, parse_qs
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from expert_references import attach_references,measurements_download,ExpertReferenceError
from per_root_estimates import attach_report, report_csv
from approved_release import verify_release

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
RUNS=HERE/'runs'
UPLOADS=HERE/'uploads'
REFERENCES=HERE/'expert_references'
TOKEN=secrets.token_urlsafe(32)
LOCK=threading.Lock()
WORKER=ThreadPoolExecutor(max_workers=1)
ACTIVE=None
RELEASE=None
Image.MAX_IMAGE_PIXELS=200_000_000


def atomic_json(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def is_withdrawn(record):
    summary=record.get('summary') or {}
    return record.get('pipeline')=='v21' or str(summary.get('pipeline_version','')).startswith('automatic_pipeline_v2.1')

def jobs():
    records=[]
    for p in sorted(RUNS.glob('*/job.json'),reverse=True):
        try:records.append(json.loads(p.read_text()))
        except (OSError,ValueError):continue
    return sorted(records,key=lambda r:r['created_at'],reverse=True)

def report_view(job):
    out=attach_references(job,REFERENCES)
    try:return attach_report(out)
    except (ValueError,KeyError,TypeError) as error:
        out.setdefault('summary',{}).pop('per_root_report',None)
        out['summary']['per_root_report_error']='Per-root estimates unavailable: '+str(error)
        return out

def task(job,source):
    global ACTIVE
    folder=RUNS/job['id']
    def report(text):
        job['message']=text
        atomic_json(folder/'job.json',job)
    try:
        job['approved_release']=verify_release()
        if job.get('pipeline', 'v1') == 'v21':
            raise ValueError('V2.1 is withdrawn: its endpoint estimates were not reliable. Use V1 or V2 for provisional checks.')
        elif job.get('pipeline', 'v1') == 'v2':
            from version_registry import verify_v2
            verify_v2()
            import pipeline_v2 as pipeline
        else:
            import pipeline
        job['status']='running';report('Starting local analysis…')
        result=pipeline.run(source,folder/'results',job['scale'],report)
        if job.get('pipeline')=='v2':
            try:
                from pat_diagnostics import augment
                result=augment(source,folder/'results',copy.deepcopy(result),report)
            except Exception as diagnostic_error:
                import traceback
                (folder/'diagnostic_error.txt').write_text(traceback.format_exc())
                partial=folder/'results/model_diagnostics'
                if partial.exists():
                    (partial/'INVALID_INCOMPLETE.txt').write_text('This supplementary calculation failed. Do not use these partial files. See diagnostic_error.txt in the run folder.\n')
                result['diagnostic_error']='The extra PAT comparison failed: '+str(diagnostic_error)+'. Main V2 results are preserved.'
                atomic_json(folder/'results/results.json',result)
        job['status']='complete';job['summary']=result
        try:
            reviewed=report_view(job)
            if reviewed['summary'].get('per_root_report_error'):
                result['per_root_report_error']=reviewed['summary']['per_root_report_error']
            if reviewed['summary'].get('per_root_report'):
                result['per_root_report']=reviewed['summary']['per_root_report']
                (folder/'results/periderm_estimates.csv').write_text(report_csv(reviewed))
            atomic_json(folder/'results/results.json',result)
        except Exception as report_error:
            result.pop('per_root_report',None)
            result['per_root_report_error']='Per-root report could not be saved: '+str(report_error)
            atomic_json(folder/'results/results.json',result)
        report('Complete — open the results below.')
    except Exception as error:
        import traceback
        (folder/'error.txt').write_text(traceback.format_exc())
        job['status']='failed';report(str(error))
    finally:
        with LOCK:ACTIVE=None

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send(self,status,body,kind='application/json'):
        if not isinstance(body,bytes):body=body.encode()
        self.send_response(status);self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers();self.wfile.write(body)
    def json(self,obj,status=200):self.send(status,json.dumps(obj,allow_nan=False))
    def safe_path(self,base,relative):
        candidate=(base/relative).resolve()
        if not candidate.is_relative_to(base.resolve()):raise ValueError('Invalid path')
        return candidate
    def do_GET(self):
        route=urlparse(self.path).path
        if route=='/':
            html=(HERE/'index.html').read_text().replace('__TOKEN__',TOKEN)
            return self.send(200,html,'text/html; charset=utf-8')
        if route=='/api/state':
            selected=parse_qs(urlparse(self.path).query).get('job',[''])[0]
            job=None
            if selected:
                job=next((record for record in jobs() if record['id']==selected),None)
                if job:
                    try:job=report_view(job)
                    except (ExpertReferenceError,ValueError) as error:return self.json({'error':str(error)},409)
            return self.json({'job':job,'active':ACTIVE,'version':'0.6.0-rootspan.1','local_only':True,'approved_release':RELEASE,
                              'pipelines':['v1','v2'],'withdrawn_pipelines':['v21']})
        if route.startswith('/files/'):
            try:
                p=self.safe_path(RUNS,unquote(route.removeprefix('/files/')))
                if p.name=='periderm_estimates.csv' and p.parent.name=='results':
                    record=p.parent.parent/'job.json'
                    if not record.is_file():return self.json({'error':'File not found'},404)
                    job=report_view(json.loads(record.read_text()))
                    return self.send(200,report_csv(job),'text/csv; charset=utf-8')
                if not p.is_file():return self.json({'error':'File not found'},404)
                # Only generated figures/tables are downloadable through this route.
                if p.suffix.lower() not in ('.png','.jpg','.json','.csv','.txt'):return self.json({'error':'Not available'},403)
                if p.name=='measurements.csv' and p.parent.name=='results':
                    record=p.parent.parent/'job.json'
                    if record.is_file():
                        job=attach_references(json.loads(record.read_text()),REFERENCES)
                        return self.send(200,measurements_download(p.read_text(),job,is_withdrawn(job)),'text/csv; charset=utf-8')
                return self.send(200,p.read_bytes(),mimetypes.guess_type(p.name)[0] or 'application/octet-stream')
            except (ExpertReferenceError,ValueError) as error:return self.json({'error':str(error)},409)
            except ValueError:return self.json({'error':'Invalid file'},400)
        return self.json({'error':'Not found'},404)
    def do_POST(self):
        global ACTIVE
        if self.headers.get('X-Session-Token')!=TOKEN:return self.json({'error':'Refresh this local page and try again.'},403)
        origin=self.headers.get('Origin')
        if origin and origin!=f'http://{self.headers.get("Host")}':return self.json({'error':'Invalid origin'},403)
        route=urlparse(self.path).path
        try:
            length=int(self.headers.get('Content-Length','0'))
            if length<=0 or length>1_000_000_000:raise ValueError('File must be smaller than 1 GB.')
            if route=='/api/upload':
                name=Path(unquote(self.headers.get('X-Filename','image.tif'))).name
                if Path(name).suffix.lower() not in ('.tif','.tiff','.png','.jpg','.jpeg'):raise ValueError('Choose a TIFF, PNG or JPEG image.')
                ident=uuid.uuid4().hex
                path=UPLOADS/(ident+Path(name).suffix.lower())
                try:
                    with path.open('xb') as f:
                        remaining=length
                        while remaining:
                            block=self.rfile.read(min(1024*1024,remaining))
                            if not block:raise ValueError('Upload was interrupted.')
                            f.write(block);remaining-=len(block)
                    with Image.open(path) as im:
                        if im.width*im.height>200_000_000:raise ValueError('Image exceeds 200 million pixels.')
                        size=list(im.size);frames=getattr(im,'n_frames',1);mode=im.mode
                        if frames!=1:raise ValueError('Use a single-plane image, not a stack.')
                        if mode not in ('RGB','RGBA'):raise ValueError('This prototype expects an RGB fluorescence image.')
                    atomic_json(UPLOADS/(ident+'.json'),{'id':ident,'path':str(path),'name':name,'size':size})
                except Exception:
                    path.unlink(missing_ok=True);raise
                return self.json({'id':ident,'name':name,'size':size})
            if route!='/api/analyse':return self.json({'error':'Not found'},404)
            if length>16000:raise ValueError('Invalid request.')
            request=json.loads(self.rfile.read(length))
            from pipeline import validate_scale
            scale=validate_scale(request.get('scale'))
            pipeline_version=request.get('pipeline','v1')
            if pipeline_version=='v21':raise ValueError('V2.1 is withdrawn: its endpoint estimates were not reliable. Use V1 or V2 for provisional checks.')
            if pipeline_version not in ('v1','v2'):raise ValueError('Choose V1 or V2. No V3 model has been released.')
            ident=request.get('upload','')
            if len(ident)!=32 or any(c not in '0123456789abcdef' for c in ident):raise ValueError('Choose an image first.')
            metadata=json.loads((UPLOADS/(ident+'.json')).read_text())
            source=Path(metadata['path']);name=metadata['name']
            with LOCK:
                if ACTIVE:return self.json({'error':'An image is already being analysed. Wait for it to finish.'},409)
                job_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
                folder=RUNS/job_id;folder.mkdir()
                job={'id':job_id,'image_name':name,'scale':scale,'pipeline':pipeline_version,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'include_diagnostics':pipeline_version=='v2',
                    'status':'queued','message':'Queued for local analysis','results_folder':str(folder/'results')}
                atomic_json(folder/'job.json',job);ACTIVE=job_id
                WORKER.submit(task,job,source)
            return self.json(job,202)
        except Exception as error:return self.json({'error':str(error)},400)

def main():
    global RELEASE
    RELEASE=verify_release()
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8766)
    a=parser.parse_args();RUNS.mkdir(exist_ok=True);UPLOADS.mkdir(exist_ok=True)
    for old in jobs():
        if old['status'] in ('running','queued'):
            old['status']='interrupted';old['message']='The app stopped before this run finished. Start a new analysis to retry.'
            atomic_json(RUNS/old['id']/'job.json',old)
    print(f'RootSpan is ready at http://127.0.0.1:{a.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()

if __name__=='__main__':main()
