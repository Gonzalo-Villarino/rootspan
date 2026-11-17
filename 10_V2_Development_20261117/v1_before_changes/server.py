"""Local-only browser GUI. Images and results remain on this Mac."""
from pathlib import Path
import argparse
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

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
RUNS=HERE/'runs'
UPLOADS=HERE/'uploads'
TOKEN=secrets.token_urlsafe(32)
LOCK=threading.Lock()
WORKER=ThreadPoolExecutor(max_workers=1)
ACTIVE=None
Image.MAX_IMAGE_PIXELS=200_000_000


def atomic_json(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

def jobs():
    records=[]
    for p in sorted(RUNS.glob('*/job.json'),reverse=True):
        try:records.append(json.loads(p.read_text()))
        except (OSError,ValueError):continue
    return sorted(records,key=lambda r:r['created_at'],reverse=True)

def task(job,source):
    global ACTIVE
    folder=RUNS/job['id']
    def report(text):
        job['message']=text
        atomic_json(folder/'job.json',job)
    try:
        import pipeline
        job['status']='running';report('Starting local analysis…')
        result=pipeline.run(source,folder/'results',job['scale'],report)
        job['status']='complete';job['summary']=result
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
            return self.json({'job':job,'active':ACTIVE,'version':'0.2.1','local_only':True})
        if route.startswith('/files/'):
            try:
                p=self.safe_path(RUNS,unquote(route.removeprefix('/files/')))
                if not p.is_file():return self.json({'error':'File not found'},404)
                # Only generated figures/tables are downloadable through this route.
                if p.suffix.lower() not in ('.png','.jpg','.json','.csv','.txt'):return self.json({'error':'Not available'},403)
                return self.send(200,p.read_bytes(),mimetypes.guess_type(p.name)[0] or 'application/octet-stream')
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
            ident=request.get('upload','')
            if len(ident)!=32 or any(c not in '0123456789abcdef' for c in ident):raise ValueError('Choose an image first.')
            metadata=json.loads((UPLOADS/(ident+'.json')).read_text())
            source=Path(metadata['path']);name=metadata['name']
            with LOCK:
                if ACTIVE:return self.json({'error':'An image is already being analysed. Wait for it to finish.'},409)
                job_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
                folder=RUNS/job_id;folder.mkdir()
                job={'id':job_id,'image_name':name,'scale':scale,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'status':'queued','message':'Queued for local analysis','results_folder':str(folder/'results')}
                atomic_json(folder/'job.json',job);ACTIVE=job_id
                WORKER.submit(task,job,source)
            return self.json(job,202)
        except Exception as error:return self.json({'error':str(error)},400)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765)
    a=parser.parse_args();RUNS.mkdir(exist_ok=True);UPLOADS.mkdir(exist_ok=True)
    for old in jobs():
        if old['status'] in ('running','queued'):
            old['status']='interrupted';old['message']='The app stopped before this run finished. Start a new analysis to retry.'
            atomic_json(RUNS/old['id']/'job.json',old)
    print(f'Periderm Lab is ready at http://127.0.0.1:{a.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()

if __name__=='__main__':main()
