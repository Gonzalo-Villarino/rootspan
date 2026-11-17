from pathlib import Path
import datetime,json,sys,uuid
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT/'09_Periderm_GUI'))
import server
from version_registry import verify_v2
verify_v2()
now=datetime.datetime.now(datetime.timezone.utc)
ident=now.strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
folder=server.RUNS/ident;folder.mkdir()
job={'id':ident,'image_name':'7_18.tif','scale':.5299,'pipeline':'v2','include_diagnostics':True,'created_at':now.isoformat(),'status':'queued','message':'Rechecking 7_18 with separate PAT cleanup data','results_folder':str(folder/'results')}
server.atomic_json(folder/'job.json',job)
print('NEW_RUN='+ident,flush=True)
server.task(job,PROJECT/'02_Data/16_New_SALK_GUI_Checks_20261117/7_18.tif')
if job['status']!='complete':raise RuntimeError(job['message'])
if 'diagnostic_error' in job['summary']:raise RuntimeError(job['summary']['diagnostic_error'])
(Path(__file__).parent/'new_run.json').write_text(json.dumps({'run':ident,'folder':str(folder),'status':job['status']},indent=2))
print('COMPLETE '+ident,flush=True)
