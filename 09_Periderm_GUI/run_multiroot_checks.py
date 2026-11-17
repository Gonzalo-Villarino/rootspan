"""Reproduce the requested execution checks. No training or human geometry."""
from pathlib import Path
import datetime,hashlib,json,traceback,uuid
import pipeline
from server import atomic_json

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent

def main():
    check=HERE/'checks/multiroot_20261117'
    sources=[
        PROJECT/'02_Data/15_Additional_Slides_20261117/01_Current_Check/1_7.tif',
        PROJECT/'02_Data/15_Additional_Slides_20261117/02_Reserved_Evaluation/22_22.tif',
        *[PROJECT/'02_Data/16_New_SALK_GUI_Checks_20261117'/name
          for name in ['7_18.tif','Col_rep1.tif','WOX4_rep1.tif']]]
    snapshot={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
              for name in ['pipeline.py','root_tracks.py','image_quality.py']}
    plan={'sources':list(map(str,sources)),'pixels_per_micrometre':.5299,
          'calibration':'Confirmed by user for these slides','no_training':True,
          'geometry_version':pipeline.VERSION,'code_before_predictions':snapshot,
          'scope':'Execution and coverage check, not biological accuracy validation',
          'first_two':'development/regression checks','last_three':'new copied slide checks; historical encoder exposure unknown',
          'runs':[]}
    atomic_json(check/'batch_checks.json',plan)
    for source in sources:
        now=datetime.datetime.now(datetime.timezone.utc)
        ident=now.strftime('%Y%m%dT%H%M%SZ')+'_multiroot_'+uuid.uuid4().hex[:8]
        folder=HERE/'runs'/ident;folder.mkdir()
        job={'id':ident,'image_name':source.name,'scale':.5299,'created_at':now.isoformat(),
             'status':'running','message':'Starting whole-slide check',
             'results_folder':str(folder/'results')}
        def report(message):
            job['message']=message;atomic_json(folder/'job.json',job)
            print(source.name+': '+message,flush=True)
        report('Starting whole-slide check')
        try:
            summary=pipeline.run(source,folder/'results',.5299,report)
            job.update(status='complete',summary=summary)
            report('Complete — open the results below.')
            print(json.dumps({'slide':source.name,'root_candidates':len(summary['rows']),
                              'statuses':[r['status'] for r in summary['rows']]}),flush=True)
        except Exception as error:
            (folder/'error.txt').write_text(traceback.format_exc())
            job['status']='failed';report(str(error))
        plan['runs'].append({'slide':source.name,'job_id':ident,'status':job['status']})
        atomic_json(check/'batch_checks.json',plan)
    after={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in snapshot}
    assert after==snapshot,'Processing code changed during checks'
    plan['code_unchanged_during_checks']=True
    atomic_json(check/'batch_checks.json',plan)

if __name__=='__main__':main()
