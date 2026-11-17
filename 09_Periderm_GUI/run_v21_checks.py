"""Replay V2.1 execution checks on the five development slides; no training.

This measures computational reproduction and coverage, not biological accuracy.
Reserved slide 15_27 is not used. Outputs always go to a new timestamped folder.
"""
from pathlib import Path
import argparse,datetime,json
import pipeline_v21
from version_registry import verify_v21,PROJECT

SOURCES={
 '1_7':'02_Data/15_Additional_Slides_20261117/01_Current_Check/1_7.tif',
 '22_22':'02_Data/15_Additional_Slides_20261117/02_Reserved_Evaluation/22_22.tif',
 **{s:'02_Data/16_New_SALK_GUI_Checks_20261117/'+s+'.tif'
    for s in ['7_18','Col_rep1','WOX4_rep1']}}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--slide',choices=list(SOURCES),help='Omit to check all five.')
    p.add_argument('--output',type=Path)
    args=p.parse_args();verify_v21()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=args.output or PROJECT/'11_V21_Development_20261117/replays'/stamp
    out.mkdir(parents=True,exist_ok=False)
    names=[args.slide] if args.slide else list(SOURCES)
    report={'training':False,'biological_accuracy_validated':False,'runs':[]}
    for name in names:
        s=pipeline_v21.run(PROJECT/SOURCES[name],out/name,.5299,
                         lambda m:print(name+': '+m,flush=True))
        report['runs'].append({'slide':name,'root_candidates':s['detected_groups'],
                              'provisional_lengths':s['provisional_lengths'],
                              'statuses':[r['status'] for r in s['rows']]})
        (out/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
    verify_v21();print('Saved:',out)

if __name__=='__main__':main()
