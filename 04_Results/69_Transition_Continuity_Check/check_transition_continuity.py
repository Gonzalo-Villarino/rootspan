"""Compare one specified decoder guard on saved predictions; no new inference."""
from pathlib import Path
from datetime import datetime,timezone
import json
import hashlib
import shutil
import numpy as np
from PIL import Image,ImageDraw
from appearance_experiment import ROOT,font
from transition_continuity_rule import continuity_rule,run_checks

OUT=ROOT/'04_Results/69_Transition_Continuity_Check'
def read(p):return json.loads(p.read_text())
def save(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    assert not (OUT/'comparison_results.json').exists()
    run_checks();plan=read(OUT/'experiment_plan.json');results=[]
    manifest=read(ROOT/'06_Models/20261116_Two_Appearance_Candidate/candidate_manifest.json')
    protected=[Path(manifest['head']),Path(manifest['encoder']),ROOT/'06_Models/working_model.pt']
    before={str(p):sha(p) for p in protected}
    for case in plan['inputs']:
        path=Path(case['prediction']);assert sha(path)==case['sha256'];p=read(path)
        scores=np.array(p.get('presence_scores',p.get('predicted_presence_scores')))
        distance=np.array(p.get('distance_along_local_path',p.get('distance_native_pixels')))
        new=continuity_rule(scores,distance)
        assert new['original_rule']==p['rule'],'Baseline replay differs from saved rule.'
        oldT=None if p['rule']['selected'] is None else p['rule']['selected']['T_distance_pixels']
        newT=None if new['selected'] is None else new['selected']['T_distance_pixels']
        row={'case':case['case'],'old_T_path_pixels':oldT,'new_T_path_pixels':newT,
             'same_T':newT==oldT,'candidate_rule_output':new,'input_sha256':sha(path),
             'scores_unchanged':True,'training':False}
        if (path.parent/'image_only_route.json').exists():
            r=read(path.parent/'image_only_route.json')
            pts=np.array(r.get('path_xy_crop_pixels',r.get('path_xy_image_pixels')))
            arc=np.array(r['distance_along_path_pixels']);offset=np.array(r.get('source_crop_xyxy',[0,0])[:2])
            row['new_T_source_xy']=None if newT is None else [float(np.interp(newT,arc,pts[:,j])+offset[j]) for j in range(2)]
            row['old_T_source_xy']=None if oldT is None else [float(np.interp(oldT,arc,pts[:,j])+offset[j]) for j in range(2)]
        results.append(row)
        save(OUT/f'{case["case"]}_decoder_result.json',row)
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),'results':results,
             'previously_accepted_T_preserved':all(r['same_T'] for r in results if r['case'] in ['COL0_root3','A04_root3']),
             '20_29_abstention_preserved':next(r for r in results if r['case']=='20_29_root4')['new_T_path_pixels'] is None,
             'no_neural_inference_or_training':True,'default_not_replaced':True,
             'evaluation_status':'Development experiment after2_12 feedback, not independent validation.'}
    save(OUT/'comparison_results.json',summary)
    assert before=={str(p):sha(p) for p in protected}
    assert all(sha(Path(c['prediction']))==c['sha256'] for c in plan['inputs'])
    save(OUT/'verification.json',{'original_prediction_hashes_unchanged':plan['inputs'],
         'model_weight_hashes_unchanged':before,'synthetic_checks_passed':True,
         'original_rule_reproduced_exactly_on_all_four_saved_profiles':True,
         'figures_visually_checked':False})
    for name in [Path(__file__).name,'transition_continuity_rule.py','test_both_appearances.py','appearance_experiment.py']:
        shutil.copy2(Path(__file__).with_name(name),OUT/name)
    row=next(r for r in results if r['case']=='2_12_root1')
    base=ROOT/'04_Results/68_Fixed_212_Root1_Check';p=read(base/'prediction_before_comparison.json');rgb=Image.open(base/'image.png')
    offset=np.array(p['source_crop_xyxy'][:2]);newxy=row['new_T_source_xy'];oldxy=row['old_T_source_xy']
    fig=Image.new('RGB',(1840,1140),'white');d=ImageDraw.Draw(fig)
    d.text((20,14),'2_12 root 1: checking continuity moves the candidate T',font=font(32),fill='black')
    d.text((20,59),'Same image scores and learned model. Only the rule for choosing T has changed.',font=font(25),fill='#334155')
    fig.paste(rgb.resize((1800,360),Image.Resampling.LANCZOS),(20,105));d=ImageDraw.Draw(fig)
    b=[6700,60,7100,340]
    d.rectangle((20+(b[0]-offset[0])*.24,105+b[1]*.24,20+(b[2]-offset[0])*.24,105+b[3]*.24),outline='white',width=2)
    d.text((20+(b[0]-offset[0])*.24+10,110),'B',font=font(27),fill='white',stroke_width=1,stroke_fill='black')
    # Both T markers remain blue: dashed old, solid new; no manual relocation.
    for xy,label,dashed in [(oldxy,'Old T',True),(newxy,'New candidate T',False)]:
        if xy is None:continue
        x=20+(xy[0]-offset[0])*.24
        if dashed:
            for y in range(105,465,20):d.line((x,y,x,min(y+10,465)),fill='#28b7ff',width=3)
        else:d.line((x,105,x,465),fill='#28b7ff',width=4)
        d.text((x+8,157),label,font=font(24),fill='#28b7ff',stroke_width=1,stroke_fill='black')
    colors=['#1aa65e','#b44dc5','#db303f','#777777']
    for xy,s in zip(p['centers_xy_crop_pixels'],p['rule']['window_states']):
        x=20+xy[0]*.24;d.rectangle((x-7,487,x+7,516),fill=colors[s])
    d.text((20,533),'Unchanged calls: green = Endoderm only; purple = both appearances; red = Periderm only.',font=font(25),fill='black')
    d.text((20,576),'Original cells at the new candidate (blue line)',font=font(27),fill='black')
    if newxy is not None:
        cx,cy=newxy;bounds=[round(cx)-750,max(0,round(cy)-210),round(cx)+750,max(0,round(cy)-210)+420]
        panel=rgb.crop((bounds[0]-offset[0],bounds[1]-offset[1],bounds[2]-offset[0],bounds[3]-offset[1])).resize((1500,420),Image.Resampling.NEAREST)
        fig.paste(panel,(170,620));d=ImageDraw.Draw(fig);x=170+cx-bounds[0]
        d.line((x,620,x,1040),fill='#28b7ff',width=4)
        d.text((x+10,635),'New candidate T',font=font(24),fill='#28b7ff',stroke_width=1,stroke_fill='black')
        save(OUT/'212_closeup_box.json',{'source_box_xyxy':bounds,'new_T_source_xy':newxy,'native_pixel_display':True})
    d.text((20,1062),'Prior accepted COL0 and A04 T positions: '+('unchanged.' if summary['previously_accepted_T_preserved'] else 'CHANGED; inspect regression.'),font=font(25),fill='#075985')
    d.text((20,1100),'The new 2_12 position needs biological review. No precise T-error or automatic length accuracy is claimed.',font=font(23),fill='#334155')
    fig.save(OUT/'212_CONTINUITY_CANDIDATE.png')
    print(json.dumps({'accepted_positions_preserved':summary['previously_accepted_T_preserved'],
          '20_29_no_T_preserved':summary['20_29_abstention_preserved'],
          '2_12_old_xy':oldxy,'2_12_new_xy':newxy,
          'cases':[{'case':r['case'],'oldT':r['old_T_path_pixels'],'newT':r['new_T_path_pixels']} for r in results]},indent=2))

if __name__=='__main__':main()
