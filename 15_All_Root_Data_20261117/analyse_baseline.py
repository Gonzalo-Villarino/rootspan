from pathlib import Path
import ast,csv,json,sys
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
Image.MAX_IMAGE_PIXELS=250_000_000
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent
sys.path.insert(0,str(PROJECT/'09_Periderm_GUI'))
from pat_diagnostics import sample_path,cleanup_candidate
BASE=HERE/'original_grid_718'
RESULTS=PROJECT/'09_Periderm_GUI/runs/20261117T143545Z_bc803d80/results'
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
summary=json.loads((RESULTS/'results.json').read_text())
stages={}
for name in ('stitched','processed','binary','closed','measurement'):
    im=np.asarray(Image.open(BASE/f'{name}_source_coordinates.png'))
    if im.ndim==3 and name in ('stitched','processed'):
        # cv2's BGR [128,128,0] is RGB [0,128,128] on disk.
        p=np.all(im==[0,128,128],axis=2)
    else:p=(im.max(axis=2) if im.ndim==3 else im)>127
    stages[name]=p.astype('uint8')
    del im
raw=np.asarray(Image.open(BASE/'stitched_source_coordinates.png'))
_,labels=cv2.connectedComponents(np.any(raw>30,axis=2).astype('uint8'))
rows=[]
for r in summary['rows']:
    full=json.loads((RESULTS/f"track_{r['track_id']:02d}/result.json").read_text())
    path=np.array(full['path_xy']);q=np.floor(path+.5).astype(int)
    ids,counts=np.unique(labels[q[:,1],q[:,0]],return_counts=True)
    mainid=int(ids[np.argmax(np.where(ids==0,0,counts))])
    entry={'root':r['track_id'],'V2_status':r['status'],'dominant_raw_foreground_component':mainid,
           'all_nonzero_foreground_components_on_path':[int(i) for i in ids if i],
           'P_labelled_path_um_by_stage':{},'note':'Fixed V2 path, not anatomical periderm measurement.'}
    for name,mask in stages.items():
        s,samples=sample_path(mask,path,.5299)
        entry['P_labelled_path_um_by_stage'][name]=s['raw_P_labelled_path_um']
    rows.append(entry)
record={'scope':'Compatibility-adjusted original PAT mask stages. Fixed V2 main-root paths used for diagnostics, not original FilFinder measurement.',
        'rows':rows,'expert_root1_periderm_um':0,'anatomical_accuracy_not_established':True,
        'candidate_promoted':False}
write(HERE/'original_postprocessing_comparison.json',record)
with (HERE/'original_postprocessing_comparison.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['root','V2_status','foreground_component']+list(stages))
    for r in rows:w.writerow([r['root'],r['V2_status'],r['dominant_raw_foreground_component']]+list(r['P_labelled_path_um_by_stage'].values()))
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',24)
smallsize=(1400,round(raw.shape[0]*1400/raw.shape[1]));out=Image.new('RGB',(2800,smallsize[1]+135),'white');d=ImageDraw.Draw(out)
source=np.asarray(Image.open(PROJECT/'02_Data/16_New_SALK_GUI_Checks_20261117/7_18.tif'))
for index,name in enumerate(('stitched','measurement')):
    panel=Image.fromarray(source).resize(smallsize)
    pred=Image.fromarray(stages[name]*255).resize(smallsize,Image.Resampling.NEAREST)
    overlay=Image.new('RGB',smallsize,'#e53845');panel=Image.composite(overlay,panel,pred)
    draw=ImageDraw.Draw(panel);sx=smallsize[0]/raw.shape[1];sy=smallsize[1]/raw.shape[0]
    for r in summary['rows']:
        full=json.loads((RESULTS/f"track_{r['track_id']:02d}/result.json").read_text());p=np.array(full['path_xy']);x,y=p[len(p)//2]*[sx,sy]
        draw.text((x,y-25),str(r['track_id']),font=font,fill='white',stroke_width=2,stroke_fill='black')
    out.paste(panel,(index*1400,90));d.text((index*1400+15,15),'Raw model Periderm pixels' if index==0 else 'After original PAT post-processing',font=font,fill='black')
d.text((15,smallsize[1]+100),'Red = predicted/processed mask, not expert ground truth. Original-grid baseline; runtime adaptations recorded.',font=font,fill='black')
out.save(HERE/'POSTPROCESSING_BEFORE_AFTER.png')
print(json.dumps(record,indent=2))
