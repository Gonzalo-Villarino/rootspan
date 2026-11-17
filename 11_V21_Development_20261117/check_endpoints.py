from pathlib import Path
import sys,json,numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'09_Periderm_GUI'))
from endpoints_v21 import proximal_endpoint
from pipeline_v2 import font
Image.MAX_IMAGE_PIXELS=200_000_000

def check(source,base,out):
    out.mkdir(exist_ok=False);im=Image.open(source).convert('RGB');green=np.array(im.getchannel('G'))
    overview=im.copy();overview.thumbnail((1800,1400));ratio=overview.width/im.width;draw=ImageDraw.Draw(overview);rows=[]
    for folder in sorted(base.glob('track_*')):
        if not (folder/'route.json').exists():continue
        r=json.loads((folder/'route.json').read_text());path=np.array(r['path_xy_crop_pixels'])+r['source_crop_xyxy'][:2];arc=np.array(r['distance_along_path_pixels']);old=r['endpoint']
        e,a=proximal_endpoint(green,path,arc,old);xy=path[e['measurement_path_index']]
        row={'root':int(folder.name[-2:]),'endpoint':e,'source_xy':xy.tolist(),'audit':a};rows.append(row)
        x,y=xy*ratio;draw.ellipse((x-5,y-5,x+5,y+5),fill='orange');draw.text((x+8,y-25),str(row['root']),font=font(22),fill='white')
        print(source.name,row['root'],xy.tolist(),e['endpoint_method'],flush=True)
    overview.save(out/'endpoints.jpg');(out/'endpoints.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':
    out=ROOT/'11_V21_Development_20261117'
    check(ROOT/'02_Data/16_New_SALK_GUI_Checks_20261117/7_18.tif',out/'intensity_tracks',out/'endpoint_trial_7_18')
    check(ROOT/'02_Data/15_Additional_Slides_20261117/01_Current_Check/1_7.tif',ROOT/'10_V2_Development_20261117/regression/1_7',out/'endpoint_trial_1_7')
