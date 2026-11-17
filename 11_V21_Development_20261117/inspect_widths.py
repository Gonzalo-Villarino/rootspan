"""Diagnostic widths from fluorescence cross sections; no endpoint labels invented."""
from pathlib import Path
import sys,json
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
Image.MAX_IMAGE_PIXELS=200_000_000

def widths(rgb,path,arc):
    d=np.arange(80,arc[-1]-80,16.)
    centres=np.c_[np.interp(d,arc,path[:,0]),np.interp(d,arc,path[:,1])]
    prev=np.c_[np.interp(d-64,arc,path[:,0]),np.interp(d-64,arc,path[:,1])]
    nxt=np.c_[np.interp(d+64,arc,path[:,0]),np.interp(d+64,arc,path[:,1])]
    tangent=nxt-prev;tangent/=np.linalg.norm(tangent,axis=1)[:,None];normal=np.c_[-tangent[:,1],tangent[:,0]]
    n=np.arange(-256,257,2);samples=[]
    for shift in [-8,-4,0,4,8]:
        xy=centres[:,None,:]+normal[:,None,:]*n[None,:,None]+tangent[:,None,:]*shift
        samples.append(ndi.map_coordinates(rgb,[xy[:,:,1],xy[:,:,0]],order=1,mode='constant'))
    profile=ndi.gaussian_filter1d(np.mean(samples,axis=0),1,axis=1)
    ww=[];sides=[];peaks=[]
    for v in profile:
        peak=np.percentile(v[np.abs(n)<=80],95)
        threshold=max(5.,.16*peak)
        mask=ndi.binary_closing(v>threshold,iterations=5)
        labels,count=ndi.label(mask);runs=[]
        for k in range(1,count+1):
            idx=np.flatnonzero(labels==k);lo,hi=n[idx[0]],n[idx[-1]]
            distance=max(lo,0,-hi)
            if distance<=24:runs.append((distance,lo,hi))
        if runs:
            _,lo,hi=min(runs);ww.append(float(hi-lo));sides.append([float(lo),float(hi)])
        else:ww.append(0.);sides.append([0.,0.])
        peaks.append(float(peak))
    return {'distance':d.tolist(),'width':ww,'sides':sides,'peak':peaks,'centres':centres.tolist()}

def main():
    source=ROOT/'02_Data/16_New_SALK_GUI_Checks_20261117/7_18.tif'
    green=np.array(Image.open(source).getchannel('G'),dtype=np.float32)
    base=ROOT/'10_V2_Development_20261117/diagnostics/v2_full_context_7_18'
    fig,axes=plt.subplots(7,1,figsize=(14,19),constrained_layout=True)
    out=Path(__file__).resolve().parent
    for i,ax in enumerate(axes,1):
        folder=base/f'track_{i:02d}';r=json.loads((folder/'route.json').read_text());q=json.loads((folder/'prediction.json').read_text());path=np.array(r['path_xy_crop_pixels'])+r['source_crop_xyxy'][:2];arc=np.array(r['distance_along_path_pixels'])
        result=widths(green,path,arc);(out/f'root{i}_widths.json').write_text(json.dumps(result,indent=2)+'\n')
        d=np.array(result['distance']);w=np.array(result['width']);smooth=ndi.median_filter(w,size=13)
        ax.plot(d,w,alpha=.2,color='grey');ax.plot(d,smooth);ax.axvline(r['endpoint']['measurement_distance_pixels'],color='orange',label='V2 J')
        if q['T_path_pixels'] is not None:ax.axvline(q['T_path_pixels'],color='deepskyblue',label='T')
        ax.set(xlim=(arc[-1]-5000,arc[-1]),ylim=(0,400),title=f'Root {i}',ylabel='Width (px)');ax.legend()
    fig.savefig(out/'normal_widths.png',dpi=110)

if __name__=='__main__':main()
