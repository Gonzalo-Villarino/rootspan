"""Periderm-length proof of concept: image-only predictions -> main-root paths -> lengths.

The automatic basal endpoint is the right-hand end of each detected main-root
component. Its anatomical identity MUST be reviewed. Length follows a smoothed
centreline, not a straight end-to-end chord or a count of foreground pixels.
The150px gap rule is inspired by PAT; this implementation is not identical PAT.
"""
from pathlib import Path
import argparse,csv,hashlib,html,json,math,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize
Image.MAX_IMAGE_PIXELS=250_000_000
SCALE=4
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def arc_length(path):return np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
def pixels_to_um(length,px_per_um):
 if not np.isfinite(px_per_um) or px_per_um<=0:raise ValueError('Calibration must be positive pixels per micrometre.')
 return float(length)/px_per_um
def close_short_gaps(flags,arc,max_gap=150):
 flags=flags.copy();runs,n=ndi.label(~flags)
 for i in range(1,n+1):
  ix=np.where(runs==i)[0];a,b=ix[0],ix[-1]
  if a>0 and b<len(flags)-1 and flags[a-1] and flags[b+1] and arc[b+1]-arc[a-1]<=max_gap:flags[ix]=True
 return flags
def skeleton_path(component):
 sk=skeletonize(component);yx=np.argwhere(sk);idx=np.full(sk.shape,-1,dtype='int32');idx[sk]=np.arange(len(yx));rows=[];cols=[];vals=[]
 for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
  yy=yx[:,0]+dy;xx=yx[:,1]+dx;v=(yy>=0)&(xx>=0)&(yy<sk.shape[0])&(xx<sk.shape[1]);aa=np.where(v)[0];bb=idx[yy[v],xx[v]];v2=bb>=0;aa=aa[v2];bb=bb[v2];rows.extend([aa,bb]);cols.extend([bb,aa]);vals.extend([np.full(len(aa),math.hypot(dy,dx))]*2)
 graph=csr_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(len(yx),len(yx)));degree=np.diff(graph.indptr);ends=np.where(degree==1)[0]
 if len(ends)<2:raise ValueError('Skeleton does not have two endpoints; possible loop or merged roots.')
 left=int(ends[np.argmin(yx[ends,1])]);right=int(ends[np.argmax(yx[ends,1])]);dist,prev=dijkstra(graph,directed=False,indices=left,return_predecessors=True)
 if not np.isfinite(dist[right]):raise ValueError('Disconnected skeleton.')
 path=[right]
 while path[-1]!=left:
  path.append(int(prev[path[-1]]))
  if path[-1]<0 or len(path)>len(yx):raise ValueError('Invalid centreline path')
 xy=yx[path[::-1],::-1].astype('float64')
 # Extend each skeleton endpoint to the component edge along its local tangent.
 def extend(p,at_end):
  origin=p[-1] if at_end else p[0];other=p[max(0,len(p)-21)] if at_end else p[min(20,len(p)-1)];direction=origin-other;direction/=max(1e-9,np.linalg.norm(direction));last=origin.copy()
  for length in np.arange(.25,100,.25):
   q=origin+direction*length;x,y=np.rint(q).astype(int)
   if x<0 or y<0 or y>=component.shape[0] or x>=component.shape[1] or not component[y,x]:break
   last=q
  return last
 xy=np.vstack([extend(xy,False),xy,extend(xy,True)])
 window=min(51,len(xy)//2*2-1)
 if window>=5:
  smooth=savgol_filter(xy,window,2,axis=0,mode='interp');smooth[0]=xy[0];smooth[-1]=xy[-1];xy=smooth
 return xy,{'skeleton_endpoints':int(len(ends)),'raw_graph_main_path_length_native_px':float(dist[right]*SCALE)}
def font(n):return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',n)
def run(prediction,output,px_per_um=None,calibration_status='unconfirmed',expected_roots=None,opening=4,transition_method="single-change"):
 if px_per_um is not None:pixels_to_um(1,px_per_um)
 if output.exists():raise FileExistsError('New output folder required; previous measurements preserved.')
 meta=json.loads((prediction/'prediction_record.json').read_text());source=Path(meta['source_image']);assert sha(source)==meta['source_sha256']
 mask=np.array(Image.open(prediction/'raw_classes.png'));qmap=np.array(Image.open(prediction/'periderm_fraction_scores.png')).astype('float32')/65535;rgb=Image.open(source).convert('RGB');w,h=rgb.size
 # Topology detection at quarter resolution; all exported coordinates return to native pixels.
 small=np.isin(mask[::SCALE,::SCALE],[1,2]);small=ndi.binary_opening(small,iterations=opening) if opening else small;small=ndi.binary_closing(small,iterations=2);lab,n=ndi.label(small);slices=ndi.find_objects(lab);objects=[]
 for i,s in enumerate(slices,1):
  if s is None:continue
  area=int((lab[s]==i).sum());width=s[1].stop-s[1].start
  if width*SCALE>=max(2000,.35*w) and area>=1000:objects.append((i,s,area))
 objects.sort(key=lambda z:(z[1][0].start+z[1][0].stop)/2)
 output.mkdir(parents=True);(output/'roots').mkdir();shutil.copy2(__file__,output/Path(__file__).name);records=[];skipped=[]
 for rid,(label,s,area) in enumerate(objects,1):
  # Pad local component so endpoint extension can safely reach its edge.
  local=np.pad(lab[s]==label,3);offset=np.array([s[1].start-3,s[0].start-3]);flags=[]
  try:xy,info=skeleton_path(local)
  except ValueError as e:skipped.append({'component':label,'reason':str(e)});continue
  radii=ndi.distance_transform_edt(local);xy=(xy+offset)*SCALE;arc=arc_length(xy);positions=np.r_[np.arange(0,arc[-1],8.),arc[-1]];path=np.column_stack([np.interp(positions,arc,xy[:,j]) for j in [0,1]]);arc=arc_length(path)
  tangent=np.gradient(path,axis=0);norm=np.linalg.norm(tangent,axis=1);tangent/=np.maximum(norm[:,None],1e-8);normal=np.column_stack([-tangent[:,1],tangent[:,0]])
  localxy=path/SCALE-offset;rad=ndi.map_coordinates(radii,[localxy[:,1],localxy[:,0]],order=1,mode='nearest')*SCALE;values=[]
  for fraction in [-.4,-.2,0,.2,.4]:
   coords=path+normal*(rad*fraction)[:,None];values.append(ndi.map_coordinates(qmap,[coords[:,1],coords[:,0]],order=1,mode='nearest'))
  score=np.mean(values,axis=0);score=ndi.gaussian_filter1d(score,3,mode='nearest');is_p=close_short_gaps(score>=.5,arc,150);runlabels,nruns=ndi.label(is_p);intervals=[]
  for k in range(1,nruns+1):
   ix=np.where(runlabels==k)[0];intervals.append((int(ix[0]),int(ix[-1])))
  length=None;start=None
  if not intervals:flags.append('no_periderm_segment_detected')
  else:
   a,b=intervals[-1];basal_gap=float(arc[-1]-arc[b])
   if basal_gap>250:flags.append('periderm_does_not_reach_automatic_basal_endpoint')
   else:
    start=a;length=float(arc[-1]-arc[start])
    if len(intervals)>1:flags.append('separate_distal_periderm_islands_discarded')
    if a==0:flags.append('transition_not_visible_or_entire_root_called_periderm')
    if basal_gap>32:flags.append('basal_end_has_low_periderm_support')
  if transition_method=='single-change':
   # Explicit anatomical prior: one E-left to P-right change on this primary root.
   # This is post-processing, not additional learning or biological verification.
   eps=1e-5;pp=np.clip(score,eps,1-eps);de=-np.log1p(-pp);dp=-np.log(pp)
   cost=np.r_[0,np.cumsum(de)] + np.r_[np.cumsum(dp[::-1])[::-1],0]
   boundary=int(np.argmin(cost))
   flags=[f for f in flags if f not in ['no_periderm_segment_detected','periderm_does_not_reach_automatic_basal_endpoint','separate_distal_periderm_islands_discarded','transition_not_visible_or_entire_root_called_periderm','basal_end_has_low_periderm_support']]
   if boundary>=len(path):start=None;length=None;flags.append('no_periderm_under_single_change_rule')
   else:
    start=boundary;length=float(arc[-1]-arc[start])
    if boundary==0:flags.append('transition_not_visible_or_entire_root_called_periderm')
   flags.append('single_E_to_P_change_assumed')
  if path[-1,0]>w-12 or path[-1,1]<12 or path[-1,1]>h-12:flags.append('basal_endpoint_near_image_border')
  if np.mean(np.abs(score-.5)<.1)>.2:flags.append('many_ambiguous_tissue_scores')
  if expected_roots is not None and len(objects)!=expected_roots:flags.append('detected_root_count_differs_from_expected')
  flags.append('biological_endpoint_and_transition_review_pending')
  transition=path[start].tolist() if start is not None else None
  rec={'root_id':rid,'component_label':label,'periderm_length_px':length,'periderm_length_um':pixels_to_um(length,px_per_um) if length is not None and px_per_um else None,'whole_visible_main_path_length_px':float(arc[-1]),'automatic_transition_xy':transition,'automatic_basal_xy':path[-1].tolist(),'path_xy_native_pixels':path.tolist(),'arc_positions_px':arc.tolist(),'periderm_start_index':start,'periderm_fraction_score':score.tolist(),'flags':flags,'quality_control':'pending','transition_method':transition_method,'manual_length_px':None,'manual_length_um':None,**info};records.append(rec)
  # Original fluorescence with measured line; no recolouring of biological tissue.
  bounds=[max(0,int(path[:,0].min())-160),max(0,int(path[:,1].min())-220),min(w,int(path[:,0].max())+160),min(h,int(path[:,1].max())+220)]
  region=rgb.crop(bounds);scale=min(1,1800/region.width);sz=(round(region.width*scale),round(region.height*scale));region=region.resize(sz,Image.Resampling.LANCZOS);d=ImageDraw.Draw(region)
  def pts(arr):return [((float(x)-bounds[0])*scale,(float(y)-bounds[1])*scale) for x,y in arr]
  if start is not None:d.line(pts(path[start:]),fill='#00e3f0',width=4)
  for labeltext,point,color in [('B',path[-1],'white')]+([('T',path[start],'#ff53e0')] if start is not None else []):
   x,y=pts([point])[0];d.ellipse((x-7,y-7,x+7,y+7),outline=color,width=3);d.text((x+10,y-22),labeltext,font=font(23),fill=color,stroke_width=1,stroke_fill='black')
  canvas=Image.new('RGB',(max(1000,region.width),region.height+105),'white');dr=ImageDraw.Draw(canvas);txt=f'Root {rid}: automatic length '+(f'{length:,.1f} pixels' if length is not None else 'NOT MEASURABLE')
  if length is not None and px_per_um:txt+=f' | {pixels_to_um(length,px_per_um):,.1f} micrometres'
  dr.text((15,12),txt,font=font(26),fill='black');dr.text((15,49),'Cyan = measured centreline; T = predicted transition; B = automatic basal endpoint. QC pending.',font=font(21),fill='#374151');canvas.paste(region,(0,95));canvas.save(output/'roots'/f'root_{rid}_length.png')
  if transition:
   tx,ty=transition;box=[max(0,int(tx)-512),max(0,int(ty)-256),min(w,int(tx)+512),min(h,int(ty)+256)];close=rgb.crop(box);dr=ImageDraw.Draw(close);cx,cy=tx-box[0],ty-box[1];dr.ellipse((cx-10,cy-10,cx+10,cy+10),outline='#ff53e0',width=3);dr.line((cx,0,cx,close.height),fill='#ff53e0',width=2);close.save(output/'roots'/f'root_{rid}_transition.png')
 # Full-slide overview, independent of reference labels.
 scale=min(1800/w,1300/h);overview=rgb.resize((round(w*scale),round(h*scale)),Image.Resampling.LANCZOS);d=ImageDraw.Draw(overview)
 for r in records:
  path=np.array(r['path_xy_native_pixels']);start=r['periderm_start_index']
  if start is not None:
   p=[tuple(q*scale) for q in path[start:]];d.line(p,fill='#00e3f0',width=3);x,y=p[0];d.ellipse((x-6,y-6,x+6,y+6),fill='#ff53e0')
  x,y=path[len(path)//2]*scale;d.text((x,y-28),str(r['root_id']),font=font(26),fill='white',stroke_width=2,stroke_fill='black')
 overview.save(output/'ALL_ROOTS_MEASUREMENT_LINES.png')
 fields=['root_id','automatic_periderm_length_px','automatic_periderm_length_um','pixels_per_um','calibration_status','quality_control','flags','manual_length_px','manual_length_um','manual_minus_automatic_px','manual_minus_automatic_um']
 with (output/'periderm_lengths_for_manual_comparison.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for r in records:writer.writerow({'root_id':r['root_id'],'automatic_periderm_length_px':round(r['periderm_length_px'],3) if r['periderm_length_px'] is not None else '', 'automatic_periderm_length_um':round(r['periderm_length_um'],3) if r['periderm_length_um'] is not None else '', 'pixels_per_um':px_per_um or '', 'calibration_status':calibration_status,'quality_control':'pending','flags':'; '.join(r['flags'])})
 record={'source':meta,'measurement_method':'Horizontal main-root skeleton path, quarter-resolution topology, smooth native-coordinate polyline and core tissue scores. Length = sum Euclidean distances along trace. Not exact original PAT implementation.','morphology_opening_iterations_at_quarter_scale':opening,'transition_method':transition_method,'transition_rule_description':('Choose one E-left/P-right change minimizing negative log core tissue scores; no reference annotations used. This anatomical prior can still give an incorrect transition.' if transition_method=='single-change' else 'Close enclosed gaps <=150px and retain basal-connected Periderm.'),'anatomical_assumption':'Basal/hypocotyl end is on the right; requires review.','pixels_per_um':px_per_um,'calibration_status':calibration_status,'detected_large_components':len(objects),'measured_root_records':len(records),'expected_roots':expected_roots,'skipped_components':skipped,'results_are_provisional':True,'annotations_used':False,'roots':records}
 (output/'measurement_record.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');assert sha(source)==meta['source_sha256']
 print(json.dumps({'detected':len(objects),'records':len(records),'lengths_px':[r['periderm_length_px'] for r in records],'flags':[r['flags'] for r in records]},indent=2),flush=True)
 return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prediction',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--pixels-per-um',type=float);p.add_argument('--calibration-status',default='unconfirmed; pixel lengths only');p.add_argument('--expected-roots',type=int);p.add_argument('--opening',type=int,default=4);p.add_argument('--transition-method',choices=['single-change','basal-run'],default='single-change');a=p.parse_args();run(a.prediction,a.output,a.pixels_per_um,a.calibration_status,a.expected_roots,a.opening,a.transition_method)
