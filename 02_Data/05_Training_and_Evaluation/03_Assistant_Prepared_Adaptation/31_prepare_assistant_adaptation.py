"""Assistant-prepared sparse labels based on Gonzalo's pre-prediction biology.

These are development/training annotations, NOT independent ground truth.
No model predictions are used to draw them. Ambiguous transition and lateral
branches remain unlabelled. Source pixels and user annotation files are untouched.
"""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'02_Data/05_Training_and_Evaluation/03_Assistant_Prepared_Adaptation'
PAL=np.zeros((256,3),dtype='uint8');PAL[0]=[48,52,59];PAL[1]=[219,48,63];PAL[2]=[26,166,94];PAL[3]=[255,220,58];PAL[255]=[200,70,200]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 OUT.mkdir(exist_ok=False)
 specs=[
 ('20_29_clearer',ROOT/'04_Results/43_20_29_Clearer_Transition_Prediction/view_20261115T155056Z_vbrbuhdb/20_29_x5440_y4256_image.png',[
  ('Endoderm',[(15,355),(80,376),(160,396),(240,415),(320,428),(390,440),(390,463),(320,448),(240,436),(160,421),(80,402),(15,383)]),
  ('Periderm',[(725,519),(790,534),(870,546),(950,547),(1005,553),(1005,574),(950,566),(870,568),(790,557),(725,547)]),
  ('Background',[(20,180),(800,180),(800,330),(20,290)]),
  ('Background',[(25,520),(400,560),(440,900),(25,900)]),
  ('Background',[(760,660),(1000,660),(1000,960),(760,960)])],
  'Before seeing prediction, Gonzalo confirmed: clearer, Endoderm left and Periderm right; transition around middle. Exact sparse polygons drawn by assistant, not biologically certified by Gonzalo. Transition x391-724 and lateral branch not labelled.'),
 ('20_29_periderm',ROOT/'04_Results/41_New_Image_20_29_Prediction/view_20261115T152946Z_kc80hx_h/20_29_x7988_y5618_image.png',[
  ('Periderm',[(25,580),(130,555),(250,530),(375,506),(490,485),(600,464),(650,455),(650,483),(600,499),(490,522),(375,546),(250,567),(130,594),(25,617)]),
  ('Periderm',[(800,423),(900,421),(1000,423),(1000,463),(900,452),(800,462)]),
  ('Background',[(30,180),(570,180),(570,390),(30,450)]),
  ('Background',[(200,680),(1000,580),(1000,900),(200,900)])],
  'Gonzalo confirmed the main root is Periderm from the cell shapes. Exact sparse polygons drawn by assistant. Upward lateral branch and other tissue excluded.')]
 manifest={'purpose':'Sparse assistant-prepared training labels; not an independent test or human-verified pixel ground truth.','class_codes':{'Periderm':1,'Endoderm':2,'Background':3},'ignore_codes':[0,255],'original_user_files_modified':False,'samples':[]}
 for name,source,shapes,basis in specs:
  folder=OUT/name;folder.mkdir();im=Image.open(source).convert('RGB');shutil.copy2(source,folder/'image.png');rgb=np.array(im)
  ann={'version':'5.0.0','flags':{},'shapes':[],'imagePath':'image.png','imageData':None,'imageHeight':1024,'imageWidth':1024,'annotation_provenance':{'author':'programmatic annotation tool','status':'Assistant-prepared development labels','biological_basis':basis,'prediction_masks_used':False}}
  mask=np.zeros((1024,1024),dtype='uint8')
  for label,pts in shapes:
   code=manifest['class_codes'][label];region=Image.new('1',im.size);ImageDraw.Draw(region).polygon(pts,fill=1);inside=np.array(region,dtype=bool);conflict=inside&(mask!=0)&(mask!=code);mask[inside&(mask==0)]=code;mask[conflict]=255
   ann['shapes'].append({'label':label,'points':pts,'group_id':None,'description':'Assistant-prepared sparse training example; not independent reference.','shape_type':'polygon','flags':{}})
  assert not (mask==255).any()
  (folder/'annotations_ASSISTANT.json').write_text(json.dumps(ann,indent=2)+'\n');Image.fromarray(mask).save(folder/'mask.png');Image.fromarray(PAL[mask]).save(folder/'mask_preview.png')
  overlay=im.copy();d=ImageDraw.Draw(overlay)
  for label,pts in shapes:d.line(pts+[pts[0]],fill=tuple(PAL[manifest['class_codes'][label]]),width=3)
  panel=Image.new('RGB',(1576,690),'white');dr=ImageDraw.Draw(panel);font=lambda n:ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',n)
  dr.text((15,12),name+': sparse training examples prepared by the assistant',font=font(27),fill='black')
  for j,(title,pic) in enumerate([('Original image',im),('Assistant polygon outlines',overlay),('Sparse answer key; grey = ignored',Image.fromarray(PAL[mask]))]):
   x=15+j*520;dr.text((x,55),title,font=font(20),fill='black');panel.paste(pic.resize((510,510),Image.Resampling.LANCZOS if j<2 else Image.Resampling.NEAREST),(x,88))
  dr.text((15,615),'Red = Periderm; green = Endoderm; yellow = Background. The uncertain transition and lateral branch are not labelled.',font=font(22),fill='black')
  dr.text((15,650),'These labels teach the model. Agreement with them is training fit, not independent accuracy.',font=font(22),fill='black');panel.save(folder/'REVIEW_assistant_annotations.png')
  manifest['samples'].append({'name':name,'source_image':str(source),'source_sha256':sha(source),'image':str((folder/'image.png').relative_to(OUT)),'mask':str((folder/'mask.png').relative_to(OUT)),'annotations':str((folder/'annotations_ASSISTANT.json').relative_to(OUT)),'mask_sha256':sha(folder/'mask.png'),'counts':{str(c):int((mask==c).sum()) for c in [0,1,2,3,255]},'basis':basis})
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');shutil.copy2(__file__,OUT/Path(__file__).name)
 print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
