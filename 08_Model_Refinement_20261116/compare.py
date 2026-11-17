from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from check_reference import metrics
from predict import PAL
ROOT=Path(__file__).resolve().parents[1];REF=ROOT/'04_Results/52_Reference_Check_20261116'
def comparison(pred,path,title):
 rgb=np.array(Image.open(REF/'image.png'));truth=np.array(Image.open(REF/'reference_mask.png'));old=np.array(Image.open(ROOT/'04_Results/44_Inference_Diagnostics/20_29_clearer/native1024.png'))
 box=(0,310,1024,620);w=768;h=round((box[3]-box[1])*w/1024);canvas=Image.new('RGB',(1586,2*(h+60)+150),'white');d=ImageDraw.Draw(canvas);font=lambda n:ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',n)
 d.text((20,12),title,font=font(27),fill='black')
 for i,(name,a) in enumerate([('1 Microscopy image',rgb),('2 Your reference labels',PAL[truth]),('3 Original model',PAL[old]),('4 Candidate model raw prediction',PAL[pred])]):
  x=20+(i%2)*783;y=65+(i//2)*(h+60);d.text((x,y),name,font=font(23),fill='black');canvas.paste(Image.fromarray(a).crop(box).resize((w,h),Image.Resampling.LANCZOS if i==0 else Image.Resampling.NEAREST),(x,y+35))
 d.text((20,canvas.height-55),'Red Periderm | Green Endoderm | Yellow Background | Purple Lateral root | Grey unlabelled',font=font(20),fill='black');d.text((20,canvas.height-28),'Same image crop in every panel. Model input = image only. Development comparison, not a blind test.',font=font(19),fill='black');canvas.save(path)
def main():
 a=argparse.ArgumentParser();a.add_argument('--folder',type=Path,required=True);x=a.parse_args();p=np.array(Image.open(x.folder/'best_prediction.png'));comparison(p,x.folder/'COMPARISON.png',x.folder.name)
 print(x.folder/'COMPARISON.png')
if __name__=='__main__':main()
