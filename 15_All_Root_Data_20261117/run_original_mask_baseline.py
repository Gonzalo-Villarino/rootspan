"""Isolated compatibility-adjusted original PAT mask-stage baseline.

Neural inference uses recovered original state_dict/current SMP/eval/MPS.
Original tile grid, colour encoding, stitching and postprocessing are retained.
This is NOT a reproduction of the full historical GUI or FilFinder measurement.
"""
from pathlib import Path
import argparse, ast, gc, hashlib, json, math, os, time
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = 250_000_000
PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT/'00_Software/PAT_Pinned_20261115'
WEIGHTS = PROJECT/'00_Software/PAT_Reference/pat_state_dict.pt'
EXPECTED = '71ba82d7eab40c0b8489f47ee373fd0ea08f2e2e33457c62152a1f8240d68348'
COLOURS = np.array([[0,0,0],[128,128,0],[128,0,0],[0,128,0]], np.uint8)

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def write(p, obj): p.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

def infer(source, dest):
    import torch
    import segmentation_models_pytorch as smp
    if dest.exists(): raise FileExistsError(dest)
    assert sha(WEIGHTS)==EXPECTED
    with Image.open(source) as im:
        if im.mode!='RGB' or getattr(im,'n_frames',1)!=1: raise ValueError('Single-plane RGB required')
        rgb=np.array(im)
    h,w=rgb.shape[:2]; ny,nx=math.ceil(h/1016),math.ceil(w/1016)
    top,left=math.ceil((ny*1016-h)/2),math.ceil((nx*1016-w)/2)
    pad=np.pad(rgb,((top,ny*1016-h-top),(left,nx*1016-w-left),(0,0)))
    dest.mkdir(parents=True); tiles=dest/'tiles'; tiles.mkdir(); original=dest/'source'; original.mkdir()
    Image.fromarray(rgb).save(original/'slide.png')
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    model=smp.UnetPlusPlus(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=4,activation='sigmoid')
    model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True)
    model.to(device).eval().requires_grad_(False)
    preprocessing=smp.encoders.get_preprocessing_fn('resnet34','imagenet')
    start=time.monotonic()
    for iy in range(ny):
        for ix in range(nx):
            n=iy*nx+ix+1
            tile=pad[iy*1008:iy*1008+1024,ix*1008:ix*1008+1024]
            assert tile.shape==(1024,1024,3)
            tensor=torch.from_numpy(preprocessing(tile).transpose(2,0,1).astype('float32'))[None].to(device)
            with torch.inference_mode(): classes=model(tensor)[0].argmax(0).cpu().numpy()
            # Original cv2.imwrite receives the palette without channel reversal.
            # PIL expects RGB, so reverse to reproduce those on-disk PNG bytes' colours.
            Image.fromarray(COLOURS[classes][:,:,::-1]).save(tiles/f'slide_{n}.png')
            if n%25==0 or n==nx*ny: print(f'Original-grid inference {n}/{nx*ny}, {time.monotonic()-start:.0f}s',flush=True)
    write(dest/'inference.json',{'source':str(source.resolve()),'source_sha256':sha(source),'weights_sha256':EXPECTED,
          'source_size_wh':[w,h],'padding_left_top':[left,top],'padded_size_wh':[nx*1016,ny*1016],
          'tiles':nx*ny,'grid_xy':[nx,ny],'patch':1024,'stride':1008,'declared_overlap':16,
          'device':device,'torch':torch.__version__,'smp':smp.__version__,
          'adaptations':['state_dict reconstructed with current SMP','explicit eval and inference mode','MPS instead of original CPU','no metadata CSV or GUI needed','fixed isolated slide filename'],
          'training':False,'scope':'Original grid and mask stages, compatibility-adjusted; not full original software reproduction.'})

def original_stage(dest, stage):
    import cv2,pandas as pd
    src=REPO/'src/segmentation_patch_cor_mac.py'; tree=ast.parse(src.read_text())
    env={'cv2':cv2,'pd':pd,'np':np,'math':math,'os':os}
    for name in ('stitch_crop_images','get_average','closest_value','buffer'):
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),env)
    if stage=='stitch':
        folder=dest/'stitched'; folder.mkdir()
        env['stitch_crop_images'](1024,16,str(dest/'source')+'/',str(dest/'tiles')+'/',str(folder)+'/')
        del env;gc.collect()
        print('Original stitch completed',flush=True)
        return
    processed=dest/'processed'; binary=dest/'binary'; closed=dest/'closed'; final=dest/'measurement'
    for p in (processed,binary,closed,final):p.mkdir()
    # Execute the exact three original top-level processing loops in a scoped environment.
    for lo,hi,inp,out in [(730,800,dest/'stitched',processed),(890,910,processed,binary),(930,960,binary,closed)]:
        loop=next(n for n in tree.body if isinstance(n,ast.For) and lo<n.lineno<hi)
        env.update(image_path=str(inp)+'/',imageList=['slide.png'],save_path=str(out)+'/',save_path_1=str(out)+'/')
        exec(compile(ast.Module(body=[loop],type_ignores=[]),str(src),'exec'),env)
        assert (out/'slide.png').is_file()
        print(f'Original stage {out.name} completed',flush=True)
    measurement=REPO/'src/new_post_processing_for_measurement.py'
    t=ast.parse(measurement.read_text()); funcs=[n for n in t.body if isinstance(n,ast.FunctionDef)]
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(measurement),'exec'),env)
    foreground=env['process_images'](str(dest/'stitched/slide.png'))
    coloured=env['post_process_images'](foreground,str(closed/'slide.png'))
    coloured=env['red_to_white'](coloured); coloured=env['red_to_black'](coloured)
    env['keep_right_remove_left'](foreground,coloured,str(final/'slide.png'))
    info=json.loads((dest/'inference.json').read_text()); left,top=info['padding_left_top']; w,h=info['source_size_wh']
    for name in ('stitched','processed','binary','closed','measurement'):
        with Image.open(dest/name/'slide.png') as im:
            im.crop((left,top,left+w,top+h)).save(dest/f'{name}_source_coordinates.png')
    write(dest/'stages.json',{'original_segmentation_source_sha256':sha(src),'original_measurement_source_sha256':sha(measurement),
          'executed_original_loops':[730,890,930],'output_registration':'remove recorded centred padding; no resizing',
          'cv2':cv2.__version__,'numpy':np.__version__,'pandas':pd.__version__,
          'full_historical_GUI_run':False,'FilFinder_measurement_run':False})
    print('Original post-processing and source-coordinate masks saved',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['infer','stitch','postprocess'],required=True)
    p.add_argument('--source',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.stage=='infer':infer(a.source,a.output)
    else:original_stage(a.output,a.stage)
