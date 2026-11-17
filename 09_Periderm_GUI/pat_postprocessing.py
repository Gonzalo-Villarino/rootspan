# Uses original PAT methods/code: Copyright (c) 2023 Salk Harnessing Plants Initiative.
# Original MIT notice: 00_Software/PAT_Reference/LICENSE. See THIRD_PARTY_NOTICES.md.
"""Run the pinned original PAT cleanup on class-safe, source-aligned labels.

An experimental comparison, not a validated measurement pipeline. The AST
executes only named functions and processing loops, in fresh output folders.
No original repository data/output directories are opened for writing.
Run with the isolated pat-audit-env, which contains OpenCV and Pillow.
"""
from pathlib import Path
import argparse,ast,hashlib,json,os
import cv2,numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=250_000_000
PROJECT=Path(__file__).resolve().parents[1]
REPO=PROJECT/'00_Software/PAT_Pinned_20261115'
MANIFEST=PROJECT/'14_Original_PAT_Code_Review_20261117/source_manifest.json'

def run(rawmask,output):
    manifest=json.loads(MANIFEST.read_text())
    sources=['src/segmentation_patch_cor_mac.py','src/new_post_processing_for_measurement.py']
    hashes={n:hashlib.sha256((REPO/n).read_bytes()).hexdigest() for n in sources}
    if any(hashes[n]!=manifest['files'][n]['sha256'] for n in sources):raise ValueError('Original PAT source fingerprint changed.')
    output.mkdir(exist_ok=False)
    m=np.asarray(Image.open(rawmask))
    if m.ndim!=2 or m.dtype.kind not in 'ui' or not np.isin(m,[0,1,2,3]).all():raise ValueError('Four-class raw mask required.')
    palette=np.array([[0,0,0],[0,128,128],[0,0,128],[0,128,0]],np.uint8)
    for name in ('raw','processed','binary','closed','measurement'):(output/name).mkdir()
    Image.fromarray(palette[m]).save(output/'raw/slide.png')
    src=REPO/sources[0];tree=ast.parse(src.read_text());env={'cv2':cv2,'np':np,'os':os}
    for name in ('get_average','closest_value','buffer'):
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(src),'exec'),env)
    for lo,hi,inp,out in [(730,800,'raw','processed'),(890,910,'processed','binary'),(930,960,'binary','closed')]:
        loop=next(n for n in tree.body if isinstance(n,ast.For) and lo<n.lineno<hi)
        env.update(image_path=str(output/inp)+'/',imageList=['slide.png'],save_path=str(output/out)+'/',save_path_1=str(output/out)+'/')
        exec(compile(ast.Module(body=[loop],type_ignores=[]),str(src),'exec'),env)
        if not (output/out/'slide.png').is_file():raise ValueError('PAT cleanup stage missing: '+out)
    src=REPO/sources[1];tree=ast.parse(src.read_text())
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),str(src),'exec'),env)
    fg=env['process_images'](str(output/'raw/slide.png'))
    col=env['post_process_images'](fg,str(output/'closed/slide.png'))
    col=env['red_to_white'](col);col=env['red_to_black'](col)
    env['keep_right_remove_left'](fg,col,str(output/'measurement/slide.png'))
    with Image.open(output/'measurement/slide.png') as im:
        if im.size!=(m.shape[1],m.shape[0]):raise ValueError('Postprocessed dimensions changed.')
    record={'source_code':hashes,'original_repo_commit':manifest['commit'],
            'input_mask_sha256':hashlib.sha256(rawmask.read_bytes()).hexdigest(),
            'output_mask_sha256':hashlib.sha256((output/'measurement/slide.png').read_bytes()).hexdigest(),
            'scope':'Original cleanup stages applied to adapted inference. No original channel-max stitch. No original FilFinder lengths.',
            'cv2':cv2.__version__,'numpy':np.__version__,'training':False,'validated':False}
    (output/'record.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mask',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.mask,a.output)
