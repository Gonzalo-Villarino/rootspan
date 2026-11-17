"""Bounded synthetic probes of unmodified original PAT code; no real images or models."""
from pathlib import Path
import ast,contextlib,hashlib,io,json,math,os,runpy,tempfile
import cv2,numpy as np,pandas as pd

HERE=Path(__file__).resolve().parent
REPO=HERE.parent/'00_Software/PAT_Pinned_20261115'

def function_from_source(file,name):
    tree=ast.parse(file.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    env={'cv2':cv2,'np':np,'pd':pd,'os':os,'math':math}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(file),'exec'),env)
    return env[name]

def stitched_example(size,colors):
    stitch=function_from_source(REPO/'src/segmentation_patch_cor_mac.py','stitch_crop_images')
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);src=root/'source';tiles=root/'tiles';out=root/'out'
        for folder in (src,tiles,out):folder.mkdir()
        cv2.imwrite(str(src/'sample.png'),np.zeros((size,size,3),np.uint8))
        n=math.ceil(size/1016)
        for i in range(n*n):
            tile=np.empty((1024,1024,3),np.uint8);tile[:]=colors[i%len(colors)]
            cv2.imwrite(str(tiles/f'sample_{i+1}.png'),tile)
        with contextlib.redirect_stdout(io.StringIO()):stitch(1024,16,str(src)+'/',str(tiles)+'/',str(out)+'/')
        return cv2.imread(str(out/'sample.png'))

def main():
    results={'scope':'Synthetic probes of original source at pinned commit; not biological accuracy or a full PAT run.'}
    # The PNG channel values are those used by original cv2 write/read, with labels from supplied CSV.
    mixed=stitched_example(2032,[[128,0,0],[0,128,0]])
    invented=int(np.all(mixed==[128,128,0],axis=2).sum())
    assert invented>0
    results['stitching_changes_class']={'input_classes':['Endodermis','Lateral_root'],'no_input_periderm_pixels':True,'output_periderm_color_pixels':invented,'source_function':'segmentation_patch_cor_mac.py:572-683','interpretation':'Channelwise max can create the Periderm color when overlapping predictions disagree as E versus LR.'}

    uniform=stitched_example(3048,[[128,128,0]])
    occupied=np.any(uniform!=0,axis=2)
    assert not occupied[-8:,:].any() and not occupied[:,-8:].any()
    results['stitching_uncovered_edges']={'source_dimensions':[3048,3048],'all_prediction_tiles_periderm':True,'black_output_pixels':int((~occupied).sum()),'bottom_8_rows_empty':True,'right_8_columns_empty':True,'unpadded_source_size_equals_output_in_this_example':True}
    width=18143;n=math.ceil(width/1016);padded=n*1016;left=math.ceil((padded-width)/2);covered=(n-1)*1008+1024
    results['718_width_coverage_by_code_formula']={'source_width':width,'columns':n,'padded_width':padded,'left_padding':left,'maximum_tile_coverage_exclusive':covered,'original_right_columns_beyond_tile_coverage':max(0,left+width-covered),'is_analytic_not_full_slide_execution':True}

    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'src').mkdir();(root/'output').mkdir()
        script=root/'src/QC_selected.py';script.write_bytes((REPO/'src/QC_selected.py').read_bytes())
        (root/'output/selected_image_names.txt').write_text('slideA_1.png\n')
        (root/'output/periderm_length_pixels.csv').write_text('Image,root_1,root_2\nslideA,100,200\nslideB,300,400\n')
        runpy.run_path(str(script),run_name='__main__')
        table=pd.read_csv(root/'output/periderm_length_after_QC_pixels.csv').set_index('Image')
        assert table.loc['slideB','root_1']==300 and table.loc['slideB','root_2']==400
        assert pd.isna(table.loc['slideA','root_2'])
        results['QC_slide_with_no_selected_roots']={'selected':['slideA_1.png'],'slideA_root2_becomes_missing':True,'unselected_slideB_retains':[300,400],'source':'QC_selected.py:35-43','interpretation':'If an image has no entry in selected_roots, its values are not cleared.'}

    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'src').mkdir();(root/'nature_accession').mkdir()
        for name in ('Segmentation_upp_v15','measurement'):(root/'output'/name).mkdir(parents=True)
        script=root/'src/generate_images_for_QC.py';script.write_bytes((REPO/'src/generate_images_for_QC.py').read_bytes())
        try:runpy.run_path(str(script),run_name='__main__')
        except FileNotFoundError as error:
            assert 'nature_accession_flip' in str(error)
            results['QC_folder_mismatch']={'normal_GUI_folder_present':'nature_accession','missing_required_folder':'nature_accession_flip','error_type':type(error).__name__,'original_script_executed_in_temporary_copy':True}
        else:raise AssertionError('Expected documented folder mismatch')

    masks=[]
    for p1,p2 in [((10,10),(110,10)),((10,10),(110,110))]:
        mask=np.zeros((128,128),np.uint8);cv2.line(mask,p1,p2,255,1)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        counts=[int(np.sum(cv2.drawContours(np.zeros_like(mask),[c],-1,255,thickness=1)==255)) for c in contours]
        masks.append({'counted_skeleton_pixels':counts[0],'endpoint_distance_pixels':float(np.linalg.norm(np.array(p2)-p1))})
    assert masks[0]['counted_skeleton_pixels']==masks[1]['counted_skeleton_pixels']
    results['skeleton_pixel_count_is_not_euclidean_arc_length']={'horizontal':masks[0],'diagonal_45_degrees':masks[1],'scope':'The original final counting expression after skeleton extraction; no FilFinder run or estimate of real-data bias.'}

    # Run the complete original measurement-postprocessing functions on one short E-only component.
    tree=ast.parse((REPO/'src/new_post_processing_for_measurement.py').read_text())
    env={'cv2':cv2,'np':np,'os':os}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'original_postprocessing','exec'),env)
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp);full=np.zeros((300,1800,3),np.uint8);full[100:160,900:1700]=[128,0,0]
        cv2.imwrite(str(folder/'whole.png'),full);cv2.imwrite(str(folder/'periderm.png'),np.zeros_like(full))
        fg=env['process_images'](str(folder/'whole.png'))
        colored=env['post_process_images'](fg,str(folder/'periderm.png'))
        colored=env['red_to_white'](colored);colored=env['red_to_black'](colored)
        with contextlib.redirect_stdout(io.StringIO()):env['keep_right_remove_left'](fg,colored,str(folder/'final.png'))
        count=int(np.count_nonzero(cv2.imread(str(folder/'final.png'),cv2.IMREAD_GRAYSCALE)))
        assert count==48000
        results['postprocessing_converts_short_E_only_component']={'image_size':[1800,300],'E_component_xyxy':[900,100,1700,160],'input_periderm_mask_pixels':0,'final_white_measurement_mask_pixels':count,'scope':'Short component satisfying the original area, bbox and right-side rules. This does not show that the long root 1 was converted this way.'}

    gui=ast.parse((REPO/'PAT_GUI_mac_cpu.py').read_text());cls=next(n for n in gui.body if isinstance(n,ast.ClassDef) and n.name=='MainWindow')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='safe_convert_pixels_to_micrometers')
    env={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'original_conversion','exec'),env)
    converted=env[fn.name](None,'7_18',0.5299)
    assert isinstance(converted,float)
    results['unit_conversion_also_converts_numeric_underscore_image_name']={'input':'7_18','output':converted,'source':'PAT_GUI_mac_cpu.py:434-447','scope':'Original conversion helper, which the GUI applies to all cells. Whole GUI was not launched.'}

    tree=ast.parse((REPO/'src/segmentation_patch_cor_mac.py').read_text())
    loop=next(n for n in tree.body if isinstance(n,ast.For) and 730<n.lineno<800)
    env={'cv2':cv2,'np':np,'os':os}
    for name in ('get_average','closest_value','buffer'):env[name]=function_from_source(REPO/'src/segmentation_patch_cor_mac.py',name)
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp);(folder/'out').mkdir()
        im=np.zeros((500,1500,3),np.uint8);im[200:260,100:500]=[128,128,0];im[200:260,600:1000]=[128,128,0]
        cv2.imwrite(str(folder/'sample.png'),im)
        env.update(image_path=str(folder)+'/',imageList=['sample.png'],save_path=str(folder/'out')+'/')
        exec(compile(ast.Module(body=[loop],type_ignores=[]),'original_gap_processing','exec'),env)
        assert len(env['contours_filter'])==0
        results['gap_processing_inverted_external_contours']={'periderm_rectangles':2,'gap_width_pixels':100,'qualifying_contours':0,'scope':'The exact original loop on two disconnected P rectangles. Gap-bridging branch receives no candidates; does not establish behavior for every topology.'}

    manifest=json.loads((HERE/'source_manifest.json').read_text())
    results['original_source_unchanged']=all(hashlib.sha256((REPO/f).read_bytes()).hexdigest()==v['sha256'] for f,v in manifest['files'].items())
    assert results['original_source_unchanged']
    (HERE/'synthetic_probe_results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
