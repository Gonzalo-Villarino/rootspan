"""Version a specific teaching-label correction after expert transition review.

The old root3 interior strip is kept geometrically identical. Its existing
Endoderm labels at x >= the reviewed working T become Periderm, including the
transition under Gonzalo's convention. Unknown pixels, other roots, and source
annotations are unchanged. A new native 768-square crop makes T visible.
This prepares data for review; it neither trains nor loads model weights.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from prepare_training import poly, bounds, touch

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / '04_Results/59_Rule_Derived_Teaching_Data'
OUT = ROOT / '04_Results/61_Reviewed_Root3_Teaching_Correction'
REVIEW = ROOT / '02_Data/09_Transition_Endpoint_Review'
Image.MAX_IMAGE_PIXELS = 300_000_000


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def main():
    if OUT.exists():
        raise FileExistsError('Preserve prior versions: output directory already exists.')
    manifest_file = OLD / 'manifest.json'
    before_manifest_sha = sha(manifest_file)
    old = json.loads(manifest_file.read_text())
    manifest = copy.deepcopy(old)
    rule_file = Path(old['rule_regions'])
    assert sha(rule_file) == old['rule_regions_sha256']
    rule = json.loads(rule_file.read_text())
    strip = next(s for s in rule['shapes'] if s['root_id'] == 3)
    source_record = next(s for s in old['sources'] if s['slide']=='COL0_Slide1')
    source = Path(source_record['image'])
    annotation = Path(source_record['annotation'])
    assert sha(source) == source_record['image_sha256']
    assert sha(annotation) == source_record['annotation_sha256']
    shapes = json.loads(annotation.read_text())['shapes']
    route = json.loads((REVIEW/'route_proposal_02/reference_route_record.json').read_text())
    transition = json.loads((REVIEW/'COL0_root3_transition_review_04.json').read_text())
    tx = int(transition['working_boundary_source_x'])
    assert tx == route['working_transition_xy'][0]
    cy = route['working_transition_xy'][1]
    OUT.mkdir(); (OUT/'masks').mkdir(); (OUT/'images').mkdir()

    def correct(mask, box):
        # Never invent foreground: only relabel existing code-2 pixels inside
        # the reviewed root3 teaching strip, on the specified side of T.
        region = poly(strip, box)
        right = np.arange(box[0],box[2])[None,:] >= tx
        changed = (mask == 2) & region & right
        result = mask.copy(); result[changed] = 1
        assert np.all(mask[changed] == 2) and np.all(result[changed] == 1)
        assert np.array_equal(mask[~changed], result[~changed])
        return result, changed

    def counts(mask):
        return {str(c):int((mask==c).sum()) for c in [0,1,2,3,4,255]}

    changed_records=[]
    for p in manifest['patches']:
        # All existing manifest paths are made explicit; images are referenced,
        # not duplicated. Original patch masks remain on disk unchanged.
        for key in ['image','mask']:
            p[key]=str((OLD/p[key]).resolve())
        if p['slide']!='COL0_Slide1' or not touch(p['box'],bounds(strip)):
            continue
        previous=Path(p['mask'])
        assert sha(previous)==p['mask_sha256']
        before=np.array(Image.open(previous))
        after,change=correct(before,p['box'])
        if not change.any():
            continue
        assert p['root']==3
        output=OUT/'masks'/previous.name
        Image.fromarray(after).save(output)
        changes={'id':p['id'],'previous_mask':str(previous),
                 'previous_mask_sha256':p['mask_sha256'],
                 'new_mask':str(output),'new_mask_sha256':sha(output),
                 'changed_label_occurrences':int(change.sum()),
                 'before_counts':counts(before),'after_counts':counts(after)}
        changed_records.append(changes)
        p['mask']=str(output);p['mask_sha256']=changes['new_mask_sha256']
        p['counts']=counts(after)
        p['sample_type']='expert_transition_corrected_Periderm' if not p['counts']['2'] else 'expert_transition_corrected_mixed'
        p['derived_codes']=[c for c in [1,2] if p['counts'][str(c)]]
        p['rule_derived_codes']=[2] if p['counts']['2'] else []
        p['loss_weight']=.5  # No unrequested weighting change in this correction.
        p['correction_record']=str(OUT/'root3_class_correction.json')

    # One new field centred on the actual expert T. Prior root3 crops all
    # started to its right, and therefore missed the reviewed boundary.
    x,y=tx-384,int(round(cy))-384
    box=[x,y,x+768,y+768]
    bits=np.zeros((768,768),dtype=np.uint8)
    for s in shapes:
        if touch(box,bounds(s)):bits[poly(s,box)] |= 1
    for s in rule['shapes']:
        if touch(box,bounds(s)):bits[poly(s,box)] |= 2
    before=np.zeros_like(bits)
    before[bits==1]=1;before[bits==2]=2;before[bits==3]=255
    after,change=correct(before,box)
    assert np.count_nonzero(after==1)>1000 and np.count_nonzero(after==2)>1000
    name=f'COL0_root3_reviewed_T_x{x}_y{y}'
    image_path=OUT/'images'/f'{name}.png';mask_path=OUT/'masks'/f'{name}.png'
    im=Image.open(source).convert('RGB');patch=im.crop(box)
    patch.save(image_path);Image.fromarray(after).save(mask_path)
    Image.fromarray(before).save(OUT/'before_correction_same_crop_mask.png')
    manifest['patches'].append({'id':name,'slide':'COL0_Slide1','root':3,
        'sample_type':'expert_transition_corrected_mixed','box':box,
        'image':str(image_path),'image_sha256':sha(image_path),
        'mask':str(mask_path),'mask_sha256':sha(mask_path),'counts':counts(after),
        'rule_derived_codes':[2],'derived_codes':[1,2],'loss_weight':.5,
        'correction_record':str(OUT/'root3_class_correction.json')})

    # Count changed physical source pixels once, independently of overlapping
    # patch copies. This strip fits entirely within the new field.
    b=bounds(strip)
    assert box[0]<=b[0] and b[2]<box[2] and box[1]<=b[1] and b[3]<box[3]
    correction={'created_utc':datetime.now(timezone.utc).isoformat(),
        'source_image':str(source),'source_sha256':source_record['image_sha256'],
        'previous_data_manifest':str(manifest_file),'previous_manifest_sha256':before_manifest_sha,
        'expert_transition_review':str(REVIEW/'COL0_root3_transition_review_04.json'),
        'original_derived_strip_file':str(rule_file),'original_derived_strip_sha256':sha(rule_file),
        'original_derived_strip_root_id':3,'working_T_source_x':tx,
        'reviewed_approximate_T_zone_source_x':transition['review_zone_source_x'],
        'rule':'Within the existing root3 teaching-strip polygon only: code2 at native x>=12621 becomes code1. Code2 left of12621 remains Endoderm. All other pixels retain their old codes.',
        'biological_basis':'Gonzalo confirmed Endoderm left of working T and Periderm including transition to its right on this main root.',
        'geometry_status':'Original computer-derived interior strip retained; this is not a new hand-drawn full-root segmentation. No outer root contour, lateral-root class, or Background labels inferred.',
        'unique_source_pixels_relabelled':int(change.sum()),
        'old_patch_copies_changed':changed_records,
        'new_transition_patch':{'box':box,'image':str(image_path),'mask':str(mask_path),
             'before_counts':counts(before),'after_counts':counts(after)},
        'new_mask_visual_review_pending':True,'training_ready':False,
        'no_new_training':True,'source_annotations_unchanged':True,
        'do_not_use_as_blind_test':True,
        'not_a_background_update':True}
    dump(OUT/'root3_class_correction.json',correction)
    manifest['previous_manifest']=str(manifest_file)
    manifest['previous_manifest_sha256']=before_manifest_sha
    manifest['policy']=old['policy']+' Versioned root3 class correction follows detailed expert T review; one T-centred native crop added. No other root, background or image change. Awaiting correction preview review; not yet training-ready.'
    manifest['role']='teaching_candidate_awaiting_review'
    manifest['training_ready']=False
    manifest['correction_record']=str(OUT/'root3_class_correction.json')
    manifest['review_status']='New corrected mask awaits visual review. Prior class identity/working T has already been confirmed.'
    manifest['class_support_note']='14 earlier roots have saved P/E labels; 6 COL0 roots have archived P plus derived interior teaching strips. Root3 has a versioned class correction based on detailed expert transition review. Contours are not all independently pixel-verified.'
    manifest['rule_regions_usage']='Historical derivation only. Apply root3_class_correction.json after rasterizing the original regions; do not regenerate corrected masks from the historical region file alone.'
    dump(OUT/'manifest.json',manifest)

    # Source and two masks share a native-pixel crop; masks are displayed
    # separately so fluorescence colour cannot be mistaken for category colour.
    palette=np.zeros((256,3),dtype=np.uint8)
    palette[0]=[48,53,59];palette[1]=[218,48,64];palette[2]=[24,168,88]
    palette[3]=[255,216,55];palette[4]=[114,79,190];palette[255]=[245,83,181]
    font=lambda n:ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',n)
    crop_in_patch=[200,250,768,550]
    images=[patch.crop(crop_in_patch),Image.fromarray(palette[before]).crop(crop_in_patch),Image.fromarray(palette[after]).crop(crop_in_patch)]
    width=568
    canvas=Image.new('RGB',(1764,630),'white');draw=ImageDraw.Draw(canvas)
    draw.text((20,15),'COL0 root 3 — correcting the teaching answer key',font=font(30),fill='black')
    draw.text((20,58),'Same root interior strip. Your chosen transition T changes the category on its right.',font=font(24),fill='#273444')
    titles=['1  Microscopy image','2  Previous teaching labels','3  Corrected teaching labels']
    for xx,title,photo in zip([20,598,1176],titles,images):
        draw.text((xx,108),title,font=font(24),fill='black')
        dd=ImageDraw.Draw(photo)
        local_t=tx-box[0]-crop_in_patch[0]
        for yy in range(0,photo.height,16):dd.line((local_t,yy,local_t,min(yy+8,photo.height-1)),fill='white',width=2)
        dd.text((local_t+8,10),'T',font=font(25),fill='white',stroke_width=2,stroke_fill='black')
        canvas.paste(photo,(xx,145))
    draw.text((20,470),'Green = Endoderm     Red = Periderm     Grey = unlabelled, ignored during training',font=font(25),fill='black')
    draw.text((20,512),'These are small interior teaching regions, not full root outlines. The lateral branch stays unlabelled here.',font=font(23),fill='black')
    draw.text((20,551),'At T we use your convention: Periderm includes the transition. This boundary remains approximate.',font=font(23),fill='black')
    draw.text((20,590),'No model was trained. Original images, annotations and earlier results are preserved.',font=font(23),fill='#374151')
    figure=OUT/'ROOT3_TEACHING_CORRECTION_REVIEW.png';canvas.save(figure)
    shutil.copy2(__file__,OUT/Path(__file__).name)
    assert sha(source)==source_record['image_sha256']
    assert sha(annotation)==source_record['annotation_sha256']
    assert sha(manifest_file)==before_manifest_sha
    for c in changed_records:
        assert sha(c['previous_mask'])==c['previous_mask_sha256']
    dump(OUT/'verification.json',{'source_image_hash_unchanged':True,
        'source_annotation_hash_unchanged':True,'old_manifest_hash_unchanged':True,
        'old_patch_masks_hashes_unchanged':True,'only_2_to_1_within_reviewed_strip_changed':True,
        'new_patch_retains_both_tissues':True,'no_unknown_pixels_relabelled':True,
        'no_background_or_lateral_root_labels_changed':True,
        'old_patch_count':len(old['patches']),'candidate_patch_count':len(manifest['patches']),
        'changed_old_patch_copies':len(changed_records),'unique_source_pixels_relabelled':int(change.sum()),
        'new_training':False})
    (OUT/'READ_ME.txt').write_text(
        'REVIEW COPY — NO NEW TRAINING\n\n'
        'The previous root3 teaching strip was all Endoderm. Gonzalo reviewed the cells and selected T: left Endoderm, right Periderm including transition.\n'
        'This copy changes existing Endoderm pixels to Periderm only inside that same strip to the right of T. It does not redraw tissue, fill unknown areas, or assume background.\n'
        'Four overlapping earlier patch copies are corrected where applicable; one native patch centred on the expert T is added. Other image/mask paths reference unchanged saved files.\n'
        'The illustration displays the same part of that new patch before and after the correction. These are answer keys, not model predictions.\n'
        'New mask preview awaits review before training. The candidate manifest explicitly says training_ready=false. Existing training scripts still use the historical data and must not simply be rerun.\n'
        'All earlier files, models and archive annotations are retained. This is a teaching root, not a fresh test.\n')
    print(json.dumps({'figure':str(figure),'unique_source_pixels_relabelled':int(change.sum()),
        'changed_old_patch_copies':len(changed_records),'new_patch_counts':counts(after),
        'candidate_patch_count':len(manifest['patches']),'training_ready':False},indent=2))


if __name__=='__main__':main()
