"""Measure the user-approved COL0 root3 reference in native coordinates.

This does not run a model. It sums distances along the approved line, then
converts them using the user-confirmed calibration. Original files are read-only.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / '02_Data/09_Transition_Endpoint_Review'
OUT = REVIEW / 'reference_measurement_01'
Image.MAX_IMAGE_PIXELS = 300_000_000


def main():
    if OUT.exists():
        raise FileExistsError('Use a new directory; existing records are preserved.')
    route_file = REVIEW / 'route_proposal_02/reference_route_record.json'
    route = json.loads(route_file.read_text())
    calibration_file = REVIEW / 'COL0_root3_calibration_review_02.json'
    calibration = json.loads(calibration_file.read_text())
    assert calibration['confirmed_for_COL0']
    source = Path(route['source'])
    with source.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == route['source_sha256']
    path = np.array(route['path_xy_source_pixels'], dtype=float)
    ti = route['transition_path_index']
    assert np.array_equal(path[ti], route['working_transition_xy'])
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total = float(lengths.sum())
    endoderm = float(lengths[:ti].sum())
    periderm = float(lengths[ti:].sum())
    assert np.isclose(endoderm + periderm, total, rtol=1e-12)
    scale = float(calibration['pixels_per_micrometre'])
    assert scale > 0 and np.isfinite(scale)
    measurements = {
        name: {'native_pixels': value, 'micrometres': value/scale,
               'millimetres': value/scale/1000,
               'display_micrometres_nearest_10': int(round(value/scale/10)*10)}
        for name, value in [('Periderm_T_to_J',periderm),
                            ('Endoderm_L_to_T',endoderm), ('Total_L_to_J',total)]
    }
    OUT.mkdir()
    record = {
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'user_approval_verbatim':'yes the cyan line follows the main root correctly. mabe use diffrent colors than cyan for endoderm . but that is a detail, where do we go from here?',
        'reference_route_file':str(route_file),
        'source':str(source),'source_sha256':route['source_sha256'],
        'route_approved_by_user':True,'reference_is_model_prediction':False,
        'reference_is_independent_manual_trace':False,
        'method':'Sum Euclidean distances between consecutive approved native-coordinate points. Split at the same T point for both tissue lengths. Divide pixel length by the user-confirmed pixels per micrometre.',
        'measurements':measurements,
        'calibration_record':str(calibration_file),'pixels_per_micrometre':scale,
        'boundary_status':'Working T and J selected in approximate expert-reviewed intervals. These reference values are not exact histological boundaries or independently calibrated experimental validation.',
        'biological_scope':'Photographed main-root section only. Periderm includes transition under the user convention; root-hypocotyl junction is the right stop. Lateral roots and hypocotyl excluded.',
        'length_plausibility_check':{
            'user_approximate_section_length_cm':2,
            'calculated_section_length_cm':total/scale/10000,
            'status':'Calculated length is longer than the earlier rough recollection. Keep flagged for checking acquisition calibration/section length before treating as experimental result. No rescaling to force agreement.'},
        'model_development_status':'COL0 has already supplied training examples. This root can help correct teaching labels and inspect model behaviour; it is not a new blind test.',
        'known_teaching_label_issue':'Detailed expert review identifies some Periderm within a previously derived Endoderm region of root3. Correct a versioned teaching-data copy before further training; preserve prior runs.',
        'new_training':False,'original_annotations_changed':False,
        'colours':{'Endoderm_line':'green','Periderm_line':'red'},
    }
    (OUT/'reference_measurements.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    image = Image.open(source).convert('RGB')
    box = [1700,2780,20100,6500]
    width=1800
    ratio=width/(box[2]-box[0])
    photo=image.crop(box).resize((width,round((box[3]-box[1])*ratio)),Image.Resampling.LANCZOS)
    draw=ImageDraw.Draw(photo)
    def xy(p):return ((p[0]-box[0])*ratio,(p[1]-box[1])*ratio)
    draw.line([xy(p) for p in path],fill='black',width=6)
    draw.line([xy(p) for p in path[:ti+1]],fill='#70ef3c',width=3)
    draw.line([xy(p) for p in path[ti:]],fill='#ff3d4a',width=3)
    font=lambda n:ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',n)
    for label,p in [('L',path[0]),('T',path[ti]),('J',path[-1])]:
        x,y=xy(p);draw.ellipse((x-6,y-6,x+6,y+6),fill='white')
        draw.text((x+10,y-34),label,font=font(26),fill='white',stroke_width=2,stroke_fill='black')
    canvas=Image.new('RGB',(1840,photo.height+420),'white')
    d=ImageDraw.Draw(canvas)
    d.text((20,14),'COL0 root 3 — your reviewed reference measurement',font=font(32),fill='black')
    d.text((20,60),'Green line: Endoderm (L–T)     Red line: Periderm including transition (T–J)',font=font(26),fill='#202b38')
    d.text((20,99),'Human-reviewed landmarks and computer tracing. This is not an automatic model result.',font=font(24),fill='#273444')
    canvas.paste(photo,(20,145))
    y=145+photo.height+25
    for x,key,label in [(20,'Periderm_T_to_J','Periderm'),(630,'Endoderm_L_to_T','Endoderm'),(1240,'Total_L_to_J','Total section')]:
        value=measurements[key]['display_micrometres_nearest_10']
        d.text((x,y),f'{label}: approximately {value:,} µm',font=font(27),fill='black')
    d.text((20,y+49),'Total = Endoderm + Periderm. Length follows the bends of the main root; lateral branches are excluded.',font=font(24),fill='black')
    d.text((20,y+88),'Conversion: native pixel length ÷ 0.5299 = micrometres. Scale confirmed by you for COL0.',font=font(24),fill='black')
    d.text((20,y+127),'Check to retain: this gives 3.62 cm total, longer than the earlier rough 2 cm section estimate.',font=font(24),fill='#784d00')
    d.text((20,y+166),'Working landmarks remain approximate. No new training or edits to original images and annotations.',font=font(23),fill='#374151')
    figure=OUT/'COL0_root3_reference_lengths.png';canvas.save(figure)
    note=(
        'COL0 ROOT 3 — REVIEWED REFERENCE, NOT AUTOMATIC MODEL OUTPUT\n\n'
        'Gonzalo approved the complete main-root route. Green = Endoderm; red = Periderm.\n'
        f'Periderm T-J: approximately {measurements["Periderm_T_to_J"]["display_micrometres_nearest_10"]:,} micrometres.\n'
        f'Endoderm L-T: approximately {measurements["Endoderm_L_to_T"]["display_micrometres_nearest_10"]:,} micrometres.\n'
        f'Total L-J: approximately {measurements["Total_L_to_J"]["display_micrometres_nearest_10"]:,} micrometres.\n'
        'The root-hypocotyl junction is the right stop; transition is included in Periderm.\n'
        'Same line and units: Endoderm = total - Periderm.\n'
        'Calibration: user-confirmed 0.5299 pixels per micrometre; divide native pixel length by this number.\n'
        'Check: total is 3.62 cm, longer than the earlier approximate 2 cm recollection. Do not force the measurement to match that estimate.\n'
        'This root is already part of teaching data, so it is not a new blind test. Detailed review revealed a disagreement in the derived Endoderm teaching region.\n'
        'Next small step: prepare corrected teaching labels around this reviewed transition in a new version. Keep source labels and earlier models unchanged.\n'
        'Then review enough varied roots, train with corrected examples, and evaluate on separate slides withheld from model development.\n'
        'No training was run in this measurement step. Word manual not modified.\n'
    )
    (OUT/'READ_ME.txt').write_text(note)
    (OUT/Path(__file__).name).write_text(Path(__file__).read_text())
    print(json.dumps({'measurements':measurements,'figure':str(figure)},indent=2))


if __name__=='__main__':
    main()
