"""Use the locked checkpoint on A04 root 1; no training or rule changes."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import subprocess
import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw, ImageFont
from extend_root3_reference_route import graph_route

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / '02_Data/13_Locked_Model_Checks_20261116/A04_root1/source_review_02'
OUT = ROOT / '04_Results/71_Locked_A04_Root1_Check'
BUNDLE = ROOT / '06_Models/Locked_Transition_v1_20261116'
Image.MAX_IMAGE_PIXELS = 300_000_000
NAMES = ['Endoderm', 'Mixed / transition', 'Periderm', 'Neither supported']
COLORS = ['#1aa65e', '#a33cc4', '#db303f', '#777777']
BLUE = '#28b7ff'
font = lambda n: ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', n)
read = lambda p: json.loads(p.read_text())
now = lambda: datetime.now(timezone.utc).isoformat()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify():
    result = subprocess.run([sys.executable, str(BUNDLE / 'verify_checkpoint.py'), '--require-read-only'],
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def prepare():
    OUT.mkdir(exist_ok=False)
    plan = read(REVIEW / 'source_review_plan.json')
    source = Path(plan['source'])
    assert sha(source) == plan['source_sha256']
    checkpoint = verify()
    expert = {'recorded_utc': now(), 'expert': 'Gonzalo Villarino',
              'verbatim': 'A is endoderm, C is Perid, B is T',
              'fields': [dict(f, biological_label=label, appearance_state=state)
                         for f, label, state in zip(plan['review_fields'], NAMES[:3], [0, 1, 2])],
              'interpretation': 'A Endoderm, B mixed/transition appearance, C Periderm on the main root.',
              'exact_T_source_xy': None, 'exact_T_interval': None,
              'scope': 'Qualitative field identities, not exact left-edge coordinates or pixel masks.',
              'used_for_training': False, 'recorded_before_this_prediction': True}
    assert not (REVIEW / 'expert_appearance_review_01.json').exists()
    save(REVIEW / 'expert_appearance_review_01.json', expert)
    save(OUT / 'expert_appearance_review_01.json', expert)
    bounds = plan['source_box_xyxy']
    rgb = np.array(Image.open(source).convert('RGB').crop(bounds))
    assert np.array_equal(rgb, np.array(Image.open(REVIEW / 'source_native.png')))
    h, w = rgb.shape[:2]
    green_native = np.pad(rgb[:, :, 1], ((0, (-h) % 4), (0, (-w) % 4)))
    green = np.array(Image.fromarray(green_native).resize((green_native.shape[1] // 4, green_native.shape[0] // 4), Image.Resampling.BOX), dtype=float)
    foreground = ndi.binary_fill_holes(ndi.binary_closing(ndi.gaussian_filter(green, 1) > 16, iterations=2))
    labels, count = ndi.label(foreground)
    eligible = []
    for k in range(1, count + 1):
        yx = np.argwhere(labels == k)
        if yx[:, 1].min() <= 4 and yx[:, 1].max() >= green.shape[1] - 5:
            eligible.append((len(yx), k, yx))
    if not eligible:
        raise ValueError('No source-fluorescence component spans the crop. Inspect geometry before inference.')
    _, k, yx = max(eligible, key=lambda item: item[0])
    xmin, xmax = yx[:, 1].min(), yx[:, 1].max()
    start = np.array([xmin, np.median(yx[yx[:, 1] <= xmin + 2, 0])])
    stop = np.array([xmax, np.median(yx[yx[:, 1] >= xmax - 2, 0])])
    raw, ends = graph_route(labels == k, start, stop)
    raw = raw * 4 + 1.5
    raw_arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
    positions = np.r_[np.arange(0, raw_arc[-1], 4), raw_arc[-1]]
    path = ndi.gaussian_filter1d(np.c_[np.interp(positions, raw_arc, raw[:, 0]), np.interp(positions, raw_arc, raw[:, 1])], 5, axis=0, mode='nearest')
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    assert np.isfinite(path).all() and (np.diff(path[:, 0]) >= -3).all()
    assert np.max(np.linalg.norm(np.diff(path, axis=0), axis=1)) < 12
    Image.fromarray(rgb).save(OUT / 'image.png')
    save(OUT / 'image_only_route.json', {
        'created_utc': now(), 'source_image': str(source), 'source_sha256': sha(source),
        'source_crop_xyxy': bounds, 'source_size_xy': [w, h],
        'path_xy_crop_pixels': path.tolist(), 'distance_along_path_pixels': arc.tolist(),
        'source_only_geometry': True, 'tissue_labels_used': False,
        'settings': {'downsample': 4, 'green_threshold': 16, 'gaussian_sigma_coarse': 1,
                     'closing_iterations_coarse': 2, 'resample_px': 4, 'path_smoothing_sigma_samples': 5},
        'geometry_bottom_right_zero_padding_yx': [(-h) % 4, (-w) % 4],
        'endpoint_distances_coarse_pixels': ends,
        'endpoints': 'Local crop-edge positions, not biological cut end or hypocotyl.',
        'limits': 'Image-only main-root geometry is separate from locked appearance inference. Windows can include branch context.'})
    save(OUT / 'check_plan.json', {
        'created_utc': now(), 'checkpoint': str(BUNDLE), 'checkpoint_integrity': checkpoint,
        'expert_review_sha256': sha(OUT / 'expert_appearance_review_01.json'),
        'source_review_plan': str(REVIEW / 'source_review_plan.json'),
        'prediction': 'Execute the locked run_checkpoint.py unchanged on this image and checked route, without supplying labels.',
        'comparison': 'After saving prediction, compare all scan centres inside each expert box and the nearest centre. Overlapping path-aligned fields differ from the expert rectangles; these are qualitative comparisons, not pixel accuracy.',
        'T_reference': 'B is mixed/transition; no exact left edge or reference interval supplied.',
        'test_scope': plan['selection_limit'], 'new_training': False,
        'calibration': 'A04 unconfirmed: no micrometre length claim.'})
    overview = Image.fromarray(rgb).resize((1680, 450), Image.Resampling.LANCZOS)
    traced = overview.copy()
    ImageDraw.Draw(traced).line([(x * .3, y * .3) for x, y in path], fill='white', width=2)
    fig = Image.new('RGB', (1720, 1050), 'white')
    fig.paste(overview, (20, 100)); fig.paste(traced, (20, 575))
    d = ImageDraw.Draw(fig)
    d.text((20, 12), 'A04 root 1: main-root path check', font=font(30), fill='black')
    d.text((20, 56), 'Top: source. Bottom: image-derived route. No tissue labels or T supplied.', font=font(24), fill='#334155')
    fig.save(OUT / 'PREPARED_MAIN_ROOT_PATH.png')
    shutil.copy2(__file__, OUT / Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('extend_root3_reference_route.py'), OUT / 'extend_root3_reference_route.py')
    print('Expert review and image-only route saved. Inspect route before prediction.', flush=True)


def predict():
    assert read(OUT / 'route_visual_check.json')['passed']
    result = subprocess.run([sys.executable, str(BUNDLE / 'run_checkpoint.py'),
                             '--image', str(OUT / 'image.png'), '--route', str(OUT / 'image_only_route.json'),
                             '--output', str(OUT / 'inference')], text=True, capture_output=True, check=True)
    (OUT / 'inference_stdout.txt').write_text(result.stdout)
    print(result.stdout, flush=True)


def compare():
    assert not (OUT / 'comparison_result.json').exists()
    pred_path = OUT / 'inference/prediction.json'
    pred = read(pred_path); before = sha(pred_path)
    expert = read(OUT / 'expert_appearance_review_01.json')
    route = read(OUT / 'image_only_route.json')
    centers = np.array(pred['centres_xy_crop_pixels']) + np.array(route['source_crop_xyxy'][:2])
    states = np.array(pred['rule']['original_rule']['window_states'])
    comparisons = []
    for f in expert['fields']:
        box = np.array(f['source_box_xyxy']); center = (box[:2] + box[2:]) / 2
        ids = np.flatnonzero((centers[:, 0] >= box[0]) & (centers[:, 0] < box[2]) & (centers[:, 1] >= box[1]) & (centers[:, 1] < box[3]))
        j = int(np.argmin(np.linalg.norm(centers - center, axis=1)))
        comparisons.append({'field': f['name'], 'expert_appearance': f['biological_label'],
            'nearest_scan_index': j, 'nearest_model_appearance': NAMES[states[j]],
            'nearest_scores': pred['presence_scores'][j], 'scan_center_source_xy': centers[j].tolist(),
            'centres_inside_expert_box': ids.tolist(),
            'inside_box_state_counts': {name: int((states[ids] == i).sum()) for i, name in enumerate(NAMES)},
            'scope': 'Centres inside expert view; windows may extend outside it. Overlapping windows, not accuracy.'})
    t = pred['T_source_xy']; b = expert['fields'][1]['source_box_xyxy']
    relation = 'No T returned' if t is None else ('left of B' if t[0] < b[0] else 'within the horizontal extent of B' if t[0] <= b[2] else 'right of B')
    result = {'created_utc': now(), 'model_T_source_xy': t, 'model_T_path_pixels': pred['T_path_pixels'],
              'relation_to_B': relation, 'field_comparisons': comparisons,
              'model_updated': False, 'new_training': False, 'precise_T_error': None,
              'no_exact_expert_T': True, 'no_full_root_length_measurement': True,
              'interpretation_limit': 'B confirms a mixed area, not the exact left edge. T placement needs biological review. Familiar slide, not blind final validation.'}
    save(OUT / 'comparison_result.json', result)
    im = Image.open(OUT / 'image.png').convert('RGB')
    fig = Image.new('RGB', (1780, 1225), 'white')
    fig.paste(im.resize((1680, 450), Image.Resampling.LANCZOS), (50, 105))
    for i, f in enumerate(expert['fields']):
        fig.paste(Image.open(REVIEW / f'{f["name"]}_native.png').resize((560, 392), Image.Resampling.NEAREST), (20 + i * 590, 745))
    d = ImageDraw.Draw(fig)
    d.text((20, 12), 'A04 root 1: the same locked model, applied to another root', font=font(30), fill='black')
    d.text((20, 58), 'Your reference: A = Endoderm; B = mixed / transition; C = Periderm.', font=font(25), fill='#334155')
    for f in expert['fields']:
        x0, y0, x1, y1 = f['source_box_xyxy']
        xx = 50 + (x0 - 3400) * .3; yy = 105 + y0 * .3
        d.rectangle((xx, yy, xx + (x1 - x0) * .3, yy + (y1 - y0) * .3), outline='white', width=2)
        d.text((xx + 3, max(107, yy - 34)), f['name'], font=font(27), fill='white', stroke_width=1, stroke_fill='black')
    if t is not None:
        tx = 50 + (t[0] - 3400) * .3
        d.line((tx, 105, tx, 555), fill=BLUE, width=4)
        d.text((min(tx + 10, 1490), 475), 'Model T', font=font(27), fill=BLUE, stroke_width=1, stroke_fill='black')
    for xy, state in zip(centers, states):
        x = 50 + (xy[0] - 3400) * .3
        d.rectangle((x - 10, 577, x + 10, 605), fill=COLORS[state])
    d.text((20, 624), 'Model windows: green = Endoderm; purple = mixed; red = Periderm; grey = neither.', font=font(24), fill='black')
    for i, c in enumerate(comparisons):
        xx = 20 + i * 590
        d.text((xx, 669), f'{c["field"]}: you = {c["expert_appearance"]}', font=font(23), fill='black')
        d.text((xx, 706), f'Nearby model = {c["nearest_model_appearance"]}', font=font(23), fill=COLORS[NAMES.index(c['nearest_model_appearance'])])
    d.text((20, 1155), 'Photos below show your original review views; model windows are nearby and path-aligned.', font=font(23), fill='#334155')
    d.text((20, 1190), 'Solid blue = automatic model T. No retraining, manual shift, or exact T-error claim.', font=font(23), fill='#075985')
    fig.save(OUT / 'A04_ROOT1_LOCKED_RESULT.png')
    if t is not None:
        # The crop is explicitly centered on the prediction for position review.
        cx, cy = t[0] - 3400, t[1]
        x0 = int(np.clip(cx - 750, 0, im.width - 1500)); y0 = int(np.clip(cy - 250, 0, im.height - 500))
        close = im.crop((x0, y0, x0 + 1500, y0 + 500))
        review = Image.new('RGB', (1540, 650), 'white'); review.paste(close, (20, 100))
        d = ImageDraw.Draw(review)
        d.text((20, 12), 'Closer look at the automatic blue T', font=font(30), fill='black')
        d.text((20, 57), 'Same source pixels at native size. No manual movement of T.', font=font(24), fill='#334155')
        tx = 20 + cx - x0; d.line((tx, 100, tx, 600), fill=BLUE, width=3)
        d.text((min(tx + 10, 1360), 115), 'Model T', font=font(24), fill=BLUE, stroke_width=1, stroke_fill='black')
        d.text((20, 615), 'Is this a reasonable left edge of the mixed-cell zone on the main root?', font=font(24), fill='black')
        review.save(OUT / 'A04_ROOT1_T_CLOSEUP.png')
        save(OUT / 'T_closeup_source_box.json', {'source_box_xyxy': [x0 + 3400, y0, x0 + 4900, y0 + 500], 'purpose': 'Prediction-centred source view for biological review, not independent reference.'})
    assert sha(pred_path) == before
    assert sha(Path(route['source_image'])) == route['source_sha256']
    save(OUT / 'verification.json', {'created_utc': now(), 'checkpoint': verify(), 'source_unchanged': True,
         'prediction_unchanged_during_comparison': True, 'expert_labels_used_for_inference': False,
         'image_sha256': sha(OUT / 'image.png'), 'route_sha256': sha(OUT / 'image_only_route.json'),
         'prediction_sha256': before, 'figure_visual_qa': 'Pending'})
    lines = ['A04 ROOT 1 — LOCKED TRANSITION v1 CHECK', '',
             'Your reference: A Endoderm; B mixed/transition; C Periderm.',
             f'Automatic T: {t}; relation: {relation}.',
             *[f'{c["field"]}: expert {c["expert_appearance"]}; nearest model field {c["nearest_model_appearance"]}; all field-centre states {c["inside_box_state_counts"]}.' for c in comparisons],
             '', 'The exact locked runner was used. No labels were given to the model; no training, threshold change or manual T shift occurred.',
             'The main-root path was derived from image fluorescence and visually checked before inference. Rectangular windows can still contain lateral-branch context.',
             'The appearance head learned from COL0. A04 was viewed in earlier checks; this is additional evidence, not a pristine blind test. Published encoder exposure is unknown.',
             'B identifies a mixed area but no exact left edge. Field comparisons use nearby, differently framed model windows, not pixel accuracy.',
             'T placement needs biological review. A04 calibration is unconfirmed; no length measurement is claimed.',
             'Source, earlier results and read-only checkpoint preserved. Word manual not edited.']
    (OUT / 'READ_ME.txt').write_text('\n'.join(lines) + '\n')
    with (ROOT / '05_Notes/03_Progress_and_Next_Step.txt').open('a') as f:
        f.write('\n16 November 2026 — Locked v1 checked on A04 root 1\n' + '\n'.join(lines[2:]) + '\nResults: ' + str(OUT) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'predict', 'compare'], required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'predict': predict, 'compare': compare}[args.phase]()
