"""Record the expert T interval without changing the model or its prediction."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / '04_Results/71_Locked_A04_Root1_Check'
OUT = PARENT / 'expert_T_review_02'
BUNDLE = ROOT / '06_Models/Locked_Transition_v1_20261116'
Image.MAX_IMAGE_PIXELS = 300_000_000


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    OUT.mkdir(exist_ok=False)
    detail = json.loads((PARENT / 'expert_T_review_01/numbered_review_details.json').read_text())
    route = json.loads((PARENT / 'image_only_route.json').read_text())
    pred_file = PARENT / 'inference/prediction.json'
    pred = json.loads(pred_file.read_text())
    source = Path(detail['source'])
    protected = [source, pred_file, PARENT / 'expert_appearance_review_01.json']
    before = {str(p): sha(p) for p in protected}
    assert before[str(source)] == detail['source_sha256']
    selected = [m for m in detail['markers'] if m['number'] in [4, 5]]
    xr = [m['source_x'] for m in selected]
    path = np.array(route['path_xy_crop_pixels']) + np.array(route['source_crop_xyxy'][:2])
    arc = np.array(route['distance_along_path_pixels'])
    assert np.all(np.diff(path[:, 0]) > 0)
    arc_interval = np.interp(xr, path[:, 0], arc)
    early_path_gap = arc_interval - pred['T_path_pixels']
    assert np.all(early_path_gap > 0)
    rule = pred['rule']; chosen = rule['selected']
    audit = next(a for a in rule['candidate_audits'] if a['original_candidate'] == chosen)
    a, b = chosen['mixed_run']['start'], chosen['mixed_run']['stop_exclusive']
    record = {
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'expert': 'Gonzalo Villarino', 'verbatim': 'Btw 4-5',
        'interpretation': 'Approximate left edge of the mixed-cell zone lies between numbered guides 4 and 5 on the main root.',
        'reference_T_source_x_interval': xr,
        'reference_T_source_endpoints': [[m['source_x'], m['main_root_center_source_y']] for m in selected],
        'reference_T_local_path_interval_pixels': arc_interval.tolist(),
        'exact_reference_T': None,
        'no_midpoint_chosen': True,
        'interval_kind': 'Expert approximate position range; not a statistical confidence interval or the full width of the mixed zone.',
        'model_T_source_xy': pred['T_source_xy'],
        'model_T_local_path_pixels': pred['T_path_pixels'],
        'model_is_too_far_left_by_path_pixels': early_path_gap.tolist(),
        'distance_scope': 'Along the checked crop-local main-root route. No complete-root length or micrometre conversion; A04 calibration unconfirmed.',
        'model_status_on_this_case': 'T placement rejected by expert; outside the supplied reference interval.',
        'earlier_A_B_C_review': 'Retained as qualitative field descriptions. B overlaps only part of the approximate reference interval; its whole rectangle is not automatically relabelled.',
        'no_new_pixel_annotations': True,
        'model_changed': False, 'training_occurred': False,
        'reference_used_to_move_model_T': False,
        'next': 'Retain as a failed fixed-v1 check. Before a future v2, review representative false-mixed appearances and maintain separate evaluation roots. This correction is not a new model prediction.'
    }
    save(OUT / 'expert_reference_interval.json', record)
    diagnosis = {
        'source': 'Existing saved predictions only; no new inference.',
        'selected_mixed_run': chosen['mixed_run'],
        'mixed_run_source_centres': [[pred['centres_xy_crop_pixels'][i][0] + 3400, pred['centres_xy_crop_pixels'][i][1]] for i in [a, b - 1]],
        'first_mixed_field_scores_Endoderm_Periderm': pred['presence_scores'][a],
        'interpretation': 'The head predicts both appearances for 24 consecutive overlapping windows beginning well before the expert interval. This is not just an isolated early mixed burst.',
        'continuity_guard_audit': audit,
        'guard_result': 'No long clear-Endoderm return before the supporting Periderm run; the fixed continuity rule retains this candidate.',
        'not_established': ['What visual feature caused the false mixed predictions', 'A causal effect of blur or lateral-branch context', 'That a manual shift or extra iterations will generalize'],
        'window_counts_are_not_independent_samples': True
    }
    save(OUT / 'saved_prediction_diagnosis.json', diagnosis)
    bounds = detail['source_box_xyxy']
    im = Image.open(source).convert('RGB').crop(bounds)
    fig = Image.new('RGB', (1740, 900), 'white'); fig.paste(im, (20, 130))
    d = ImageDraw.Draw(fig)
    font = lambda n: ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', n)
    blue = '#28b7ff'
    d.text((20, 12), 'A04 root 1: saved model prediction versus your reference', font=font(30), fill='black')
    d.text((20, 59), 'Solid blue: model T. Dashed blue boundaries: your approximate T interval, between 4 and 5.', font=font(24), fill='#334155')
    tx = 20 + pred['T_source_xy'][0] - bounds[0]
    d.line((tx, 130, tx, 790), fill=blue, width=3)
    d.text((tx + 10, 733), 'Model T: too far left', font=font(24), fill=blue, stroke_width=1, stroke_fill='black')
    for m in detail['markers']:
        px = 20 + m['source_x'] - bounds[0]
        yy = 130 + m['main_root_center_source_y'] - bounds[1]
        d.text((px - 9, max(138, yy - 140)), str(m['number']), font=font(28), fill='white', stroke_width=1, stroke_fill='black')
        if m['number'] in [4, 5]:
            for y in range(130, 790, 24):
                d.line((px, y, px, min(y + 12, 790)), fill=blue, width=2)
        else:
            d.line((px, yy - 105, px, yy - 90), fill='white', width=2)
    left, right = [20 + x - bounds[0] for x in xr]
    d.line((left, 115, right, 115), fill=blue, width=3)
    d.text((left + 10, 84), 'Your T interval: 4–5', font=font(24), fill='#075985')
    d.text((20, 817), 'Your reference has been saved. The model line has not moved; v1 remains locked.', font=font(24), fill='black')
    d.text((20, 856), 'This range marks uncertainty in the boundary location, not the entire mixed-cell zone.', font=font(23), fill='#334155')
    fig.save(OUT / 'A04_ROOT1_MODEL_VS_REFERENCE.png')
    verification = subprocess.run([sys.executable, str(BUNDLE / 'verify_checkpoint.py'), '--require-read-only'], capture_output=True, text=True, check=True)
    assert before == {str(p): sha(p) for p in protected}
    save(OUT / 'verification.json', {'protected_files_unchanged': before, 'checkpoint': json.loads(verification.stdout), 'figure_visual_check': 'Pending'})
    shutil.copy2(__file__, OUT / Path(__file__).name)
    note = ('16 November 2026 — A04 root1 expert T interval recorded: guides 4–5\n'
            'Gonzalo answers Btw 4-5. Saved approximate source-x interval 6000–6300, not an exact midpoint. '
            'v1 T at x4897.161375 remains too far left; original prediction and read-only checkpoint unchanged. '
            'Saved-score diagnosis: 24 consecutive overlapping mixed calls begin early; no long Endoderm-only return triggers the continuity guard. '
            'Cause in image features is not established. No retraining, threshold change, pixel relabelling or micrometre conversion. '
            'Earlier A/B/C descriptions retained. Source comparison and interval record saved in 04_Results/71_Locked_A04_Root1_Check/expert_T_review_02. '
            'Word manual not edited.\n')
    with (ROOT / '05_Notes/03_Progress_and_Next_Step.txt').open('a') as f:
        f.write('\n' + note)
    (OUT / 'READ_ME.txt').write_text(note + '\nThis is an expert reference and a read-only diagnosis, not an improved-model result.\n')
    print(json.dumps({'reference_x': xr, 'model_x': pred['T_source_xy'][0], 'early_path_pixels': early_path_gap.tolist(), 'figure': str(OUT / 'A04_ROOT1_MODEL_VS_REFERENCE.png')}, indent=2))


if __name__ == '__main__':
    main()
