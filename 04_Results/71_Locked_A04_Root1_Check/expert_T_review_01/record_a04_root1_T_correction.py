"""Record expert rejection of T and prepare numbered source-only position guides."""
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
OUT = PARENT / 'expert_T_review_01'
BUNDLE = ROOT / '06_Models/Locked_Transition_v1_20261116'
Image.MAX_IMAGE_PIXELS = 300_000_000


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    OUT.mkdir(exist_ok=False)
    prediction_path = PARENT / 'inference/prediction.json'
    prediction = json.loads(prediction_path.read_text())
    route = json.loads((PARENT / 'image_only_route.json').read_text())
    source = Path(route['source_image'])
    before = {str(p): sha(p) for p in [source, prediction_path, PARENT / 'expert_appearance_review_01.json']}
    assert before[str(source)] == route['source_sha256']
    recorded = datetime.now(timezone.utc).isoformat()
    correction = {
        'recorded_utc': recorded, 'expert': 'Gonzalo Villarino', 'verbatim': 'is too far left',
        'case': 'A04 root 1', 'checkpoint': 'Transition_v1_20261116',
        'model_T_source_xy': prediction['T_source_xy'],
        'expert_judgement': 'Automatic T is too far left; its position is not accepted.',
        'exact_reference_T': None, 'reference_interval': None,
        'earlier_appearance_labels_retained': 'A Endoderm; B mixed/transition; C Periderm.',
        'scope': 'Directional position correction, not a new exact coordinate or pixel annotation.',
        'status': 'Does not pass expert T-placement check on this root.',
        'model_unchanged': True, 'prediction_unchanged': True, 'training_occurred': False,
        'next': 'Ask for a reference position or range using source-only numbered guides. Any future method change must be in a separate version and this root then counts as development data for that change.'
    }
    save(OUT / 'expert_T_correction.json', correction)
    im = Image.open(source).convert('RGB')
    bounds = [4800, 80, 6500, 740]
    native = im.crop(bounds); native.save(OUT / 'source_native.png')
    fig = Image.new('RGB', (1740, 870), 'white')
    fig.paste(native, (20, 110))
    font = lambda n: ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', n)
    d = ImageDraw.Draw(fig)
    d.text((20, 12), 'A04 root 1: where does the mixed-cell zone begin?', font=font(30), fill='black')
    d.text((20, 59), 'Blue = unchanged model T, which you judged too far left. Numbers are position guides.', font=font(24), fill='#334155')
    oldx = 20 + prediction['T_source_xy'][0] - bounds[0]
    d.line((oldx, 110, oldx, 770), fill='#28b7ff', width=3)
    d.text((oldx + 10, 705), 'Model T', font=font(24), fill='#28b7ff', stroke_width=1, stroke_fill='black')
    path = np.array(route['path_xy_crop_pixels']) + np.array(route['source_crop_xyxy'][:2])
    markers = []
    for n, x in enumerate([5100, 5400, 5700, 6000, 6300], 1):
        y = float(np.interp(x, path[:, 0], path[:, 1]))
        px = 20 + x - bounds[0]
        # Guide ends are above/below the main-root tissue, leaving cell contours visible.
        top = 110 + y - bounds[1] - 90
        bottom = 110 + y - bounds[1] + 90
        label_y = max(116, top - 50)
        d.text((px - 9, label_y), str(n), font=font(28), fill='white', stroke_width=1, stroke_fill='black')
        if label_y + 34 < top:
            d.line((px, label_y + 34, px, top), fill='white', width=2)
        d.line((px, bottom, px, bottom + 22), fill='white', width=2)
        markers.append({'number': n, 'source_x': x, 'main_root_center_source_y': y, 'biological_meaning': None})
    d.text((20, 794), 'Choose a number or an interval. Use the main-root cells; ignore lateral-root cells.', font=font(24), fill='black')
    d.text((20, 832), 'No retraining, new prediction, or manual movement of the blue line.', font=font(23), fill='#334155')
    fig.save(OUT / 'A04_ROOT1_NUMBERED_T_REVIEW.png')
    save(OUT / 'numbered_review_details.json', {
        'source': str(source), 'source_sha256': before[str(source)], 'source_box_xyxy': bounds,
        'markers': markers, 'display': 'Native source pixels, no brightness change or rescaling.',
        'selection': 'Source area between the rejected model T and the already-reviewed mixed field B, with right context. Guides do not define new biological labels.',
        'model_T_preserved': prediction['T_source_xy'], 'expert_reference_pending': True,
        'guide_center_geometry': 'Previously checked fluorescence-derived main-root path; no tissue class inferred.'})
    checked = subprocess.run([sys.executable, str(BUNDLE / 'verify_checkpoint.py'), '--require-read-only'],
                             capture_output=True, text=True, check=True)
    assert before == {str(p): sha(p) for p in [source, prediction_path, PARENT / 'expert_appearance_review_01.json']}
    save(OUT / 'verification.json', {'protected_files_unchanged': before,
         'locked_checkpoint': json.loads(checked.stdout), 'figure_visual_check': 'Pending'})
    shutil.copy2(__file__, OUT / Path(__file__).name)
    note = ('16 November 2026 — Expert rejects locked v1 T on A04 root 1\n'
            'Gonzalo says the automatic blue T is too far left. Recorded directional error; no exact replacement coordinate inferred. '
            'Earlier A/B/C appearance reviews remain unchanged. Correct classifications in sampled areas did not establish a correct transition boundary. '
            'The checkpoint remains read-only and the original prediction is preserved. Prepared numbered native source guides for a reference position/range; no training or new inference. '
            'Files: 04_Results/71_Locked_A04_Root1_Check/expert_T_review_01. Word manual not edited.\n')
    with (ROOT / '05_Notes/03_Progress_and_Next_Step.txt').open('a') as f:
        f.write('\n' + note)
    readme = PARENT / 'READ_ME.txt'
    readme.write_text('EXPERT UPDATE — 16 November 2026\nThe model T was judged too far left. Its position is NOT accepted on this root. The earlier A/B/C appearance labels are retained. See expert_T_review_01 for the recorded correction and source-only numbered review. The locked checkpoint and prediction remain unchanged.\n\n' + readme.read_text())
    print(OUT / 'A04_ROOT1_NUMBERED_T_REVIEW.png')


if __name__ == '__main__':
    main()
