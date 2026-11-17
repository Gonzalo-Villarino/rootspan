"""Prepare source-only views before using the locked transition detector."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import numpy as np
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '02_Data/13_Locked_Model_Checks_20261116/A04_root1/source_review_02'
SOURCE = ROOT / '02_Data/07_Additional_Archived_Annotations_20261116/A04_Slide2.png'
BUNDLE = ROOT / '06_Models/Locked_Transition_v1_20261116'
Image.MAX_IMAGE_PIXELS = 300_000_000


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    assert not (OUT / 'source_review_plan.json').exists()
    before = sha(SOURCE)
    assert before == '529ad2cf67513b8a4a4bae737261ff068fca498c166a5bf2d073d59ad713fe4c'
    checked = subprocess.run([sys.executable, str(BUNDLE / 'verify_checkpoint.py'), '--require-read-only'],
                             text=True, capture_output=True, check=True)
    integrity = json.loads(checked.stdout)
    im = Image.open(SOURCE).convert('RGB')
    bounds = (3400, 0, 9000, 1500)
    native = im.crop(bounds)
    native.save(OUT / 'source_native.png')
    font = lambda n: ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', n)
    fig = Image.new('RGB', (1780, 1080), 'white')
    d = ImageDraw.Draw(fig)
    d.text((20, 12), 'Next check: A04, root 1', font=font(32), fill='black')
    d.text((20, 58), 'Original microscopy. No model prediction or T line is shown.', font=font(25), fill='#334155')
    fig.paste(native.resize((1680, 450), Image.Resampling.LANCZOS), (50, 106))
    fields = []
    for i, (name, cx, low, high) in enumerate([('A', 4200, 500, 1050), ('B', 6000, 100, 450), ('C', 7000, 0, 350)]):
        band = np.array(im.crop((cx - 200, low, cx + 200, high)))
        cy = low + int(np.argmax(gaussian_filter1d(band[:, :, 1].astype(float).sum(1), 10)))
        y0 = max(0, cy - 140)
        box = [cx - 200, y0, cx + 200, y0 + 280]
        crop = im.crop(box)
        crop.save(OUT / f'{name}_native.png')
        x0 = 50 + (box[0] - bounds[0]) * .3
        yy0 = 106 + (box[1] - bounds[1]) * .3
        d = ImageDraw.Draw(fig)
        d.rectangle((x0, yy0, x0 + 120, yy0 + 84), outline='white', width=2)
        d.text((x0 + 2, max(108, yy0 - 34)), name, font=font(27), fill='white', stroke_width=1, stroke_fill='black')
        xx = 20 + i * 590
        d.text((xx, 577), f'{name}: main-root cells', font=font(26), fill='black')
        fig.paste(crop.resize((560, 392), Image.Resampling.NEAREST), (xx, 620))
        fields.append({'name': name, 'source_box_xyxy': box, 'biological_label': None})
    d = ImageDraw.Draw(fig)
    d.text((20, 1035), 'A, B, C: Endoderm, mixed / transition, Periderm, or uncertain? Look only at the main root.', font=font(24), fill='#334155')
    fig.save(OUT / 'A04_ROOT1_SOURCE_REVIEW.png')
    record = {
        'prepared_utc': datetime.now(timezone.utc).isoformat(),
        'source': str(SOURCE), 'source_sha256': before, 'root_number_from_top': 1,
        'source_box_xyxy': list(bounds), 'review_fields': fields,
        'layout_revision': 'Supersedes the initial source preparation in the parent folder. Detail fields moved away from lateral-root junctions after visual inspection, before predictions or expert labels.',
        'selection': 'Root 1 selected from the existing source review for visible cells and continuous main-root context. Archived Periderm outline helped locate the area; no locked-detector predictions consulted.',
        'selection_limit': 'Source-quality and archive-guided selection, not random sampling. A04 slide was used in earlier method checks; this is an additional fixed-version check, not a pristine independent test.',
        'teaching_status': 'The current appearance head was taught on COL0 roots 2 and 3, not this A04 root. Published PAT encoder exposure is unknown.',
        'image_only_preparation': True, 'model_prediction_done': False, 'training_done': False,
        'reference_status': 'Awaiting expert review; no new labels inferred from old outlines.',
        'checkpoint': str(BUNDLE), 'checkpoint_integrity': integrity,
        'display': 'Overview downscaled; detail fields enlarged with nearest-neighbour interpolation. No brightness adjustment, sharpening, or restoration.',
        'next': 'Record expert main-root appearance judgement, then run the unchanged checkpoint with a separately checked main-root path.',
        'calibration': 'A04 scale not confirmed; do not report micrometre lengths yet.'
    }
    assert sha(SOURCE) == before
    (OUT / 'source_review_plan.json').write_text(json.dumps(record, indent=2) + '\n')
    shutil.copy2(__file__, OUT / Path(__file__).name)
    (OUT / 'READ_ME.txt').write_text('A04 ROOT 1 — SOURCE REVIEW BEFORE LOCKED-MODEL PREDICTION\n\nA/B/C are source microscopy views, not model predictions or tissue annotations. No training, prediction or model changes occurred. The checkpoint remains read-only and its file verification passed. This is an additional root check on a familiar slide. Expert review is pending. The next step is to run the unchanged checkpoint after checking the main-root path.\n')
    with (ROOT / '05_Notes/03_Progress_and_Next_Step.txt').open('a') as f:
        f.write('\n16 November 2026 — A04 root 1 source review layout 02\nMoved A/B/C detail fields away from lateral-root junctions after source-only visual QA. This supersedes the initial detail layout, before any expert label or locked-model prediction. Source and checkpoint fingerprints pass. Current figure and plan: 02_Data/13_Locked_Model_Checks_20261116/A04_root1/source_review_02. No training or model changes; Word manual not edited.\n')
    print(json.dumps({'figure': str(OUT / 'A04_ROOT1_SOURCE_REVIEW.png'), 'fields': fields}, indent=2))


if __name__ == '__main__':
    main()
