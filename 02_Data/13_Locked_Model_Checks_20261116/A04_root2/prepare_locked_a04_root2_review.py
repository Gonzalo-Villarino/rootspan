"""Source-only A04 root-2 review. No training or prediction."""
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
OUT = ROOT / '02_Data/13_Locked_Model_Checks_20261116/A04_root2'
BUNDLE = ROOT / '06_Models/Locked_Transition_v1_20261116'
Image.MAX_IMAGE_PIXELS = 300_000_000


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    assert not (OUT / 'source_review_plan.json').exists()
    selected = json.loads((OUT / 'preparation_selection.json').read_text())
    source = Path(selected['source']); bounds = selected['source_box_xyxy']
    assert sha(source) == selected['source_sha256']
    verified = subprocess.run([sys.executable, str(BUNDLE / 'verify_checkpoint.py'), '--require-read-only'], text=True, capture_output=True, check=True)
    im = Image.open(source).convert('RGB')
    native = im.crop(bounds)
    assert np.array_equal(np.asarray(native), np.asarray(Image.open(OUT / 'source_native.png')))
    fig = Image.new('RGB', (1780, 1035), 'white')
    fig.paste(native.resize((1560, 420), Image.Resampling.LANCZOS), (110, 100))
    fields = []
    for i, (name, cx) in enumerate([('A', 7400), ('B', 8500), ('C', 10000)]):
        low, high = 2380, 2980
        band = np.asarray(im.crop((cx - 200, low, cx + 200, high)))
        cy = low + int(np.argmax(gaussian_filter1d(band[:, :, 1].astype(float).sum(1), 10)))
        box = [cx - 200, cy - 140, cx + 200, cy + 140]
        field = im.crop(box); field.save(OUT / f'{name}_native.png')
        fig.paste(field.resize((560, 392), Image.Resampling.NEAREST), (20 + i * 590, 586))
        fields.append({'name': name, 'source_box_xyxy': box, 'biological_label': None})
    # Draw text after pasting all panels so the final raster is inspected as saved.
    d = ImageDraw.Draw(fig)
    font = lambda n: ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', n)
    d.text((20, 12), 'Next check: A04, root 2', font=font(32), fill='black')
    d.text((20, 58), 'Original microscopy. Your review first; no model prediction or T line yet.', font=font(25), fill='#334155')
    for i, field in enumerate(fields):
        x0, y0, x1, y1 = field['source_box_xyxy']
        px = 110 + (x0 - bounds[0]) * .3; py = 100 + (y0 - bounds[1]) * .3
        d.rectangle((px, py, px + (x1 - x0) * .3, py + (y1 - y0) * .3), outline='white', width=2)
        d.text((px + 2, max(103, py - 34)), field['name'], font=font(27), fill='white', stroke_width=1, stroke_fill='black')
        d.text((20 + i * 590, 540), f'{field["name"]}: main-root cells', font=font(26), fill='black')
    d.text((20, 998), 'A, B, C: Endoderm, mixed / transition, Periderm, or uncertain? Look only at the main root.', font=font(24), fill='#334155')
    fig.save(OUT / 'A04_ROOT2_SOURCE_REVIEW.png')
    record = dict(selected,
        prepared_utc=datetime.now(timezone.utc).isoformat(), review_fields=fields,
        selection_detail='Next numbered root after root 1. Detail fields placed away from conspicuous lateral junctions using source microscopy alone; no detector scores consulted.',
        selection_limit='Additional root on the familiar A04 slide, not a fresh independent slide or pristine blind test. Original PAT encoder exposure unknown.',
        teaching_status='The frozen appearance head was taught on COL0 roots 2 and 3, not this A04 root.',
        image_only_preparation=True, training_done=False,
        reference_status='Awaiting expert A/B/C appearance review; old archive polygons not treated as confirmed tissue truth.',
        checkpoint=str(BUNDLE), checkpoint_integrity=json.loads(verified.stdout),
        display='Overview downscaled; details enlarged nearest-neighbour. Native crops preserved. No brightness adjustment, sharpening, or restoration.',
        calibration='A04 scale not confirmed; no micrometre lengths.',
        next='Record expert review, prepare and visually check image-only main-root route, then use the unchanged locked checkpoint.',
        visual_qa='Pending')
    assert sha(source) == selected['source_sha256']
    (OUT / 'source_review_plan.json').write_text(json.dumps(record, indent=2) + '\n')
    (OUT / 'READ_ME.txt').write_text('A04 ROOT 2 — SOURCE REVIEW FIRST\n\nA/B/C show original main-root microscopy for expert review. No prediction, training, new tissue annotation or model change occurred. The read-only checkpoint integrity check passed. This is another root on a familiar slide, not independent final validation. Source and native detail crops are preserved. Wait for the expert appearance review before inference.\n')
    shutil.copy2(__file__, OUT / Path(__file__).name)
    with (ROOT / '05_Notes/03_Progress_and_Next_Step.txt').open('a') as f:
        f.write('\n16 November 2026 — A04 root 2 selected for the next locked-v1 check\nUser requests another root, same locked v1, image first and slow guidance. Selected next root 2 on familiar A04 before new predictions. Prepared source-only overview and three detail views away from conspicuous branch junctions. Labels pending; source and read-only checkpoint integrity verified. No training or inference. Saved 02_Data/13_Locked_Model_Checks_20261116/A04_root2. Word manual not edited.\n')
    print(json.dumps({'figure': str(OUT / 'A04_ROOT2_SOURCE_REVIEW.png'), 'fields': fields}, indent=2))


if __name__ == '__main__':
    main()
