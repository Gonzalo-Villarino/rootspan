"""Record successful replay, seal this checkpoint, and verify its final contents."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess

from build_locked_transition_v1 import BUNDLE, ROOT, VERSION, manifest, save


def main():
    result_dir = ROOT / '04_Results/70_Locked_Checkpoint_Verification'
    report = json.loads((result_dir / 'self_check.json').read_text())
    assert report['passed'] and report['image_to_scores_max_absolute_difference'] == 0.0
    assert len(report['saved_profile_replays']) == 4
    assert all(r['passed'] for r in report['saved_profile_replays'])
    assert BUNDLE.stat().st_mode & 0o200, 'Checkpoint is already sealed.'
    shutil.copy2(result_dir / 'self_check.json', BUNDLE / 'evidence/self_check.json')
    shutil.copy2(__file__, BUNDLE / 'source_snapshot' / Path(__file__).name)
    save(BUNDLE / 'evidence/review_status.json', {
        'status': 'Independent review unavailable; local verification completed.',
        'prior_review_attempt': str(ROOT / '04_Results/69_Transition_Continuity_Check/review/review_status.json'),
        'prior_attempt_outcome': 'external reviewer subscription login required; no review sent.',
        'checkpoint_packaging_independently_reviewed': False,
        'biological_review': 'Gonzalo approved the solid blue T on 2_12 root 1.',
        'local_checks': 'Exact image-to-score replay, exact approved T, and four saved-profile decoder replays passed.'
    })
    python = ROOT / '00_Software/ml-env/bin/python'
    readme = f'''LOCKED TRANSITION CHECKPOINT v1 — 16 November 2026

This is the version Gonzalo approved: the solid blue T on 2_12 root 1.

WHAT IS LOCKED?
The learned image-recognition weights, image preparation, T-selection rule,
settings, source code and example inputs are preserved together. The checkpoint
files are read-only. File fingerprints detect accidental changes. Any later
improvements belong in a NEW version folder.

WHAT WAS VERIFIED?
The saved copy was run from outside the project folder. It reproduced every
appearance score exactly and placed T at exactly the approved position.
The rule also reproduced all four saved profiles: the accepted COL0 and A04
positions stayed unchanged; 20_29 still returned no supported T.
These checks establish computational reproduction, not accuracy on new roots.

WHAT DOES IT DO?
Given a microscopy crop and a checked path along the main root, it looks for
Endoderm-like and Periderm-like appearances. A mixed call requires both.
It selects a supported T or returns no T. The continuity rule rejects an early
candidate when a long stretch of clear Endoderm contradicts that candidate.
The blue T represents the left edge of the mixed-cell zone. For the agreed
regional measurements, the mixed zone belongs with Periderm.

This checkpoint combines an image-recognition model with a T-selection rule.
The latest improvement changed the rule; it did not retrain the weights.
It is not yet a validated program for unattended whole-slide length measurements.
It does not directly outline or count individual cells.

WHAT HAPPENS NEXT?
Keep this version fixed and check it on additional roots. Record its prediction
before comparing with Gonzalo's biological judgement. Only after checking T and
the measurement endpoints should we assess lengths in micrometres. 2_12 helped
develop this rule, so it is not an independent final test of this version.

SAVED EVIDENCE
ACCEPTANCE.json records Gonzalo's approval and exact coordinates.
evidence/approved_result.png is the original candidate picture he approved;
its historical captions predate approval. ACCEPTANCE.json is the latest status.
evidence/self_check.json records the successful computational replay.
Independent external reviewer review was unavailable because subscription login was absent;
no review was sent. Local checks and Gonzalo's biological review are recorded.

TECHNICAL COMMANDS — FOR LATER REFERENCE
No command is needed now. The following first command only checks file integrity:

"{python}" "{BUNDLE / 'verify_checkpoint.py'}" --require-read-only

To reproduce the approved example, choose a NEW results folder outside this
checkpoint and run:

"{python}" "{BUNDLE / 'run_checkpoint.py'}" --self-check --output "/absolute/path/to/a/new/results_folder"

For another root, use the same Python and runner with:
--image "/absolute/path/to/crop.png" --route "/absolute/path/to/route.json" --output "/absolute/path/to/a/new/results_folder"

Route keys: path_xy_crop_pixels and distance_along_path_pixels. Optional
source_crop_xyxy supplies the full-slide offset. The checked path and image must
match, with Endoderm toward the left and hypocotyl toward the right. Whole-slide
root finding and path preparation remain separate checked steps.

SOFTWARE AND PRESERVATION
The existing project ml-env environment runs this copy. environment.json and
requirements.freeze.txt record its software versions. No training or downloads
occur during inference. Other hardware or software may change numerical results;
rerun the self-check after such changes. Previous models and source images remain
unchanged. The earlier segmentation U-Net is a separate saved model.
'''
    (BUNDLE / 'READ_ME_FIRST.txt').write_text(readme)
    save(BUNDLE / 'LOCK_RECORD.json', {
        'version': VERSION,
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'Approved research checkpoint; ready to seal after successful replay.',
        'permissions_on_sealing': {'files': '0444', 'directories': '0555'},
        'future_changes': 'Create a new version; do not edit this checkpoint.',
        'computational_reproduction': 'Exact on the current Mac and software.',
        'scope_limit': 'Not yet validated for unattended whole-slide measurements.'
    })
    manifest()
    for path in BUNDLE.rglob('*'):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((p for p in BUNDLE.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        path.chmod(0o555)
    BUNDLE.chmod(0o555)
    completed = subprocess.run([str(python), str(BUNDLE / 'verify_checkpoint.py'), '--require-read-only'],
                               cwd='/tmp', text=True, capture_output=True, check=True)
    verification = json.loads(completed.stdout)
    assert verification['passed'] and verification['read_only_checked']
    save(result_dir / 'final_verification.json', verification)
    provenance = json.loads((BUNDLE / 'provenance.json').read_text())
    with (ROOT / '06_Models/working_model.pt').open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == provenance['working_U_Net_sha256']
    pointer = ROOT / '06_Models/current_transition_checkpoint.json'
    if pointer.exists():
        raise FileExistsError('Preserve and review the existing checkpoint pointer before replacing it.')
    save(pointer, {
        'version': VERSION, 'checkpoint': str(BUNDLE),
        'status': 'Locked, read-only, computational reproduction verified.',
        'manifest_sha256': verification['manifest_sha256'],
        'verification': str(result_dir / 'final_verification.json'),
        'biological_acceptance': str(BUNDLE / 'ACCEPTANCE.json'),
        'use': 'Fixed transition detector for additional root checks.',
        'not_yet': 'Validated unattended whole-slide length measurement.',
        'working_pixel_segmentation_unet_unchanged': True
    })
    note = f'''16 November 2026 — APPROVED TRANSITION CHECKPOINT LOCKED
Gonzalo approved the solid blue T on 2_12 root 1 and requested a lock.
Saved the exact weights, image preparation, continuity rule, settings, software
versions, source code, inputs and biological acceptance as {VERSION}.
The complete copy reproduces the original appearance scores exactly (maximum
difference 0.0) and the approved T at source x=8414.342292421163,
y=411.8250414267261. All four saved-profile decoder replays passed, retaining
COL0/A04 positions and no supported T for 20_29.
All checkpoint files are read-only; final inventory and fingerprints passed.
Checkpoint: {BUNDLE}
Current transition pointer: {pointer}
Verification: {result_dir / 'final_verification.json'}
Next: test this version unchanged on additional roots, then assess measurement
endpoints and lengths. This is a reproducible research checkpoint, not a claim
of validated unattended measurements. 2_12 informed rule development.
Original images, annotations and the earlier segmentation U-Net are unchanged.
Independent review unavailable: prior helper attempt required external reviewer login and
sent nothing. The Word manual was not edited.
'''
    model_readme = ROOT / '06_Models/READ_ME.txt'
    old_text = model_readme.read_text()
    model_readme.write_text(note + '\n--- HISTORICAL MODEL RECORDS BELOW ---\n\n' + old_text)
    with (ROOT / '05_Notes/03_Progress_and_Next_Step.txt').open('a') as f:
        f.write('\n\n' + note)
    print(json.dumps(verification, indent=2))
    print(f'Locked checkpoint: {BUNDLE}')


if __name__ == '__main__':
    main()
