"""Install RootSpan and import the original, checksum-verified PAT model.

Project author: Gonzalo Villarino. Original PAT model: Salk HPI.
Run with Python 3.13 on macOS Apple silicon. No training occurs.
"""
from pathlib import Path
import argparse, hashlib, json, platform, shutil, subprocess, sys, tempfile, venv
ROOT = Path(__file__).resolve().parent
ORIGINAL_HASH = 'fb07dc1f1a254f7ba067de7f9b8b491bc7bb7f1c8da4564159ad23bcbc208f90'
STATE_HASH = '71ba82d7eab40c0b8489f47ee373fd0ea08f2e2e33457c62152a1f8240d68348'
def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def import_weights(source):
    if digest(source) != ORIGINAL_HASH:
        raise ValueError('PAT checkpoint checksum differs. Obtain the exact upstream file linked in README.md.')
    import importlib
    import torch
    import segmentation_models_pytorch as smp
    names = torch.serialization.get_unsafe_globals_in_checkpoint(source)
    allowed = json.loads((ROOT / '00_Software/PAT_Reference/download_and_conversion_record.json').read_text())['allowed_globals']
    if set(names) != set(allowed):
        raise ValueError('Unexpected checkpoint classes; model was not loaded.')
    classes = [getattr(importlib.import_module(n.rsplit('.', 1)[0]), n.rsplit('.', 1)[1]) for n in names]
    with torch.serialization.safe_globals(classes):
        model = torch.load(source, map_location='cpu', weights_only=True)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    state = model.state_dict()
    check = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=None,
                            in_channels=3, classes=4, activation='sigmoid')
    check.load_state_dict(state, strict=True)
    target = ROOT / '00_Software/PAT_Reference/pat_state_dict.pt'
    if target.exists():
        if digest(target) != STATE_HASH:
            raise ValueError('Existing model differs; it was not overwritten.')
    else:
        # Keep the final model absent on failure, and preserve the archive basename.
        with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
            candidate = Path(temporary) / target.name
            torch.save(state, candidate)
            if digest(candidate) != STATE_HASH:
                raise ValueError('Converted model bytes differ. Temporary output removed; check the pinned PyTorch version before retrying.')
            candidate.replace(target)
    second = ROOT / '06_Models/Locked_Transition_v1_20261116/weights/pat_state_dict.pt'
    second.parent.mkdir(parents=True, exist_ok=True)
    if second.exists():
        if digest(second) != STATE_HASH:
            raise ValueError('Existing transition encoder differs; it was not overwritten.')
    else:
        shutil.copy2(target, second)
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pat-checkpoint', type=Path, required=True,
                   help='Original PAT .pth from the upstream link in README.md')
    p.add_argument('--skip-env-install', action='store_true', help='Use already installed, pinned environments')
    p.add_argument('--import-weights-only', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if sys.version_info[:2] != (3, 13):
        p.error('Use Python 3.13 to install this release.')
    if sys.platform != 'darwin':
        p.error('This first package targets macOS; other platforms have not been verified.')
    if platform.machine() != 'arm64':
        p.error('This recorded release requires an Apple silicon Mac (arm64).')
    if not args.pat_checkpoint.is_file() or digest(args.pat_checkpoint) != ORIGINAL_HASH:
        p.error('Provide the original PAT checkpoint with the expected SHA-256; see README.md.')
    if args.import_weights_only:
        import_weights(args.pat_checkpoint)
        return
    for name in ('ml-env', 'pat-audit-env'):
        env = ROOT / '00_Software' / name
        if not args.skip_env_install:
            if not env.exists():
                venv.EnvBuilder(with_pip=True).create(env)
            subprocess.run([str(env / 'bin/python'), '-m', 'pip', 'install', '-r',
                            str(ROOT / 'requirements' / (name + '.txt'))], check=True)
        if not (env / 'bin/python').is_file():
            p.error('Missing environment: ' + str(env))
    python = ROOT / '00_Software/ml-env/bin/python'
    subprocess.run([str(python), str(Path(__file__).resolve()), '--pat-checkpoint',
                    str(args.pat_checkpoint.resolve()), '--import-weights-only'], check=True)
    for folder in ('runs', 'uploads'):
        (ROOT / '09_Periderm_GUI' / folder).mkdir(exist_ok=True)
    subprocess.run([str(python), str(ROOT / '09_Periderm_GUI/approved_release.py')], check=True)
    subprocess.run([str(python), str(ROOT / '06_Models/Locked_Transition_v1_20261116/verify_checkpoint.py')], check=True)
    print('RootSpan is ready. Double-click RootSpan.command in the project folder.')
if __name__ == '__main__':
    main()
