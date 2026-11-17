"""Verify file fingerprints and optional read-only permissions in this bundle."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse
import hashlib
import json

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def verify(bundle,require_read_only=False):
    bundle=Path(bundle)
    if sha(bundle/'MANIFEST.json')!=(bundle/'MANIFEST.sha256').read_text().strip():raise ValueError('Manifest fingerprint changed.')
    manifest=json.loads((bundle/'MANIFEST.json').read_text())
    expected=set(manifest['files'])|{'MANIFEST.json','MANIFEST.sha256'}
    actual={str(p.relative_to(bundle)) for p in bundle.rglob('*') if p.is_file()}
    if expected!=actual:raise ValueError(f'File inventory changed: {sorted(expected^actual)}')
    for rel,entry in manifest['files'].items():
        path=bundle/rel
        if path.is_symlink() or sha(path)!=entry['sha256'] or path.stat().st_size!=entry['bytes']:raise ValueError(f'File changed: {rel}')
    if require_read_only:
        for path in [bundle,*bundle.rglob('*')]:
            if path.stat().st_mode&0o222:raise ValueError(f'Writable checkpoint path: {path}')
    return {'passed':True,'files':len(actual),'version':manifest['version'],
            'read_only_checked':require_read_only,'manifest_sha256':sha(bundle/'MANIFEST.json')}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-read-only',action='store_true');a=p.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parent,a.require_read_only),indent=2))
