"""Verify the RootSpan publication package; not an accuracy certificate."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
PROJECT = Path(__file__).resolve().parents[1]
def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()
def verify_release(project=PROJECT, **unused):
    project = Path(project)
    manifest_file = project / 'ROOTSPAN_MANIFEST.json'
    record = json.loads(manifest_file.read_text())
    for relative, expected in record['runtime_files'].items():
        path = project / relative
        if not path.is_file():
            raise ValueError('RootSpan setup is incomplete: ' + relative + '. See README.md.')
        if digest(path) != expected['sha256']:
            raise ValueError('RootSpan runtime file changed: ' + relative)
    for name, expected in record['required_packages'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('RootSpan package version differs: ' + name + '. See requirements/ml-env.txt.')
    return {'release_id': record['release_id'], 'app_version': record['app_version'],
            'manifest_sha256': digest(manifest_file), 'passed': True,
            'checked_files': len(record['runtime_files']),
            'environment_checked': True, 'snapshot_checked': False,
            'scope': 'Public package integrity and main-environment package versions; not biological validation.'}
if __name__ == '__main__':
    print(json.dumps(verify_release(), indent=2))
