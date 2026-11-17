"""Check saved V2 source fingerprints before a labelled V2 execution."""
from pathlib import Path
import hashlib,json

PROJECT=Path(__file__).resolve().parents[1]
BUNDLE=PROJECT/'06_Models/Periderm_Pipeline_V2_20261117'

def verify_v2():
    record=json.loads((BUNDLE/'manifest.json').read_text())
    for relative,expected in record['runtime_source_sha256'].items():
        actual=hashlib.sha256((PROJECT/relative).read_bytes()).hexdigest()
        if actual!=expected:
            raise ValueError('V2 source changed since its saved checks: '+relative)
    return {'version':'V2.0','runtime_sources_match_saved_version':True}


def verify_v21():
    bundle=PROJECT/'06_Models/Periderm_Pipeline_V21_20261117'
    record=json.loads((bundle/'manifest.json').read_text())
    for relative,expected in record['runtime_source_sha256'].items():
        actual=hashlib.sha256((PROJECT/relative).read_bytes()).hexdigest()
        if actual!=expected:
            raise ValueError('V2.1 source changed since its saved checks: '+relative)
    return {'version':'V2.1','runtime_sources_match_saved_version':True}
