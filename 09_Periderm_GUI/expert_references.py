"""Display expert measurements separately; never replace an automatic result."""
from pathlib import Path
import copy,csv,hashlib,io,json,math

class ExpertReferenceError(ValueError):
    """Reference needs review; never silently replace a conflicting record."""

def track_signature(row):
    # Bind to a protected pipeline version, source image, and identical detection.
    fields={k:row.get(k) for k in ('track_id','detection_bbox_xyxy','transition_xy','endpoint_xy')}
    return hashlib.sha256(json.dumps(fields,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def attach_references(job, directory):
    out=copy.deepcopy(job)
    summary=out.get('summary') or {}
    rows=summary.get('rows',[])
    for row in rows:
        row.pop('expert_reference',None)
    for file in sorted(Path(directory).glob('*.json')):
        try:
            ref=json.loads(file.read_text())
            if not isinstance(ref,dict):raise ValueError('Expected a reference object')
            for key in ('id','source_sha256','pipeline_version','track_signature','finding'):
                if not isinstance(ref.get(key),str) or not ref[key].strip():raise ValueError('Missing or invalid '+key)
            if type(ref.get('track_id')) is not int or ref['track_id']<1:raise ValueError('Invalid track_id')
            value=ref.get('periderm_length_um')
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('Invalid expert length')
        except (OSError,ValueError) as error:
            raise ExpertReferenceError('Expert reference unreadable: '+file.name) from error
        if ref.get('source_sha256')!=summary.get('source_sha256') or ref.get('pipeline_version')!=summary.get('pipeline_version'):
            continue
        for row in rows:
            if row.get('track_id')==ref.get('track_id') and track_signature(row)==ref.get('track_signature'):
                if row.get('expert_reference') is not None:
                    raise ExpertReferenceError('Expert reference conflict: '+file.name+'. Review the local reference records.')
                row['expert_reference']={k:ref.get(k) for k in ('id','periderm_length_um','finding','expert','expert_words','reviewed_at_utc')}
    summary['expert_reference_count']=sum(r.get('expert_reference') is not None for r in rows)
    if 'summary' in out and out['summary'] is not None:out['summary']=summary
    return out

def measurements_download(text,job,withdrawn=False):
    reader=csv.DictReader(io.StringIO(text));fields=list(reader.fieldnames or [])
    references={str(row['track_id']):row['expert_reference'] for row in (job.get('summary') or {}).get('rows',[]) if row.get('expert_reference') is not None}
    if not references and not withdrawn:return text
    if references:fields+=['expert_periderm_length_um','expert_reference_status','expert_reference_id','expert_reference_author']
    if withdrawn:fields+=['record_status','review_warning']
    output=io.StringIO();writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader()
    for row in reader:
        ref=references.get(row.get('track_id',''))
        if references:
            row.update(expert_periderm_length_um='' if ref is None else ref['periderm_length_um'],
                       expert_reference_status='' if ref is None else ref['finding'].replace('confirmed by you','expert reference'),
                       expert_reference_id='' if ref is None else ref['id'],
                       expert_reference_author='' if ref is None else ref.get('expert',''))
        if withdrawn:row.update(record_status='withdrawn',review_warning='V2.1 endpoint estimates are unreliable; archived experiment only, not research measurements.')
        writer.writerow(row)
    return output.getvalue()
