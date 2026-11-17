"""Report mask-based periderm estimates without requiring an accepted T or J.

This is a reporting/measurement mode, not a new trained model. The numerical
quantity is cleaned P-mask arc length on the existing detected root route.
Explicit, image-and-geometry-bound expert corrections take precedence in the
reported column; the automatic estimate is always retained separately.
"""
import copy
import csv
import io
import math
import datetime

VERSION = 'per_root_mask_estimates_1.0'
SCOPE = ('Estimated length of cleaned Periderm-labelled portions of the detected '
         'root path. Path extent is automatic, without requiring a confirmed '
         'junction. May include hypocotyl or omit tissue; not validated T-to-J length.')
FIELDS = ['root', 'periderm_length_um', 'value_source', 'automatic_mask_estimate_um',
          'status', 'review_note', 'expert_reference_id', 'source_sha256',
          'pixels_per_micrometre', 'report_version', 'diagnostic_version',
          'raw_mask_sha256', 'snapshot_generated_at_utc', 'measurement_definition']


def attach_report(job):
    out = copy.deepcopy(job)
    summary = out.get('summary') or {}
    summary.pop('per_root_report', None)
    summary.pop('per_root_report_error', None)
    diagnostics = summary.get('model_diagnostics')
    if not diagnostics or summary.get('diagnostic_error'):
        return out
    if (not isinstance(summary.get('source_sha256'),str) or not summary['source_sha256']
            or diagnostics.get('source_sha256') != summary['source_sha256']):
        raise ValueError('Per-root estimate source does not match this image.')
    scale=summary.get('pixels_per_micrometre')
    if (type(scale) not in (int,float) or not math.isfinite(scale) or scale<=0
            or diagnostics.get('pixels_per_micrometre') != scale):
        raise ValueError('Per-root estimate scale does not match this analysis.')
    diags = {}
    for row in diagnostics['rows']:
        ident = row['track_id']
        if ident in diags:
            raise ValueError('Duplicate root in mask estimates.')
        diags[ident] = row
    records = []
    generated=datetime.datetime.now(datetime.timezone.utc).isoformat()
    for original in summary['rows']:
        diag = diags.get(original['track_id'], {})
        automatic = diag.get('PAT_cleanup_P_labelled_path_um')
        if automatic is not None:
            if (type(automatic) not in (int, float) or not math.isfinite(automatic)
                    or automatic < 0 or not diag.get('diagnostic_status', '').startswith('computed')):
                raise ValueError('Invalid mask estimate for root '+str(original['track_id']))
        ref = original.get('expert_reference')
        if ref is not None:
            value = ref['periderm_length_um']
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:
                raise ValueError('Invalid expert correction.')
            origin = 'Expert correction'
            status = 'expert_corrected'
            note = ref['finding']
        elif automatic is not None:
            value = automatic
            origin = 'PAT mask estimate'
            status = 'estimated'
            if value == 0:
                note = 'No Periderm retained on the sampled path; automatic zero is not confirmed absence.'
            elif diag.get('path_end_type') == 'visible_right_end_only':
                note = 'Unconfirmed root end; estimate may include hypocotyl.'
            else:
                note = 'Estimated from the mask and automatic root path; check against the image.'
        else:
            value = None
            origin = 'No calculable estimate'
            status = 'unmeasurable'
            note = diag.get('diagnostic_reason') or '; '.join(original.get('notes', []))
        records.append(dict(root=original['track_id'], periderm_length_um=value,
                            value_source=origin, automatic_mask_estimate_um=automatic,
                            status=status, review_note=note,
                            expert_reference_id=ref['id'] if ref else '',
                            source_sha256=summary['source_sha256'],
                            pixels_per_micrometre=summary['pixels_per_micrometre'],
                            report_version=VERSION, diagnostic_version=diagnostics['version'],
                            raw_mask_sha256=diagnostics['raw_mask_sha256'],
                            snapshot_generated_at_utc=generated, measurement_definition=SCOPE))
    summary['per_root_report'] = dict(version=VERSION, rows=records,
                                    numeric_count=sum(r['periderm_length_um'] is not None for r in records),
                                    measurement_definition=SCOPE, training_occurred=False)
    return out


def report_csv(job):
    report = (job.get('summary') or {}).get('per_root_report')
    if report is None:
        raise ValueError('Mask estimates are not available for this analysis.')
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(report['rows'])
    return out.getvalue()
