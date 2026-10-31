"""Resolve video experiment ownership from declared training-run references."""
import posixpath


def _path(value):
    return posixpath.normpath(str(value).replace('\\', '/')) if value else None


def _node(record):
    value = str(record.get('execution_target') or record.get('execution_node') or '').lower()
    return {'h100-ksy-01': 'h100', '4x4090': 'local_4090'}.get(value, value)


def run_key(record):
    return tuple(str(record.get(k) or '') for k in ('video_source', 'task', 'experiment', 'run_id'))


def training_lineage(records):
    """Missing, conflicting or cross-task references leave the original group."""
    records = [r for r in records if isinstance(r, dict) and r.get('schema') == 'experiment_run.v1']
    parents = {}
    for record in records:
        if record.get('kind') == 'eval' or record.get('training_run_ref'):
            continue
        origin = record.get('origin') or {}
        for value in (record.get('path'), origin.get('legacy_path'), origin.get('physical_legacy_path')):
            if value:
                parents.setdefault(_path(value), []).append(record)
    result = {}
    conflicts = set()
    for record in records:
        if record.get('kind') != 'eval':
            continue
        reference = record.get('training_run_ref') or {}
        if not isinstance(reference, dict):
            continue
        candidates = []
        for value in (reference.get('canonical_path'), reference.get('path')):
            candidates.extend(parents.get(_path(value), []))
        candidates = [parent for parent in candidates
                      if all(record.get(k) and record.get(k) == parent.get(k) for k in ('video_source', 'task'))
                      and not (_node(record) and _node(parent) and _node(record) != _node(parent))]
        identities = {(parent.get('experiment'), parent.get('run_id'), _node(parent)) for parent in candidates}
        if len(identities) != 1 or not all(next(iter(identities))[:2]):
            continue
        parent = candidates[0]
        key = run_key(record)
        relation = dict(experiment=parent['experiment'], training_run_id=parent['run_id'],
                        training_run_ref=reference, lineage_evidence='training_run_ref')
        if key in result and result[key] != relation:
            conflicts.add(key)
        else:
            result[key] = relation
    return {key: value for key, value in result.items() if key not in conflicts}


def apply_lineage(rows, lineage):
    labels = {(row.get('video_source'), row.get('task'), row.get('experiment')):
              row.get('experiment_label') or row.get('experiment') for row in rows}
    result = []
    for original in rows:
        row = dict(original)
        relation = lineage.get(run_key(row))
        if relation:
            row.update(artifact_experiment=row['experiment'],
                       artifact_experiment_label=row.get('experiment_label') or row['experiment'],
                       artifact_role='independent_eval', **relation)
            row['experiment_label'] = labels.get((row.get('video_source'), row.get('task'), row['experiment']), row['experiment'])
        result.append(row)
    return result
