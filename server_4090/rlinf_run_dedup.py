"""Deduplicate dashboard records by storage provenance, not names or job IDs."""
from pathlib import PurePosixPath


def node_id(value):
    value = str(value or '').lower()
    return {'h100-ksy-01': 'h100', '4x4090': 'local_4090'}.get(value, value)


def identity_keys(run):
    node = node_id(run.get('execution_target') or run.get('target'))
    keys = {('execution:' + node, str(PurePosixPath(p))) for p in
            [run.get('path'), run.get('_physical_path'), *(run.get('_aliases') or [])] if p}
    source = run.get('_source') or {}
    physical = run.get('_physical_path')
    # A Slurm storage source executes on a node-local filesystem, even if its
    # submit SSH host is login-server. Never merge those hosts by path alone.
    if physical and source.get('ssh_host') and not source.get('storage_location'):
        keys.add(('storage:' + source['ssh_host'], str(PurePosixPath(physical))))
    keys.update(map(tuple, run.get('_dedup_identity_keys', [])))
    return keys


def deduplicate_runs(candidates):
    """Return records and an old-ID map; retain all source records on disk."""
    rows = sorted(candidates, key=lambda r: (
        not bool(r.get('_canonical')),
        not bool((r.get('_source') or {}).get('status_from_process')),
        str(r.get('source_id', '')).startswith('live-'),
        -float(r.get('synced_at') or r.get('updated_at') or 0)))
    kept, aliases, owners = [], {}, {}
    for original in rows:
        row = dict(original)
        keys = identity_keys(row)
        matches = [owners[k] for k in keys if k in owners]
        # Canonical metadata and native stage directories belong to their run.
        # A nested canonical run is independent and must never be swallowed.
        if not matches and not row.get('_canonical'):
            for parent in kept:
                if not parent.get('_canonical'):
                    continue
                parent_keys = identity_keys(parent)
                for host, path in keys:
                    if any(host == ph and (
                        path == pp + '/metadata' or path == pp + '/metrics'
                        or path.startswith(pp + '/artifacts/native/'))
                        for ph, pp in parent_keys):
                        matches = [parent]
                        break
                if matches:
                    break
        if matches:
            parent = matches[0]
            aliases[row['id']] = parent['id']
            parent['_dedup_identity_keys'] = list(set(map(tuple, parent.get('_dedup_identity_keys', []))) | keys)
            for key in keys:
                owners[key] = parent
            continue
        kept.append(row)
        for key in keys:
            owners[key] = row
    return kept, aliases
