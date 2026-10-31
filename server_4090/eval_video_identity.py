"""Video identity from verified files and provenance, independent of episode names."""
import hashlib
import json
from pathlib import Path
import posixpath
import re
import socket


def local_identities(rows, destination):
    """Background-only: hash equal-size candidates and cache by file signature."""
    cache_path = Path(destination) / 'video-fingerprints.json'
    try:
        cache = json.loads(cache_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        cache = {}
    result, sizes = [], {}
    for original in rows:
        row = dict(original)
        row.pop('content_sha256', None)
        row.pop('file_identity', None)
        path = Path(row['root']) / row['relative_path']
        try:
            stat = path.stat()
            identity = dict(host=socket.gethostname(), device=stat.st_dev, inode=stat.st_ino,
                            size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            row['file_identity'] = identity
            signature = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
            key = str(path.resolve())
            sizes.setdefault(stat.st_size, []).append((row, path, key, signature))
        except OSError:
            pass
        result.append(row)
    new_cache = {}
    for size, candidates in sizes.items():
        if not size or len(candidates) < 2:
            continue
        for row, path, key, signature in candidates:
            saved = new_cache.get(key, cache.get(key, {}))
            if saved.get('signature') != signature:
                try:
                    digest = hashlib.sha256()
                    with path.open('rb') as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            digest.update(block)
                    after = path.stat()
                    if signature != [after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns]:
                        continue
                    saved = dict(signature=signature, sha256=digest.hexdigest())
                except OSError:
                    continue
            row['content_sha256'] = saved['sha256']
            new_cache[key] = saved
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(new_cache, separators=(',', ':')), encoding='utf-8')
    temporary.replace(cache_path)
    return result


def identity_keys(row):
    keys = []
    if row.get('id'):
        keys.append(('id', row['id']))
    source = row.get('provenance') or {}
    namespace = source.get('storage_source') or (row.get('file_identity') or {}).get('host') or row.get('source')
    root = source.get('root') or row.get('root')
    relative = source.get('relative_path') or row.get('relative_path')
    if namespace and root and relative:
        keys.append(('origin', namespace, posixpath.normpath(posixpath.join(root, relative))))
    physical = row.get('file_identity') or {}
    fields = ('host', 'device', 'inode', 'size', 'mtime_ns')
    if all(physical.get(k) is not None for k in fields):
        keys.append(('file', *(physical[k] for k in fields)))
    digest = row.get('content_sha256', '')
    size = physical.get('size') or row.get('size_bytes')
    if isinstance(digest, str) and re.fullmatch('[a-f0-9]{64}', digest) and size:
        keys.append(('sha256', int(size), digest))
    return keys


def cached_identities(rows, destination):
    """Legacy readers may reuse a verified digest, but never read video bytes."""
    try:
        cache = json.loads((Path(destination) / 'video-fingerprints.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        cache = {}
    result = []
    for original in rows:
        row = dict(original)
        signature = row.pop('_file_signature', None)
        if signature:
            saved = cache.get(str((Path(row['root']) / row['relative_path']).resolve()), {})
            if saved.get('signature') == signature:
                row['content_sha256'] = saved['sha256']
        result.append(row)
    return result


def deduplicate_videos(rows):
    """Keep all experiment memberships while returning one preview per file."""
    rows = list(rows)
    parents = list(range(len(rows)))
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    owners = {}
    for index, row in enumerate(rows):
        for key in identity_keys(row):
            if key in owners:
                parents[find(index)] = find(owners[key])
            else:
                owners[key] = index
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(find(index), []).append(row)
    result = []
    for items in groups.values():
        def priority(row):
            richness = sum(bool(row.get(k)) for k in ('video_layout', 'video_views', 'has_events', 'model_name'))
            return (bool(row.get('remote')), row.get('playable') is False, -richness,
                    -float(row.get('mtime') or 0), str(row.get('id', '')))
        representative = min(items, key=priority)
        row = dict(representative)
        if len(items) > 1:
            fields = ('id', 'group_id', 'video_source', 'task', 'experiment', 'run_id', 'checkpoint_id',
                      'source', 'root', 'relative_path', 'success', 'safe_success', 'has_events')
            row['duplicate_count'] = len(items) - 1
            row['duplicate_records'] = [{k: item.get(k) for k in fields} for item in items if item is not representative]
            row['duplicate_group_ids'] = list(dict.fromkeys(item['group_id'] for item in items if item.get('group_id')))
        result.append(row)
    return result
