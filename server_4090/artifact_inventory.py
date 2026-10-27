"""Local manifest inventory. Queries never crawl payload trees or load tensors.

Filesystem writes and SQLite cannot share a transaction: delete intents are
durable, scans use optimistic revisions, and tombstones reject stale discovery.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import shutil
import time
import zipfile
from contextlib import contextmanager

TERMINAL = {'completed', 'complete', 'failed', 'stopped', 'cancelled', 'canceled', 'succeeded'}
KINDS = {'video', 'trajectory', 'image', 'feature'}


def read_json(path, default=None):
    try:
        if Path(path).stat().st_size > 8 * 1024 * 1024:
            return default
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def uid(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def signature(path):
    s = Path(path).stat()
    return f'{s.st_dev}:{s.st_ino}:{s.st_size}:{s.st_mtime_ns}'


def model_signature(path):
    from .storage_worker import file_signatures, digest
    return digest(file_signatures(path)) if Path(path).is_dir() and not Path(path).is_symlink() else signature(path)


def inside(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def category(path):
    from .storage_worker import file_kind
    return file_kind(str(path))


def npz_fields(path):
    """Only ZIP directory + bounded NPY headers; never unpickle/load arrays."""
    rows = []
    try:
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist()[:512]:
                if not item.filename.endswith('.npy'): continue
                with archive.open(item) as stream:
                    if stream.read(6) != b'\x93NUMPY': continue
                    version = stream.read(2)
                    size = struct.unpack('<H' if version[0] == 1 else '<I', stream.read(2 if version[0] == 1 else 4))[0]
                    if size > 65536: continue
                    header = ast.literal_eval(stream.read(size).decode('utf-8' if version[0] == 3 else 'latin1').strip())
                    rows.append(dict(name=item.filename[:-4], shape=header.get('shape'), dtype=header.get('descr')))
    except (OSError, ValueError, SyntaxError, zipfile.BadZipFile, EOFError, struct.error):
        pass
    return rows


class Inventory:
    def __init__(self, config):
        self.config = config
        self.path = Path(config['workspace_root']) / 'artifact-inventory.sqlite3'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
              CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
              INSERT OR IGNORE INTO meta VALUES('revision','0');
              CREATE TABLE IF NOT EXISTS records(
                id TEXT PRIMARY KEY, type TEXT NOT NULL, parent TEXT, scope TEXT,
                path TEXT NOT NULL, kind TEXT, status TEXT, signature TEXT,
                revision INTEGER NOT NULL, seen REAL, data TEXT NOT NULL);
              CREATE INDEX IF NOT EXISTS records_scope ON records(type,scope,status);
              CREATE INDEX IF NOT EXISTS records_parent ON records(parent,type,status);
              CREATE INDEX IF NOT EXISTS records_path ON records(path,type);
              CREATE TABLE IF NOT EXISTS operations(
                id TEXT PRIMARY KEY, record_id TEXT, state TEXT, created REAL, error TEXT);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db: yield db
        finally: db.close()

    def revision(self, db=None):
        if db is not None: return int(db.execute("SELECT value FROM meta WHERE key='revision'").fetchone()[0])
        with self.connect() as conn: return self.revision(conn)

    def bump(self, db):
        value = self.revision(db) + 1
        db.execute("UPDATE meta SET value=? WHERE key='revision'", (str(value),))
        return value

    def upsert(self, row, observed_revision):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT * FROM records WHERE id=?', (row['id'],)).fetchone()
            if old and (old['revision'] > observed_revision or old['status'] in {'deleting', 'delete_failed'}): return False
            if old and old['status'] == 'deleted' and old['signature'] == row.get('signature'): return False
            payload = json.dumps(row['data'], ensure_ascii=False, sort_keys=True)
            # Unchanged manifests/headers need no database rewrite.
            if old and old['data'] == payload and old['signature'] == row.get('signature') and old['status'] == row['status']:
                return True
            values = (row['id'], row['type'], row.get('parent'), row.get('scope'), row['path'], row.get('kind'),
                      row['status'], row.get('signature'), self.bump(db), time.time(), payload)
            db.execute('INSERT OR REPLACE INTO records VALUES(?,?,?,?,?,?,?,?,?,?,?)', values)
            return True

    def get(self, record_id):
        with self.connect() as db:
            row = db.execute('SELECT * FROM records WHERE id=?', (record_id,)).fetchone()
        if row is None: raise ValueError('Unknown indexed artifact')
        return dict(row) | {'data': json.loads(row['data'])}

    def status(self):
        with self.connect() as db:
            meta = dict(db.execute('SELECT key,value FROM meta').fetchall())
            counts = dict(db.execute("SELECT type,count(*) FROM records WHERE status='present' GROUP BY type").fetchall())
        return dict(ready='last_scan' in meta, updated_at=float(meta.get('last_scan', 0)), counts=counts,
                    errors=json.loads(meta.get('errors', '[]')), revision=int(meta['revision']))

    def groups(self):
        with self.connect() as db:
            rows = db.execute("SELECT scope,count(*) n FROM records WHERE type='episode' AND status='present' GROUP BY scope").fetchall()
            result = []
            for row in rows:
                example = json.loads(db.execute("SELECT data FROM records WHERE type='episode' AND scope=? AND status='present' LIMIT 1", (row['scope'],)).fetchone()[0])
                result.append(dict(id=row['scope'], count=row['n'], **{k: example.get(k, 'unknown') for k in ('video_source', 'task', 'experiment')}))
        return dict(groups=result, **self.status())

    def page(self, scopes, offset=0, limit=25, phase='all', kind='all'):
        if not isinstance(scopes, list) or len(scopes) > 256 or not 0 <= offset <= 10000000 or not 1 <= limit <= 100: raise ValueError('Invalid artifact page')
        if not scopes: return dict(episodes=[], total=0, selection_required=True)
        where = "e.type='episode' AND e.status='present' AND e.scope IN (" + ','.join('?' for _ in scopes) + ')'
        args = list(scopes)
        if phase != 'all':
            where += " AND json_extract(e.data,'$.phase')=?"; args.append(phase)
        if kind in KINDS:
            where += " AND EXISTS(SELECT 1 FROM records f WHERE f.parent=e.id AND f.kind=? AND f.status='present')"; args.append(kind)
        with self.connect() as db:
            total = db.execute('SELECT count(*) FROM records e WHERE ' + where, args).fetchone()[0]
            rows = db.execute('SELECT e.id,e.data FROM records e WHERE ' + where + ' ORDER BY e.path LIMIT ? OFFSET ?', args + [limit, offset]).fetchall()
            episodes = []
            for row in rows:
                files = [dict(json.loads(f['data']), id=f['id'], status=f['status']) for f in db.execute(
                    "SELECT * FROM records WHERE parent=? AND type='file' ORDER BY kind,path", (row['id'],))]
                episodes.append(dict(json.loads(row['data']), id=row['id'], files=files))
        return dict(episodes=episodes, total=total, offset=offset, limit=limit, **self.status())

    def models(self, backend=None):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM records WHERE type='model' AND status='present'").fetchall()
        return [dict(json.loads(r['data']), id=r['id']) for r in rows if not backend or json.loads(r['data']).get('backend') == backend]

    def invalidate_missing(self, path):
        """Compatibility mutations also invalidate registered paths immediately."""
        path = Path(path).resolve()
        if path.exists(): return
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute("UPDATE records SET status='deleted',revision=? WHERE path=?", (self.bump(db), str(path)))

    def roots(self):
        raw = [self.config.get('experiment_artifact_root') or str(Path(self.config['workspace_root']) / 'experiment_artifacts')]
        raw += self.config.get('checkpoint_allowed_roots', [])
        raw += [value.get('artifact_root') or str(Path(value['dataset_root']).parent/'simulation-artifacts')
                for value in self.config.get('local_storage_locations',{}).values() if value.get('dataset_root')]
        return list(dict.fromkeys(Path(os.path.expandvars(p)).expanduser().resolve() for p in raw))

    def reconcile(self):
        from .eval_video_browser import group_id
        started_revision = self.revision()
        manifests = set()
        errors = []
        for root in self.roots():
            manifests.update(p.resolve() for p in root.glob('*/*/*/*/run.json'))
        for manifest in sorted(manifests):
            try:
                run = read_json(manifest, {})
                if run.get('schema') != 'experiment_run.v1': continue
                root = manifest.parent
                if not any(inside(root, allowed) for allowed in self.roots()): continue
                run_id = uid(root); scope = group_id(run)
                run = dict(run, path=str(root), inventory_id=run_id)
                self.upsert(dict(id=run_id, type='run', scope=scope, path=str(root), status='present', signature=signature(manifest), data=run), started_revision)
                for phase in ('train', 'eval'):
                    for ep in (root / phase).glob('*/*/*/episode.json'):
                        if not inside(ep, root / phase) or ep.is_symlink(): continue
                        self.scan_episode(ep, run, run_id, scope, phase, started_revision)
                self.scan_models(root, run, run_id, scope, started_revision)
            except (OSError, ValueError, sqlite3.Error) as exc:
                errors.append(f'{manifest}: {exc}')
        # Recheck disappearance without bulk invalidation after partial scans.
        with self.connect() as db:
            candidates = db.execute("SELECT id,path,revision,status FROM records WHERE status IN ('present','deleting','delete_failed')").fetchall()
        for row in candidates:
            if Path(row['path']).exists(): continue
            with self.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                current = db.execute('SELECT revision,status FROM records WHERE id=?', (row['id'],)).fetchone()
                if current['revision'] != row['revision']: continue
                state = 'deleted' if current['status'] in {'deleting', 'delete_failed'} else 'missing'
                db.execute('UPDATE records SET status=?,revision=? WHERE id=?', (state, self.bump(db), row['id']))
                if state == 'deleted': db.execute("UPDATE operations SET state='completed' WHERE record_id=?", (row['id'],))
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO meta VALUES('last_scan',?)", (str(time.time()),))
            db.execute("INSERT OR REPLACE INTO meta VALUES('errors',?)", (json.dumps(errors[:50]),))
        return self.status()

    def scan_episode(self, manifest, run, run_id, scope, phase, revision):
        ep = read_json(manifest, {})
        if ep.get('schema') != 'rollout_episode.v1': return
        root = manifest.parent; episode_id = uid(root)
        keys = ('episode_id', 'split', 'success', 'safe_success', 'checkpoint_id', 'counted_in_success_rate', 'trajectory_reference', 'status')
        data = {k: ep.get(k) for k in keys}
        data.update({k: run.get(k, 'unknown') for k in ('video_source', 'task', 'experiment', 'run_id')})
        data.update(path=str(root), phase=phase, run_inventory_id=run_id, run_state=run.get('state'), scope=run.get('scope', run.get('origin', {}).get('scope', 'unknown')))
        self.upsert(dict(id=episode_id, type='episode', parent=run_id, scope=scope, path=str(root), status='present', signature=signature(manifest), data=data), revision)
        for current, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(current)/d).is_symlink() and d not in {'checkpoints', 'logs', 'metadata', 'artifacts'}]
            for name in names:
                path = Path(current) / name
                if not path.is_file() or not inside(path, root): continue
                kind = category(path)
                if kind not in KINDS: continue
                sig = signature(path); ident = uid(path)
                try: old = self.get(ident)
                except ValueError: old = None
                fields = old['data'].get('fields', []) if old and old['signature'] == sig else npz_fields(path) if path.suffix == '.npz' else []
                filedata = dict(path=str(path), name=str(path.relative_to(root)), kind=kind, size_bytes=path.stat().st_size,
                                fields=fields, contains_images=any(len(f.get('shape') or []) >= 3 and any(x in f['name'].lower() for x in ('image','rgb','camera','depth')) for f in fields),
                                deletable=not path.is_symlink(), run_inventory_id=run_id, episode_inventory_id=episode_id)
                self.upsert(dict(id=ident, type='file', parent=episode_id, scope=scope, path=str(path), kind=kind,
                                 status='present', signature=sig, data=filedata), revision)

    def scan_models(self, root, run, run_id, scope, revision):
        index = read_json(root / 'checkpoints/index.json', {})
        if index.get('schema') != 'checkpoint_index.v1': return
        for entry in index.get('checkpoints', []):
            relative = Path(str(entry.get('path', '')))
            if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0] != 'checkpoints': continue
            path = root / relative
            if path == root / 'checkpoints' or not inside(path, root / 'checkpoints') or not path.exists(): continue
            data = dict(entry, path=str(path), run_path=str(root), run_inventory_id=run_id,
                        backend=run.get('backend'), experiment=run.get('experiment'), run_id=run.get('run_id'), origin=run.get('origin', {}),
                        source='manifest_index', deletable=not path.is_symlink())
            self.upsert(dict(id=uid(path), type='model', parent=run_id, scope=scope, path=str(path), kind='model', status='present', signature=model_signature(path), data=data), revision)

    def delete_files(self, ids, active_check, *, models=False):
        if not isinstance(ids, list) or not ids or len(ids) > 200 or any(not isinstance(i, str) for i in ids): raise ValueError('Select 1–200 files')
        selected = [self.get(i) for i in dict.fromkeys(ids)]
        # Validate the whole request before any mutation. Never delete a run/episode tree.
        for row in selected:
            if row['status'] != 'present' or (row['type'] != 'model' if models else row['type'] != 'file' or row['kind'] not in KINDS): raise ValueError('Artifact belongs to a different management area')
            path = Path(row['path']); parent = self.get(row['parent']); run = parent if models else self.get(parent['parent'])
            root = Path(run['path']); relative = path.relative_to(root)
            if models:
                if len(relative.parts) < 2 or relative.parts[0] != 'checkpoints' or not inside(path, root/'checkpoints'): raise ValueError('Not a canonical checkpoint')
                for current, dirs, names in os.walk(path, followlinks=False):
                    if any((Path(current)/n).is_symlink() for n in dirs + names) or set(dirs + names) & {'train','eval','run.json','episode.json','logs'}: raise ValueError('Checkpoint contains links or mixed rollout/log payloads')
            elif len(relative.parts) < 5 or relative.parts[0] not in {'train','eval'} or not inside(path, root / relative.parts[0]): raise ValueError('Not a canonical rollout payload')
            if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root and root in p.parents): raise ValueError('Linked payloads are read-only')
            for owned in [path, *path.rglob('*')] if path.is_dir() else [path]:
                if hasattr(os,'getuid') and owned.lstat().st_uid != os.getuid(): raise ValueError('Artifact contains another owner')
            live = read_json(root/'run.json', {})
            if live.get('state', '').lower() not in TERMINAL: raise ValueError('Only finished runs can be cleaned')
            active_check(path if models else root)
            if not any(inside(root, allowed) for allowed in self.roots()) or (model_signature(path) if models else signature(path)) != row['signature']: raise ValueError('Artifact changed or moved; refresh before deleting')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for row in selected:
                current = db.execute('SELECT revision,status FROM records WHERE id=?', (row['id'],)).fetchone()
                if current['revision'] != row['revision'] or current['status'] != 'present': raise ValueError('Concurrent artifact update; refresh')
            for row in selected:
                db.execute("UPDATE records SET status='deleting',revision=? WHERE id=?", (self.bump(db), row['id']))
                db.execute('INSERT INTO operations VALUES(?,?,?,?,?)', (uid(f"{time.time_ns()}:{row['id']}"), row['id'], 'pending', time.time(), None))
        result = []
        for row in selected:
            error = None
            try:
                path = Path(row['path'])
                if (model_signature(path) if models else signature(path)) != row['signature']: raise ValueError('File changed after deletion was prepared')
                if models and path.is_dir(): shutil.rmtree(path)
                else: path.unlink()
            except (OSError, ValueError) as exc: error = str(exc)
            with self.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                db.execute('UPDATE records SET status=?,revision=? WHERE id=?', ('delete_failed' if error else 'deleted', self.bump(db), row['id']))
                db.execute('UPDATE operations SET state=?,error=? WHERE record_id=? AND state=?', ('failed' if error else 'completed', error, row['id'], 'pending'))
            result.append(dict(id=row['id'], deleted=error is None, error=error))
        return dict(results=result, deleted_count=sum(r['deleted'] for r in result))


def main():
    import fcntl
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True); args = parser.parse_args()
    config = json.loads(os.path.expandvars(Path(args.config).read_text()))
    inventory = Inventory(config)
    with inventory.path.with_suffix('.lock').open('w') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return
        print(json.dumps(inventory.reconcile(), ensure_ascii=False))


if __name__ == '__main__': main()
