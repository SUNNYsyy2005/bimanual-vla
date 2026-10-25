"""Pin inference files before checkpoint GC, then stage checksummed tar parts."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
import subprocess
import shlex

PART_BYTES = 256 * 1024 * 1024


def pin_checkpoint(source, snapshot):
    source, snapshot = Path(source), Path(snapshot)
    required = ['_CHECKPOINT_METADATA', 'params/_METADATA']
    for name in required:
        if not (source / name).is_file():
            raise ValueError(f'checkpoint was removed or is not finalized: {source / name}')
    files = []
    directories=[source,*[p for p in source.rglob('*') if p.is_dir() and p.parts[len(source.parts)]!='train_state']]
    directory_times={p:p.stat().st_mtime_ns for p in directories}
    for child in source.iterdir():
        if child.name == 'train_state' or child.name.startswith('.'):
            continue
        if child.is_symlink():
            raise ValueError(f'checkpoint symlink is not supported: {child}')
        files.extend(child.rglob('*') if child.is_dir() else [child])
    rows = []
    for file in sorted(files):
        if file.is_symlink():
            raise ValueError(f'checkpoint symlink is not supported: {file}')
        if file.is_dir():
            continue
        rel = str(file.relative_to(source))
        dest = snapshot / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        # A hard link keeps the inode alive if Orbax removes the original.
        # Never fall back to a copy: GC could truncate the snapshot in flight.
        os.link(file, dest)
        rows.append({'path': rel, 'size': dest.stat().st_size})
    if not all((snapshot / name).is_file() for name in required):
        raise ValueError('checkpoint disappeared during snapshot; retry with a retained step')
    if any(p.stat().st_mtime_ns!=stamp for p,stamp in directory_times.items()):
        raise ValueError('checkpoint changed while pinning; retry with a retained step')
    return rows


class PartWriter:
    def __init__(self, root, part_bytes=PART_BYTES):
        self.root, self.part_bytes = Path(root), part_bytes
        self.parts, self.file, self.size, self.digest = [], None, 0, None

    def write(self, data):
        original = len(data)
        data = memoryview(data)
        while data:
            if self.file is None:
                self.name = f'part-{len(self.parts):05d}'
                self.file = (self.root / self.name).open('wb')
                self.digest, self.size = hashlib.sha256(), 0
            block = data[:self.part_bytes - self.size]
            self.file.write(block)
            self.digest.update(block)
            self.size += len(block)
            data = data[len(block):]
            if self.size == self.part_bytes:
                self.close_part()
        return original

    def close_part(self):
        if self.file:
            self.file.close()
            self.parts.append(dict(name=self.name, size=self.size, sha256=self.digest.hexdigest()))
            self.file = None


def export_checkpoint(source, staging, parallelism=4, staging_host=None):
    started = time.monotonic()
    source, staging = Path(source), Path(staging)
    if not staging_host:staging.parent.mkdir(parents=True, exist_ok=True)
    # Snapshots and packing use local compute storage, never login-server/NAS.
    with tempfile.TemporaryDirectory(prefix=f'.policy-export-{source.name}-', dir=source.parent) as work:
        work = Path(work)
        snapshot = work / 'snapshot'
        snapshot.mkdir()
        rows = pin_checkpoint(source, snapshot)
        marker=source.parent.parent/'.policy_action_conventions'/(source.parent.name+'.json')
        contract=json.loads(marker.read_text())
        pin_s = time.monotonic() - started
        packed = work / 'packed'
        packed.mkdir()
        writer = PartWriter(packed)
        with tarfile.open(fileobj=writer, mode='w|', format=tarfile.PAX_FORMAT, dereference=True) as archive:
            for row in rows:
                archive.add(snapshot / row['path'], arcname=row['path'], recursive=False)
        writer.close_part()
        packing_s = time.monotonic() - started - pin_s
        manifest = dict(version=1, scope='inference', full_state_available=False, action_contract=contract,
                        source=str(source), files=rows, parts=writer.parts,
                        bytes=sum(row['size'] for row in rows), pin_s=pin_s, packing_s=packing_s)
        (packed / 'manifest.json').write_text(json.dumps(manifest))
        env=os.environ.copy()
        for key in ['LD_LIBRARY_PATH','LD_PRELOAD','OPENSSL_CONF','OPENSSL_MODULES']:env.pop(key,None)
        transport=tempfile.TemporaryDirectory(prefix='checkpoint-ssh-')
        sock=str(Path(transport.name)/'master')
        ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3']
        if staging_host:
            subprocess.run([*ssh,'-M','-N','-f','-S',sock,'-o','ControlPersist=600',staging_host],check=True,env=env)
            subprocess.run([*ssh,'-S',sock,staging_host,'mkdir -p '+shlex.quote(str(staging))],check=True,env=env)
        else:staging.mkdir(exist_ok=False)
        # Publish immutable part sizes/hashes before uploading their payloads,
        # so the receiver can fetch each completed part while others stage.
        if staging_host:
            with (packed/'manifest.json').open('rb') as inp:
                subprocess.run([*ssh,'-S',sock,staging_host,'cat > '+shlex.quote(str(staging/'plan.json'))],
                               stdin=inp,check=True,env=env)
        else:shutil.copyfile(packed/'manifest.json',staging/'plan.json')
        def upload(part):
            src = packed / part['name']
            if staging_host:
                for attempt in range(3):
                    remote=str(staging/part['name']);temporary=remote+'.partial'
                    command='cat > '+shlex.quote(temporary)+' && mv '+shlex.quote(temporary)+' '+shlex.quote(remote)
                    with src.open('rb') as inp:
                        result=subprocess.run([*ssh,'-S',sock,staging_host,command],stdin=inp,env=env,capture_output=True)
                    if result.returncode==0:break
                    if attempt==2:raise RuntimeError('SSH checkpoint staging failed: '+result.stderr.decode(errors='replace')[-1000:])
                    time.sleep(2*(attempt+1))
                print(f"[dashboard] staged {part['name']} bytes={part['size']} via=ssh",flush=True)
                return
            tmp = staging / (part['name'] + '.partial')
            for attempt in range(5):
                try:
                    with src.open('rb') as inp, tmp.open('wb') as out:
                        shutil.copyfileobj(inp, out, 4 * 1024 * 1024)
                    tmp.replace(staging / part['name'])
                    break
                except OSError:
                    if attempt==4:raise
                    time.sleep(2*(attempt+1))
            print(f"[dashboard] staged {part['name']} bytes={part['size']}", flush=True)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(8, parallelism))) as pool:
                list(pool.map(upload, writer.parts))
            # Publication is last: consumers never accept a partially staged bundle.
            if staging_host:
                with (packed/'manifest.json').open('rb') as inp:
                    subprocess.run([*ssh,'-S',sock,staging_host,'cat > '+shlex.quote(str(staging/'manifest.json'))],
                                   stdin=inp,check=True,env=env)
            else:shutil.copyfile(packed / 'manifest.json', staging / 'manifest.json')
        finally:
            if staging_host:subprocess.run([*ssh,'-S',sock,'-O','exit',staging_host],capture_output=True,env=env)
            transport.cleanup()
        print('[dashboard] export receipt ' + json.dumps(dict(bytes=manifest['bytes'],
              parts=len(writer.parts), pin_s=pin_s, packing_s=packing_s,
              elapsed_s=time.monotonic()-started)), flush=True)
    return manifest
