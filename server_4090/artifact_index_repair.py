"""Coalesced index repair in a detached, low-priority process."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import shutil
import time
import uuid


def request_repair(config_path, remote=False):
    config_path = Path(config_path).resolve()
    config = json.loads(os.path.expandvars(config_path.read_text(encoding='utf-8')))
    root = Path(config['workspace_root']) / 'artifact-index-repair'
    queue = root / 'pending'
    queue.mkdir(parents=True, exist_ok=True)
    lock = (root / 'worker.lock').open('a+')
    try:
        with (root / 'queue.lock').open('a+') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            (queue / uuid.uuid4().hex).write_text(json.dumps(dict(remote=remote)))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return dict(queued=True, started=False)
            repo = Path(__file__).resolve().parents[1]
            env = dict(os.environ)
            env['PYTHONPATH'] = str(repo) + os.pathsep + env.get('PYTHONPATH', '')
            if os.environ.get('INVOCATION_ID') and shutil.which('systemd-run'):
                # A task service uses KillMode=control-group. A new session
                # alone would still be killed when that task finishes.
                unit = 'bimanual-vla-index-repair-' + uuid.uuid4().hex
                subprocess.run(['systemd-run', '--user', '--quiet', '--collect', '--no-block', '--unit=' + unit,
                    '--property=Nice=10', '--property=IOSchedulingClass=idle', '--working-directory=' + str(repo),
                    '--setenv=PYTHONPATH=' + env['PYTHONPATH'], config['openpi_python'], '-m',
                    'server_4090.artifact_index_repair', '--config', str(config_path), '--worker'],
                    check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=5)
                return dict(queued=True, started=True, unit=unit)
            with (root / 'worker.log').open('ab') as log:
                child = subprocess.Popen([config['openpi_python'], '-m', 'server_4090.artifact_index_repair',
                    '--config', str(config_path), '--lock-fd', str(lock.fileno())],
                    cwd=repo, env=env, pass_fds=(lock.fileno(),), start_new_session=True,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            return dict(queued=True, started=True, pid=child.pid)
    finally:
        # flock belongs to the inherited open file description. The child
        # retains it after the caller exits; never unlock it in the parent.
        lock.close()


def repair(config, remote=False):
    from .artifact_inventory import Inventory
    from .eval_video_catalog_runner import refresh
    from .artifact_history_runner import refresh as refresh_history
    Inventory(config).reconcile()
    path = Path(config['workspace_root']) / 'eval_video_catalog.lock'
    with path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = refresh(config, apply=True, local_only=not remote)
    if config.get('artifact_history_sources'):
        result['history'] = refresh_history(config)
    return result


def drain(config, worker_lock=None):
    from .eval_video_catalog import atomic_write
    root = Path(config['workspace_root']) / 'artifact-index-repair'
    queue = root / 'pending'
    while True:
        with (root / 'queue.lock').open('a+') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            pending = list(queue.iterdir())
            if not pending:
                if worker_lock is not None:
                    fcntl.flock(worker_lock, fcntl.LOCK_UN)
                return
        start = time.time()
        atomic_write(root / 'status.json', dict(refreshing=True, started_at=start))
        try:
            remote = False
            for path in pending:
                try: remote = remote or bool(json.loads(path.read_text()).get('remote'))
                except (OSError,ValueError): pass
            result = repair(config, remote=remote)
        except Exception as exc:
            atomic_write(root / 'status.json', dict(refreshing=False, started_at=start, error=str(exc), pending=True))
            raise
        for path in pending:
            path.unlink(missing_ok=True)
        atomic_write(root / 'status.json', dict(refreshing=False, updated_at=time.time(), result=result, error=None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--request', action='store_true')
    parser.add_argument('--lock-fd', type=int)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--remote', action='store_true')
    args = parser.parse_args()
    if args.request:
        print(json.dumps(request_repair(args.config, remote=args.remote)), flush=True)
        return
    if args.lock_fd is None and not args.worker:
        parser.error('use --request to enqueue a repair')
    os.nice(10)
    if shutil.which('ionice'):
        subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())],
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Keep the inherited descriptor open for the worker's entire lifetime.
    config = json.loads(os.path.expandvars(Path(args.config).read_text(encoding='utf-8')))
    path = Path(config['workspace_root']) / 'artifact-index-repair/worker.lock'
    with (path.open('a+') if args.worker else os.fdopen(args.lock_fd, 'a+')) as lock:
        if args.worker:
            fcntl.flock(lock, fcntl.LOCK_EX)
        drain(config, lock)


if __name__ == '__main__':
    main()
