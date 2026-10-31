"""Refresh discovered Slurm runs independently of filesystem discovery."""
import re
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from .dashboard_snapshot import Snapshot


ACTIVE = {'running', 'queued', 'starting', 'stopping'}
STATE_MAP = {
    'RUNNING': 'running', 'COMPLETING': 'running', 'PENDING': 'queued',
    'CONFIGURING': 'queued', 'REQUEUED': 'queued', 'RESIZING': 'running',
    'COMPLETED': 'completed', 'FAILED': 'failed', 'OUT_OF_MEMORY': 'failed',
    'TIMEOUT': 'failed', 'CANCELLED': 'stopped', 'PREEMPTED': 'stopped',
    'SUSPENDED': 'stopped', 'NODE_FAIL': 'failed', 'BOOT_FAIL': 'failed',
    'DEADLINE': 'failed', 'REVOKED': 'stopped',
}


def job_statuses(host, job_ids):
    ids = {str(value) for value in job_ids}
    if not ids or any(not re.fullmatch(r'\d+(?:_\d+)?', value) for value in ids):
        raise ValueError('Invalid Slurm job identity')

    def query(args):
        result = subprocess.run(
            ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', host, shlex.join(args)],
            capture_output=True, text=True, timeout=12)
        if result.returncode:
            raise RuntimeError('无法核对 Slurm 作业状态：' + result.stderr.strip()[-200:])
        return result.stdout.splitlines()

    rows = {}
    for line in query(['squeue', '--me', '-r', '-h', '-o', '%i|%T']):
        fields = line.strip().split('|')
        if len(fields) == 2 and fields[0] in ids:
            rows[fields[0]] = {'slurm_state': fields[1]}
    missing = ids - rows.keys()
    if missing:
        for line in query(['sacct', '-X', '-n', '-P', '-S', '1970-01-01', '-j', ','.join(sorted(missing)),
                           '-o', 'JobID,State,ExitCode,End']):
            fields = line.strip().split('|')
            if len(fields) >= 4 and fields[0] in missing:
                try:
                    finished = datetime.fromisoformat(fields[3]).timestamp()
                except ValueError:
                    finished = None
                rows[fields[0]] = dict(slurm_state=fields[1].split()[0].rstrip('+'),
                                      exit_code=fields[2], finished_at=finished)
    checked = time.time()
    for job_id in ids:
        row = rows.setdefault(job_id, {'slurm_state': 'UNKNOWN'})
        row.update(job_id=job_id, state=STATE_MAP.get(row['slurm_state'], 'observed'),
                   checked_at=checked, status_reason=f"Slurm {job_id}; {row['slurm_state']}; 已实时核对")
    return rows


def job_identity(config, run):
    source = run.get('_source') or {}
    target_id = str(run.get('execution_target') or run.get('target', '')).lower()
    target_id = 'h100' if target_id == 'h100-ksy-01' else target_id
    target = config.get('rlinf_dashboard', {}).get('targets', {}).get(target_id, {})
    # Mirrors can be read on login-server while execution belongs to H200.
    host = target.get('submit_host') or source.get('ssh_host')
    job_id = run.get('job_id') or run.get('origin', {}).get('job_id')
    return (host, str(job_id)) if host and re.fullmatch(r'\d+(?:_\d+)?', str(job_id)) else None


class RunStatusCache:
    def __init__(self, config, path, runs):
        self.config, self.runs = config, runs
        self.lock, self.selected = threading.Lock(), {}
        self.snapshot = Snapshot(path, self._refresh, ttl=20)

    def _refresh(self):
        groups = {}
        for run in self.runs():
            key = job_identity(self.config, run)
            if key:
                groups.setdefault(key[0], set()).add(key[1])

        def read(item):
            host, ids = item
            try:
                return {host + ':' + job: row for job, row in job_statuses(host, ids).items()}
            except Exception as exc:
                return {host + ':' + job: dict(status_check_error=str(exc), attempted_at=time.time()) for job in ids}

        result = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            for rows in pool.map(read, groups.items()):
                result.update(rows)
        return result

    def remember(self, run, status):
        key = job_identity(self.config, run)
        if key:
            with self.lock:
                self.selected[':'.join(key)] = dict(status)

    def apply(self, run, snapshot):
        key = job_identity(self.config, run)
        if not key:
            return run
        key = ':'.join(key)
        status = (snapshot.get('data') or {}).get(key, {})
        with self.lock:
            selected = self.selected.get(key, {})
        if selected.get('checked_at', 0) > status.get('checked_at', 0):
            status = selected
        result = dict(run)
        if time.time() - status.get('checked_at', 0) <= 120:
            if status.get('slurm_state') == 'UNKNOWN' and run.get('state') not in ACTIVE:
                result.update(checked_at=status['checked_at'], status_check_error='Slurm 暂无该历史作业记录；保留已记录的结果')
            else:
                result.update(status)
        else:
            if status.get('status_check_error'):
                result['status_check_error'] = status['status_check_error']
            if run.get('state') in ACTIVE:
                result.update(state='observed', status_reason='作业状态待核对；后台刷新中' +
                              ('；' + status['status_check_error'] if status.get('status_check_error') else ''))
        return result
