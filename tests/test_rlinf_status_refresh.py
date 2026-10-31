import json
import os
import subprocess
import sys
import threading
import time
from unittest import mock

from server_4090.app import create_app
from server_4090.rlinf_dashboard import _scan_root
from server_4090.rlinf_status import RunStatusCache, job_identity, job_statuses


def test_batch_status_expands_arrays_and_uses_accounting_for_finished_jobs():
    outputs = ['7_1|RUNNING\n', '8|COMPLETED|0:0|2026-10-31T15:00:00|\n9|CANCELLED by 1000|0:15|2026-10-31T15:00:00|\n']
    with mock.patch('server_4090.rlinf_status.subprocess.run', side_effect=[
            subprocess.CompletedProcess([], 0, output, '') for output in outputs]) as run:
        rows = job_statuses('login-server', ['7_1', '8', '9'])
    assert [rows[j]['state'] for j in ('7_1', '8', '9')] == ['running', 'completed', 'stopped']
    assert '-r' in run.call_args_list[0].args[0][-1]
    assert '8,9' in run.call_args_list[1].args[0][-1]
    assert time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(rows['8']['finished_at'])) == '2026-10-31T15:00:00'


def test_status_freshness_is_independent_of_successful_directory_refresh(tmp_path):
    row = dict(id='cached', state='running', job_id='123', target='h100', _source={'ssh_host': 'login-server'})
    cache = RunStatusCache({}, tmp_path/'status.json', lambda: [row])
    assert cache.apply(row, {'data': {}})['state'] == 'observed'
    old = {'state': 'running', 'checked_at': time.time()-121}
    assert cache.apply(row, {'data': {'login-server:123': old}})['state'] == 'observed'
    status = dict(state='completed', checked_at=time.time(), slurm_state='COMPLETED')
    cache.remember(row, status)
    assert cache.apply(row, {'data': {}})['state'] == 'completed'
    assert job_identity({'rlinf_dashboard': {'targets': {'h200-ali-02': {'submit_host': 'h200-submit'}}}},
                        dict(row, execution_target='h200-ali-02')) == ('h200-submit', '123')


def test_scan_timeout_is_concise_and_does_not_expose_embedded_program():
    with mock.patch('server_4090.rlinf_dashboard.subprocess.run', side_effect=subprocess.TimeoutExpired('huge embedded program', 45)):
        try:
            _scan_root(dict(ssh_host='login-server'), '/experiments/example', 2)
        except RuntimeError as exc:
            assert '/experiments/example' in str(exc)
            assert '45' in str(exc) and len(str(exc)) < 180
        else:
            raise AssertionError('timeout was swallowed')


def test_missing_historical_directory_retains_curves_without_sync_failure():
    from server_4090.rlinf_dashboard import live_metrics_for_run
    run = dict(id='history', path='/removed/run', state='completed', _training=[{'round':1,'loss':.2}], _evaluation=[])
    with mock.patch('server_4090.rlinf_dashboard._scan_root', return_value={'exists':False}):
        result = live_metrics_for_run(run)
    assert result['run']['availability'] == 'missing'
    assert result['training_points'] == run['_training']
    assert result['run']['state'] == 'completed'


def make_app(tmp_path, overlay):
    workspace = tmp_path/'workspace'
    workspace.mkdir()
    (workspace/'rlinf-discovered-runs.json').write_text(json.dumps(overlay))
    (workspace/'rlinf-runs-index.json').write_text(json.dumps(dict(data=[], updated_at=time.time())))
    config = dict(openpi_repo=str(tmp_path), openpi_python=sys.executable,
                  dataset_root=str(tmp_path/'datasets'), workspace_root=str(workspace),
                  assets_base_dir=str(tmp_path/'assets'), checkpoint_base_dir=str(tmp_path/'checkpoints'),
                  base_checkpoint=str(tmp_path/'base'), task_monitor_interval_s=0, dashboard_profile='simulation')
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    with mock.patch.dict(os.environ, BIMANUAL_VLA_SERVER_TOKEN='x'*32):
        return create_app(path)


def test_list_and_metrics_share_selected_status_and_list_does_not_wait_for_ssh(tmp_path):
    row = dict(id='run', name='run', path='/owned/run', target='h100', state='running', job_id='123',
               _source=dict(ssh_host='login-server'), _training=[], _evaluation=[], metrics=[])
    gate = threading.Event()
    def slow(host, ids):
        assert gate.wait(5)
        return {'123': dict(job_id='123', state='completed', checked_at=time.time(), slurm_state='COMPLETED')}
    with mock.patch('server_4090.rlinf_status.job_statuses', side_effect=slow):
        app = make_app(tmp_path, {'run': row})
        client = app.test_client()
        headers = {'Authorization': 'Bearer '+'x'*32}
        try:
            start = time.monotonic()
            result = client.get('/api/rlinf/runs', headers=headers)
            assert time.monotonic()-start < 1
            assert result.json['runs'][0]['state'] == 'observed'
            gate.set()
            response = client.get('/api/rlinf/runs/run/status', headers=headers)
            assert response.status_code == 200 and response.json['run']['state'] == 'completed'
            assert client.get('/api/rlinf/runs', headers=headers).json['runs'][0]['state'] == 'completed'
            # A persisted curve mirror can still contain yesterday's run state.
            with mock.patch.object(app.extensions['live_mirror'], 'get', return_value={'run': dict(row)}):
                response = client.get('/api/rlinf/runs/run/metrics', headers=headers)
                assert response.json['run']['state'] == 'completed'
        finally:
            gate.set()
            app.extensions['live_mirror'].close()


def test_successful_empty_discovery_clears_old_source_error(tmp_path):
    error = dict(id='source-error-fast', source_id='fast', state='unavailable', error='old timeout')
    source = dict(id='fast', target='4x4090', root=str(tmp_path/'runs'))
    with mock.patch('server_4090.training_discovery.experiment_parent_sources', return_value=[source]), \
         mock.patch('server_4090.app.scan_runs', return_value=[]):
        app = make_app(tmp_path, {error['id']: error})
        client = app.test_client()
        headers = {'Authorization': 'Bearer '+'x'*32}
        try:
            deadline = time.monotonic()+3
            while time.monotonic() < deadline:
                result = client.post('/api/rlinf/runs/discover', headers=headers, json={'source':'fast'})
                if result.json['ready'] and not result.json['refreshing']:
                    break
                time.sleep(.01)
            assert result.json['ready']
            assert client.get('/api/rlinf/runs', headers=headers).json['runs'] == []
            assert json.loads((tmp_path/'workspace/rlinf-discovered-runs.json').read_text()) == {}
        finally:
            app.extensions['live_mirror'].close()
