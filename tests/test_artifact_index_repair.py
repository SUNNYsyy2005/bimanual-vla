import fcntl
import json
import sys
import os
import time
from unittest import mock

import pytest
from server_4090 import artifact_index_repair as repair


def config_file(tmp_path):
    path=tmp_path/'config.json'
    path.write_text(json.dumps(dict(workspace_root=str(tmp_path),openpi_python=sys.executable)))
    return path


def test_busy_worker_merges_requests_without_starting_another_process(tmp_path):
    config=config_file(tmp_path)
    root=tmp_path/'artifact-index-repair'
    root.mkdir()
    with (root/'worker.lock').open('a+') as lock, mock.patch.object(repair.subprocess,'Popen') as spawn:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for _ in range(5):
            assert repair.request_repair(config)==dict(queued=True,started=False)
        spawn.assert_not_called()
    assert len(list((root/'pending').iterdir()))==5


def test_new_worker_inherits_lock_and_runs_outside_the_calling_process(tmp_path):
    with mock.patch.object(repair.subprocess,'Popen') as spawn:
        spawn.return_value.pid=123
        result=repair.request_repair(config_file(tmp_path))
    assert result['started'] and result['pid']==123
    args,kwargs=spawn.call_args
    assert args[0][0]==sys.executable and '-m' in args[0]
    assert kwargs['start_new_session'] and kwargs['pass_fds']
    assert kwargs['cwd'].name=='validation' or (kwargs['cwd']/'server_4090').is_dir()


def test_systemd_task_repairs_use_an_independent_service_cgroup(tmp_path):
    with mock.patch.dict(os.environ,INVOCATION_ID='task-service'), \
         mock.patch.object(repair.shutil,'which',return_value='/usr/bin/systemd-run'), \
         mock.patch.object(repair.subprocess,'run') as run, \
         mock.patch.object(repair.subprocess,'Popen') as spawn:
        result=repair.request_repair(config_file(tmp_path))
    assert result['unit'].startswith('bimanual-vla-index-repair-')
    args=run.call_args.args[0]
    assert '--worker' in args and '--property=IOSchedulingClass=idle' in args
    assert '--no-block' in args
    spawn.assert_not_called()


def test_requests_arriving_during_refresh_are_repaired_in_a_followup_pass(tmp_path):
    queue=tmp_path/'artifact-index-repair/pending'
    queue.mkdir(parents=True)
    for index in range(5):(queue/str(index)).touch()
    def refresh(config, **kwargs):
        if refresh_mock.call_count==1:(queue/'arrived-later').touch()
        return dict(local_videos=2)
    with mock.patch.object(repair,'repair',side_effect=refresh) as refresh_mock:
        repair.drain(dict(workspace_root=str(tmp_path)))
    assert refresh_mock.call_count==2 and not list(queue.iterdir())
    assert json.loads((queue.parent/'status.json').read_text())['error'] is None


def test_remote_directory_refresh_requests_are_preserved_when_coalesced(tmp_path):
    queue=tmp_path/'artifact-index-repair/pending'
    queue.mkdir(parents=True)
    (queue/'local').write_text(json.dumps(dict(remote=False)))
    (queue/'remote').write_text(json.dumps(dict(remote=True)))
    with mock.patch.object(repair,'repair',return_value={}) as refresh:
        repair.drain(dict(workspace_root=str(tmp_path)))
    refresh.assert_called_once_with(dict(workspace_root=str(tmp_path)),remote=True)


def test_failed_repair_retains_pending_requests_and_last_published_index(tmp_path):
    queue=tmp_path/'artifact-index-repair/pending'
    queue.mkdir(parents=True);(queue/'pending').touch()
    index=tmp_path/'browse-index.json';index.write_text('old-valid-index')
    with mock.patch.object(repair,'repair',side_effect=RuntimeError('scan failed')):
        with pytest.raises(RuntimeError):repair.drain(dict(workspace_root=str(tmp_path)))
    assert (queue/'pending').exists() and index.read_text()=='old-valid-index'
    assert json.loads((queue.parent/'status.json').read_text())['error']=='scan failed'


def test_detached_worker_indexes_new_canonical_video_and_removes_missing_entry(tmp_path):
    from server_4090.eval_video_browser import VideoBrowser
    artifacts=tmp_path/'artifacts'
    run=artifacts/'robotwin/task/experiment/run'
    video=run/'eval/round-1/success/episode/review.mp4'
    video.parent.mkdir(parents=True)
    video.write_bytes(b'video-content')
    os.utime(video,(time.time()-10,time.time()-10))
    (run/'run.json').write_text(json.dumps(dict(schema='experiment_run.v1',video_source='robotwin',task='task',
        experiment='experiment',run_id='run',backend='rlinf',kind='eval',state='completed')))
    (video.parent/'episode.json').write_text(json.dumps(dict(schema='rollout_episode.v1',episode_id='episode',status='completed')))
    config=dict(workspace_root=str(tmp_path/'workspace'),openpi_python=sys.executable,experiment_artifact_root=str(artifacts))
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    def run_worker():
        started=time.monotonic()
        result=repair.request_repair(path)
        assert time.monotonic()-started<1
        deadline=time.monotonic()+10
        status=tmp_path/'workspace/artifact-index-repair/status.json'
        while time.monotonic()<deadline:
            if status.exists():
                data=json.loads(status.read_text())
                assert not data.get('error'),data
                pending=tmp_path/'workspace/artifact-index-repair/pending'
                if not data.get('refreshing') and not list(pending.iterdir()):break
            time.sleep(.02)
        else:raise AssertionError('worker did not finish')
        return result
    run_worker()
    browser=VideoBrowser(tmp_path/'workspace/eval_video_catalog')
    assert browser.groups()['total']==1 and video.exists()
    video.unlink()
    run_worker()
    assert browser.groups()['total']==0
