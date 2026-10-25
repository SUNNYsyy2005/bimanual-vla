import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from flask import Flask

from bimanual_vla.data.contract import EpisodeContract
from bimanual_vla.data.lerobot import Pi0LeRobotDatasetWriter
from bimanual_vla.data.upload import source_signature
from server_4090.dataset_timing import inspect_dataset, align_dataset, atomic_json
from server_4090.dataset_timing_api import register_timing_routes


def dataset(root, *, schema='joint', alignment='next_observation', delayed_camera=True):
    contract = EpisodeContract(schema=schema, arm_side='left', fps=30,
        action_alignment=alignment, action_source='next_measured_joint_fallback' if alignment=='next_observation' else 'master_joint_mapped_absolute_target')
    writer = Pi0LeRobotDatasetWriter(root, fps=30, robot_type=contract.robot_type,
        state_names=list(contract.state_names), action_names=list(contract.action_names),
        camera_keys=list(contract.camera_keys), schema=schema, arm_mode='single', arm_side='left',
        action_source=contract.action_source, action_alignment=alignment, image_resize_mode='stretch')
    n = 18
    state = np.zeros((n, contract.state_dim), np.float32)
    state[:, 0] = np.arange(n) * .02
    state[:, -1] = np.linspace(0, 1, n)
    if schema == 'delivery':
        state[:, 3] = state[:, 7] = 1
    timestamps = 100 + np.arange(n) / 30
    actions = state[np.minimum(np.arange(n) + contract.action_offset, n-1)].copy()
    if not contract.action_offset:
        actions[:, 0] += .005
    images = {key: np.stack([np.full((224, 224, 3), 30 + index * 5 + camera * 50, np.uint8) for index in range(n)]) for camera,key in enumerate(contract.camera_keys)}
    writer.append_episode(states=state, actions=actions, timestamps=timestamps, state_timestamps=timestamps,
        action_timestamps=timestamps + contract.action_offset / 30,
        image_timestamps={key: timestamps + (2/30 if delayed_camera else 0) for key in contract.camera_keys},
        images=images, task_name='cube', instruction='pick up the cube')
    return root


def first_table(root):
    return pq.read_table(next((root/'data').glob('chunk-*/*.parquet')))


@pytest.mark.parametrize('schema', ['joint', 'delivery'])
def test_alignment_reorders_real_camera_frames_and_preserves_source(tmp_path, schema):
    source = dataset(tmp_path/'source', schema=schema)
    signature = source_signature(source)
    preview = inspect_dataset(source, {'fps':30})
    assert preview['can_align']
    assert preview['episodes'][0]['before']['cam_high']['max_ms'] == pytest.approx(1000*2/30)
    assert preview['episodes'][0]['after']['cam_high']['max_ms'] < .001
    target = tmp_path/'aligned'
    result = align_dataset(source, target, preview)
    assert result['validation'] == 'passed'
    assert source_signature(source) == signature
    mapping = np.load(target/'meta/time_alignment/episode_000000.npz')
    assert mapping['state_indices'][0] == 2
    assert mapping['image_indices_cam_high'][0] == 0
    table = first_table(target)
    original = first_table(source)
    expected_states = np.asarray(original['observation.state'].to_pylist())[mapping['state_indices']]
    np.testing.assert_allclose(table['observation.state'].to_pylist(), expected_states)
    np.testing.assert_allclose(table['action'].to_pylist(), np.asarray(original['observation.state'].to_pylist())[mapping['action_indices']])
    np.testing.assert_allclose(table['action_timestamp'].to_numpy() - table['state_timestamp'].to_numpy(), 1/30)
    import cv2
    video = cv2.VideoCapture(str(next((target/'videos').glob('chunk-*/observation.images.cam_high/*.mp4'))))
    ok, image = video.read(); video.release()
    assert ok and abs(float(image.mean()) - 30) < 6  # Source camera frame 0, not state row 2.
    assert result['output_dataset'] == 'aligned'


def test_offsets_downsampling_and_same_step_commands(tmp_path):
    source = dataset(tmp_path/'source', alignment='same_step_command')
    preview = inspect_dataset(source, {'fps':15, 'camera_offsets_ms':{'cam_high':-1000*2/30, 'cam_left_wrist':-1000*2/30}})
    assert preview['can_align']
    target = tmp_path/'aligned'
    align_dataset(source, target, preview)
    table = first_table(target)
    assert len(table) == 9
    assert json.loads((target/'meta/info.json').read_text())['fps'] == 15
    np.testing.assert_allclose(np.asarray(table['action'].to_pylist())[:,0] - np.asarray(table['observation.state'].to_pylist())[:,0], .005, atol=1e-7)
    assert np.allclose(table['action_timestamp'].to_numpy(), table['state_timestamp'].to_numpy())


def test_low_fps_and_missing_timestamp_are_blocked(tmp_path):
    source = dataset(tmp_path/'source', delayed_camera=False)
    path = next((source/'data').glob('chunk-*/*.parquet'))
    table = pq.read_table(path)
    original = table['image_timestamp.cam_high'].to_numpy().copy()
    original[1::2] = original[::2]
    table = table.set_column(table.column_names.index('image_timestamp.cam_high'), 'image_timestamp.cam_high', pa.array(original))
    pq.write_table(table, path)
    report = inspect_dataset(source, {'fps':30})
    assert not report['can_align'] and '不能生成真实' in report['errors'][0]
    table = table.drop(['state_timestamp'])
    pq.write_table(table, path)
    report = inspect_dataset(source, {})
    assert not report['can_align'] and '缺少原始时间戳' in report['errors'][0]
    assert not (tmp_path/'aligned').exists()


def test_stale_preview_and_existing_destination_never_overwrite(tmp_path):
    source = dataset(tmp_path/'source')
    preview = inspect_dataset(source, {})
    (source/'meta/tasks.jsonl').write_text((source/'meta/tasks.jsonl').read_text()+'\n')
    with pytest.raises(ValueError, match='已变化'):
        align_dataset(source, tmp_path/'target', preview)
    preview = inspect_dataset(source, {})
    target = tmp_path/'target'; target.mkdir(); (target/'keep').write_text('untouched')
    with pytest.raises(ValueError, match='不存在'):
        align_dataset(source, target, preview)
    assert (target/'keep').read_text() == 'untouched'


class FakeTasks:
    def __init__(self): self.rows = {}; self.commands = []
    def list(self): return list(self.rows.values())
    def get(self, key): return self.rows[key]
    def start(self, kind, command, *, env, metadata):
        key = f'dataset-alignment-{len(self.rows)}'
        task = {'id':key, 'type':kind, 'state':'running', 'metadata':metadata}
        self.rows[key] = task; self.commands.append((command, env))
        return task


def test_api_requires_successful_preview_and_launches_cpu_scope(tmp_path):
    source = dataset(tmp_path/'datasets/source')
    app = Flask(__name__)
    app.config['TESTING'] = True
    @app.errorhandler(ValueError)
    def invalid(error): return {'error':str(error)}, 400
    tasks = FakeTasks()
    editor = SimpleNamespace(storage_root=tmp_path/'datasets', dataset_read_roots=[tmp_path/'datasets'],
                             _dataset_path=lambda name:tmp_path/'datasets'/name)
    register_timing_routes(app, config={'workspace_root':str(tmp_path/'workspace'), 'openpi_python':'conda-python'},
                          tasks=tasks, editor=editor, environment=lambda:{}, assert_idle=lambda _:None)
    client = app.test_client()
    response = client.post('/api/datasets/source/time-alignment/inspect', json={'fps':30})
    assert response.status_code == 202
    task = response.get_json()
    command, env = tasks.commands[0]
    assert command[:4] == ['systemd-run','--user','--scope','--collect']
    assert env['CUDA_VISIBLE_DEVICES'] == ''
    blocked = client.post('/api/datasets/source/time-alignment/apply', json={'preview_task_id':task['id'], 'target_dataset_id':'out'})
    assert blocked.status_code == 400
    report = inspect_dataset(source, {'fps':30})
    atomic_json(Path(task['metadata']['result_path']), report)
    tasks.rows[task['id']]['state'] = 'completed'
    assert client.get(f"/api/datasets/source/time-alignment/jobs/{task['id']}").get_json()['report']['can_align']
    apply = client.post('/api/datasets/source/time-alignment/apply', json={'preview_task_id':task['id'], 'target_dataset_id':'out'})
    assert apply.status_code == 202
    bad = client.post('/api/datasets/source/time-alignment/apply', json={'preview_task_id':task['id'], 'target_dataset_id':'../source'})
    assert bad.status_code == 400
    reserved = client.post('/api/datasets/source/time-alignment/apply', json={'preview_task_id':task['id'], 'target_dataset_id':'out'})
    assert reserved.status_code == 400


def test_dashboard_auth_and_route_registration(tmp_path, monkeypatch):
    from server_4090.app import create_app, TaskManager
    source = dataset(tmp_path/'datasets/source')
    config = {'openpi_repo': str(Path.cwd()), 'openpi_python': 'conda-python',
              'dataset_root': str(source.parent), 'workspace_root': str(tmp_path/'workspace'),
              'assets_base_dir': str(tmp_path/'assets'), 'checkpoint_base_dir': str(tmp_path/'checkpoints'),
              'base_checkpoint': str(tmp_path/'base'), 'checkpoint_allowed_roots': [str(tmp_path/'checkpoints')],
              'eval_video_roots': [], 'task_monitor_interval_s':0}
    path = tmp_path/'config.json'; path.write_text(json.dumps(config))
    monkeypatch.setenv('BIMANUAL_VLA_SERVER_TOKEN', 'test-token-'+'x'*32)
    monkeypatch.setattr(TaskManager, 'list', lambda self: [])
    monkeypatch.setattr(TaskManager, 'start', lambda self, kind, command, **kwargs: {'id':'alignment-test', 'type':kind, 'command':command})
    app = create_app(path)
    client = app.test_client()
    endpoint = '/api/datasets/source/time-alignment/inspect'
    assert client.post(endpoint, json={}).status_code == 401
    response = client.post(endpoint, json={}, headers={'Authorization':'Bearer test-token-'+'x'*32})
    assert response.status_code == 202
    assert response.get_json()['type'] == 'dataset_alignment'


def test_alignment_keeps_split_and_episode_labels(tmp_path):
    source = dataset(tmp_path/'source', delayed_camera=False)
    info_path = source/'meta/info.json'
    info = json.loads(info_path.read_text())
    info['splits'] = {'val': '0:1'}
    atomic_json(info_path, info)
    episodes_path = source/'meta/episodes.jsonl'
    row = json.loads(episodes_path.read_text())
    row['tags'] = ['heldout']
    episodes_path.write_text(json.dumps(row)+'\n')
    preview = inspect_dataset(source, {'fps':15})
    target = tmp_path/'aligned'
    align_dataset(source, target, preview)
    assert json.loads((target/'meta/info.json').read_text())['splits'] == {'val': '0:1'}
    assert json.loads((target/'meta/episodes.jsonl').read_text())['tags'] == ['heldout']
    assert (target/'meta/time_alignment/source_episodes.jsonl').read_bytes() == episodes_path.read_bytes()


def test_delivery_alignment_keeps_same_step_gripper_command(tmp_path):
    source = dataset(tmp_path/'source', schema='delivery', delayed_camera=False)
    info_path = source/'meta/info.json'
    info = json.loads(info_path.read_text())
    info['action_source'] = 'next_measured_eef_fallback_with_same_step_commanded_gripper'
    info['action_alignment'] = 'next_observation_pose_same_step_gripper'
    atomic_json(info_path, info)
    path = next((source/'data').glob('chunk-*/*.parquet'))
    table = pq.read_table(path)
    actions = np.asarray(table['action'].to_pylist(), np.float32)
    actions[:, 9] = .25
    table = table.set_column(table.column_names.index('action'), 'action', pa.array(actions.tolist()))
    pq.write_table(table, path)
    preview = inspect_dataset(source, {'fps':15})
    assert preview['can_align']
    target = tmp_path/'aligned'
    align_dataset(source, target, preview)
    output = np.asarray(first_table(target)['action'].to_pylist())
    mapping = np.load(target/'meta/time_alignment/episode_000000.npz')
    states = np.asarray(table['observation.state'].to_pylist())
    np.testing.assert_allclose(output[:, :9], states[mapping['action_indices'], :9])
    np.testing.assert_allclose(output[:, 9], .25)
