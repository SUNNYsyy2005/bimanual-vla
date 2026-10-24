"""Authenticated Dashboard endpoints for persistent CPU alignment jobs."""
from __future__ import annotations

import json
from pathlib import Path
import re
import threading
import uuid

from flask import jsonify, request

from server_4090.dataset_timing import atomic_json, options_for

SAFE_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')


def register_timing_routes(app, *, config, tasks, editor, environment, assert_idle):
    root = Path(config['workspace_root']) / 'dataset_alignment'
    root.mkdir(parents=True, exist_ok=True)
    submit_lock = threading.Lock()

    def name(value):
        if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
            raise ValueError('无效的数据集或任务 ID')
        return value

    def source_path(dataset_id):
        source = editor._dataset_path(name(dataset_id)).resolve()
        if not (source / 'meta/info.json').is_file():
            raise FileNotFoundError(dataset_id)
        if not any(source.is_relative_to(path.resolve()) for path in editor.dataset_read_roots):
            raise ValueError('数据集路径不在允许的读取目录内')
        return source

    def task_report(dataset_id, task_id):
        task = tasks.get(name(task_id))
        metadata = task.get('metadata', {})
        if task.get('type') != 'dataset_alignment' or metadata.get('dataset_id') != dataset_id:
            raise ValueError('任务不属于此数据集的时间对齐操作')
        request_id = name(metadata['alignment_request_id'])
        result_path = root / request_id / 'result.json'
        report = json.loads(result_path.read_text()) if result_path.is_file() else None
        return task, report, result_path

    def launch(dataset_id, payload):
        request_id = uuid.uuid4().hex
        directory = root / request_id
        directory.mkdir()
        request_file = directory / 'request.json'
        repo = Path(__file__).resolve().parents[1]
        scope = f'dataset-alignment-{request_id}'
        atomic_json(request_file, {**payload, 'expected_scope': scope + '.scope'})
        command = ['systemd-run', '--user', '--scope', '--collect', f'--unit={scope}',
                   config['openpi_python'], '-m', 'server_4090.dataset_timing', '--request', str(request_file)]
        env = environment()
        env['PYTHONPATH'] = str(repo) + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        env['CUDA_VISIBLE_DEVICES'] = ''  # CPU media processing; never reserve a service GPU.
        env['OMP_NUM_THREADS'] = '2'
        return tasks.start('dataset_alignment', command, env=env,
            metadata={'dataset_id': dataset_id, 'alignment_request_id': request_id,
                      'operation': payload['operation'], 'target_dataset_id': payload.get('target_dataset_id'),
                      'scope': scope + '.scope', 'result_path': str(directory / 'result.json')})

    @app.post('/api/datasets/<dataset_id>/time-alignment/inspect')
    def inspect_timing(dataset_id):
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            raise ValueError('请求须为 JSON 对象')
        with submit_lock:
            source = source_path(dataset_id)
            assert_idle(dataset_id)
            info = json.loads((source / 'meta/info.json').read_text())
            options = options_for(info, payload)
            task = launch(dataset_id, {'operation': 'inspect', 'source': str(source), 'options': options})
        return jsonify(task), 202

    @app.get('/api/datasets/<dataset_id>/time-alignment/jobs/<task_id>')
    def timing_result(dataset_id, task_id):
        task, report, _ = task_report(name(dataset_id), task_id)
        return jsonify({'task': task, 'report': report})

    @app.post('/api/datasets/<dataset_id>/time-alignment/apply')
    def apply_timing(dataset_id):
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            raise ValueError('请求须为 JSON 对象')
        with submit_lock:
            source = source_path(dataset_id)
            assert_idle(dataset_id)
            target = name(payload.get('target_dataset_id'))
            if target == dataset_id:
                raise ValueError('目标数据集必须使用新名称')
            assert_idle(target)
            # A queued alignment also reserves its output name.
            if any(t.get('metadata', {}).get('target_dataset_id') == target
                   and t.get('state') in ('starting', 'running', 'stopping') for t in tasks.list()):
                raise ValueError('目标数据集已有正在执行的对齐任务')
            destination = editor.storage_root / target
            if destination.exists():
                raise ValueError('目标数据集已存在，不会覆盖')
            task, report, result_path = task_report(dataset_id, payload.get('preview_task_id'))
            if task.get('state') != 'completed' or not report or report.get('status') != 'inspected' or not report.get('can_align'):
                raise ValueError('请先完成可执行的时间对齐检查')
            task = launch(dataset_id, {'operation': 'apply', 'source': str(source),
                'destination': str(destination), 'target_dataset_id': target, 'preview_report': str(result_path)})
        return jsonify(task), 202
