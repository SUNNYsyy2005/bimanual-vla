"""Inspect/resample recorded LeRobot timestamps into a separate dataset.

No hardware synchronization is inferred. Nearest measured values are retained;
images are never duplicated to claim a higher capture rate. Run via the CPU
worker, not in a Flask request or a GPU process.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import signal
from pathlib import Path
import shutil
import uuid

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from bimanual_vla.data.check import check_dataset
from bimanual_vla.data.contract import EpisodeContract
from bimanual_vla.data.lerobot import Pi0LeRobotDatasetWriter
from bimanual_vla.data.upload import source_signature


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def options_for(info, options):
    fps = float(options.get('fps') or info['fps'])
    tolerance = float(options.get('max_skew_ms', 50)) / 1000
    if not math.isfinite(fps) or fps not in (15, 20, 25, 30):
        raise ValueError('目标频率须为 15、20、25 或 30 Hz')
    if not math.isfinite(tolerance) or not 0 < tolerance <= .050:
        raise ValueError('最大时间偏差须大于 0 且不超过 50 ms')
    offsets = options.get('camera_offsets_ms', {})
    if not isinstance(offsets, dict):
        raise ValueError('相机时间修正必须按相机名称指定')
    keys = camera_keys(info)
    if set(offsets) - set(keys):
        raise ValueError('时间修正含未知相机')
    offsets = {key: float(offsets.get(key, 0)) for key in keys}
    if any(not math.isfinite(value) or abs(value) > 1000 for value in offsets.values()):
        raise ValueError('相机时间修正须在 ±1000 ms 内')
    return {'fps': int(fps), 'max_skew_ms': tolerance * 1000, 'camera_offsets_ms': offsets}


def camera_keys(info):
    keys = list(info.get('camera_keys', []))
    if not keys:
        keys = [key.removeprefix('observation.images.') for key, value in info['features'].items()
                if value.get('dtype') in ('image', 'video') and key.startswith('observation.images.')]
    if not keys:
        raise ValueError('未找到具名相机流')
    return keys


def contract_for(info):
    if info.get('contract_version') != 3 or info.get('legacy'):
        raise ValueError('目前支持 v3 joint / delivery 绝对动作；旧增量数据须先转换契约')
    contract = EpisodeContract(schema=info['schema'], arm_mode=info['arm_mode'],
        arm_side=info['arm_side'], camera_keys=tuple(camera_keys(info)),
        fps=int(info['fps']), action_source=info['action_source'],
        action_alignment=info['action_alignment'], action_offset=int(info['action_offset']),
        action_horizon=int(info.get('action_horizon', 50)))
    if info['state_dim'] != contract.state_dim or info['raw_action_dim'] != contract.raw_action_dim:
        raise ValueError('状态/动作维度不符合已声明契约')
    if contract.action_offset == 1 and 'next_measured' not in contract.action_source:
        raise ValueError('无法从未知动作来源重建下一观测目标')
    if info.get('gripper_semantics') != contract.gripper_semantics:
        raise ValueError('夹爪单位/语义不符合 v3 开度比例契约')
    return contract


def series(table, name, *, repeats=False):
    if name not in table.column_names:
        raise ValueError(f'缺少原始时间戳 {name}；不能用帧编号伪造实测同步')
    values = np.asarray(table[name].to_pylist(), dtype=np.float64).reshape(-1)
    if len(values) != table.num_rows or not np.isfinite(values).all():
        raise ValueError(f'{name} 长度或有限值检查失败')
    differences = np.diff(values)
    if np.any(differences < 0 if repeats else differences <= 0):
        raise ValueError(f'{name} 时间倒退或重复')
    return values


def nearest(source, target):
    right = np.searchsorted(source, target).clip(0, len(source) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(np.abs(source[left] - target) <= np.abs(source[right] - target), left, right)


def metrics(values):
    signed_median = float(np.median(values) * 1000)
    values = np.abs(values) * 1000
    return {'signed_p50_ms': signed_median, 'p50_ms': float(np.median(values)), 'p95_ms': float(np.percentile(values, 95)),
            'max_ms': float(np.max(values))}


def plan_episode(table, info, options):
    contract = contract_for(info)
    fps = options['fps']
    state_ts, action_ts = series(table, 'state_timestamp'), series(table, 'action_timestamp')
    states = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
    actions = np.asarray(table['action'].to_pylist(), dtype=np.float32)
    if (states.shape != (table.num_rows, contract.state_dim)
            or actions.shape != (table.num_rows, contract.raw_action_dim)
            or not np.isfinite(states).all() or not np.isfinite(actions).all()):
        raise ValueError('状态或动作的维度/有限值检查失败')
    if len(set(table['task_index'].to_pylist())) != 1:
        raise ValueError('同一 episode 内含多种任务指令，须先分段')
    if len(state_ts) < 3:
        raise ValueError('少于三帧，无法对齐')
    camera_ts = {key: series(table, f'image_timestamp.{key}', repeats=True) for key in camera_keys(info)}
    corrected = {key: value + options['camera_offsets_ms'][key] / 1000 for key, value in camera_ts.items()}
    rates = {}
    for key, values in {'state': state_ts, **camera_ts}.items():
        unique = np.unique(values)
        rates[key] = float((len(unique) - 1) / (unique[-1] - unique[0])) if len(unique) > 1 else 0.0
    before = {key: metrics(values - state_ts) for key, values in camera_ts.items()}
    summary = {'input_frames': len(state_ts), 'effective_fps': rates, 'before': before}
    if min(rates.values()) < fps / 1.05:
        raise ValueError(f'有效采样频率最低为 {min(rates.values()):.2f} Hz，不能生成真实 {fps} Hz 数据；请降低目标频率')
    start = max([state_ts[0], *(v[0] for v in corrected.values())])
    end = min([state_ts[-1], *(v[-1] for v in corrected.values())])
    if contract.action_offset == 0:
        start, end = max(start, action_ts[0]), min(end, action_ts[-1])
    if end - start < 2 / fps:
        raise ValueError('各流公共时间范围不足')
    grid = start + np.arange(int(np.floor((end - start) * fps + 1e-6)) + 1) / fps
    state_indices = nearest(state_ts, grid)
    if np.any(np.diff(state_indices) <= 0):
        raise ValueError('目标频率会重复使用状态样本；请降低频率')
    state_skew = np.abs(state_ts[state_indices] - grid)
    if np.max(state_skew) > min(options['max_skew_ms'] / 1000, .5 / fps) + 1e-6:
        raise ValueError('状态缺帧或与目标时间轴偏差过大')
    # Next-observation targets require one additional real state; no synthetic hold.
    selected = state_indices[:-1] if contract.action_offset else state_indices
    output_grid = grid[:len(selected)]
    image_indices = {key: nearest(values, output_grid) for key, values in corrected.items()}
    after = {}
    for key, indices in image_indices.items():
        residual = corrected[key][indices] - state_ts[selected]
        after[key] = metrics(residual)
        if np.any(np.diff(corrected[key][indices]) <= 0):
            raise ValueError(f'{key} 会重复使用相机帧；请降低频率或修正时间戳')
        if np.max(np.abs(residual)) > options['max_skew_ms'] / 1000 + 1e-6:
            raise ValueError(f'{key} 对齐后仍超过时间偏差上限')
    if contract.action_offset:
        action_indices = state_indices[1:]
        output_action_ts = state_ts[action_indices]
    else:
        action_indices = nearest(action_ts, output_grid)
        if np.any(np.diff(action_indices) <= 0) or np.max(np.abs(action_ts[action_indices] - state_ts[selected])) > options['max_skew_ms'] / 1000:
            raise ValueError('动作命令时间戳无法在容差内对齐')
        output_action_ts = action_ts[action_indices]
    return {**summary, 'output_frames': len(selected), 'after': after,
            'grid': output_grid, 'state_indices': selected, 'action_indices': action_indices,
            'image_indices': image_indices, 'state_timestamps': state_ts[selected],
            'action_timestamps': output_action_ts,
            'image_timestamps': {key: corrected[key][indices] for key, indices in image_indices.items()}}


def inspect_dataset(source, options):
    source = Path(source)
    signature = source_signature(source)
    info = json.loads((source / 'meta/info.json').read_text())
    options = options_for(info, options)
    contract_for(info)
    paths = sorted((source / 'data').glob('chunk-*/episode_*.parquet'))
    if not paths:
        raise ValueError('没有 episode parquet 文件')
    episodes, errors = [], []
    if [int(path.stem.split('_')[-1]) for path in paths] != list(range(len(paths))):
        errors.append('episode 编号不连续，须先整理编号以保留训练/验证划分')
    output_bytes = 0
    peak_memory = 0
    # Do not silently discard event/action labels that this worker cannot remap.
    allowed = {'observation.state', 'action', 'timestamp', 'state_timestamp', 'action_timestamp',
               'frame_index', 'episode_index', 'index', 'task_index'}
    if (source / 'meta/dashboard_event_overrides').exists() or (source / 'meta/event_semantics.json').exists():
        errors.append('数据集含 Event 标注，请先导出不含事件标注的副本；当前对齐不改写事件语义')
    for path in paths:
        table = pq.read_table(path)
        try:
            extra = [key for key in table.column_names if key not in allowed
                     and not key.startswith(('observation.images.', 'image_timestamp.'))]
            if extra:
                raise ValueError(f'含尚不能重排的附加列 {extra}，不能丢弃标签')
            episode = int(path.stem.split('_')[-1])
            for key in camera_keys(info):
                feature = f'observation.images.{key}'
                if info['features'][feature]['dtype'] == 'video':
                    relative = info['video_path'].format(episode_chunk=episode // info.get('chunks_size', 1000), episode_index=episode, video_key=feature)
                    video = safe_media_path(source, relative)
                    capture = cv2.VideoCapture(str(video))
                    try:
                        video_fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
                        if not capture.isOpened() or count != table.num_rows:
                            raise ValueError(f'{key}: 视频帧数与时间戳行数不符')
                        if abs(video_fps - float(info['fps'])) > .05 * float(info['fps']):
                            raise ValueError(f'{key}: 视频编码频率 {video_fps:.2f} 与数据集标称 {info["fps"]} 不符')
                    finally:
                        capture.release()
            plan = plan_episode(table, info, options)
            episodes.append({'episode': path.stem, **{key: plan[key] for key in ('input_frames', 'output_frames', 'effective_fps', 'before', 'after')}})
            # MP4 plus optional raw RGB copy and filesystem/codec overhead.
            output_bytes += plan['output_frames'] * len(camera_keys(info)) * 3 * 256 * 256 * 2
            peak_memory = max(peak_memory, plan['output_frames'] * len(camera_keys(info)) * 3 * 256 * 256 * 6)
        except ValueError as exc:
            errors.append(f'{path.stem}: {exc}')
    if source_signature(source) != signature:
        raise ValueError('检查期间原数据发生变化，请重新检查')
    return {'status': 'inspected', 'source_signature': signature, 'source_dataset': source.name,
            'options': options, 'can_align': not errors, 'errors': errors, 'episodes': episodes,
            'estimated_output_bytes': output_bytes, 'estimated_peak_memory_bytes': peak_memory + 512 * 1024**2,
            'note': '基于已记录的接收时间戳进行最近样本对齐；不能证明曝光同步，也不能恢复未记录的真实时间。正的相机修正值加到原相机时间戳上。'}


def safe_media_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'媒体路径无效: {relative}')
    return path


def selected_images(root, info, table, episode, key, indices):
    def prepare(frame):
        if frame.shape[:2] == (224, 224):
            return frame
        if info.get('image_resize_mode') == 'stretch':
            from bimanual_vla.data.image_resize import resize_stretch
            return resize_stretch(frame, 224, 224)
        return cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    feature = f'observation.images.{key}'
    if info['features'][feature]['dtype'] == 'video':
        relative = info['video_path'].format(episode_chunk=episode // info.get('chunks_size', 1000),
                                            episode_index=episode, video_key=feature)
        video = safe_media_path(root, relative)
        capture = cv2.VideoCapture(str(video))
        frames, wanted = [], set(map(int, indices))
        try:
            if not capture.isOpened():
                raise ValueError(f'无法打开视频 {video}')
            for index in range(int(indices[-1]) + 1):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f'{video}: 实际视频帧数不足，不能复制帧补齐')
                if index in wanted:
                    frames.append(prepare(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        finally:
            capture.release()
        return np.stack(frames)
    cells = table[feature].to_pylist()
    images = []
    for index in indices:
        cell = cells[int(index)]
        if isinstance(cell, dict) and cell.get('bytes'):
            stream = io.BytesIO(cell['bytes'])
        elif isinstance(cell, dict) and cell.get('path'):
            stream = safe_media_path(root, cell['path'])
        else:
            raise ValueError(f'{feature}: 不支持的图像单元格式')
        with Image.open(stream) as image:
            images.append(prepare(np.asarray(image.convert('RGB'))))
    return np.stack(images)


def align_dataset(source, destination, preview):
    source, destination = Path(source), Path(destination)
    if destination.exists() or source.resolve() == destination.resolve():
        raise ValueError('目标须为尚不存在的新数据集')
    if not preview.get('can_align') or source_signature(source) != preview['source_signature']:
        raise ValueError('检查结果失效或原数据已变化，请重新检查')
    if shutil.disk_usage(destination.parent).free < preview['estimated_output_bytes'] + 1024**3:
        raise ValueError('目标磁盘空间不足（含临时输出和安全余量）')
    meminfo = Path('/proc/meminfo')
    if meminfo.is_file():
        available = next(int(line.split()[1]) * 1024 for line in meminfo.read_text().splitlines() if line.startswith('MemAvailable:'))
        if available < preview['estimated_peak_memory_bytes'] + 2 * 1024**3:
            raise ValueError('可用主机内存不足，请在空闲窗口重试')
    info = json.loads((source / 'meta/info.json').read_text())
    contract, options = contract_for(info), preview['options']
    stage = destination.parent / f'.{destination.name}.align-{uuid.uuid4().hex}'
    lock = destination.parent / f'.{destination.name}.alignment-lock'
    lock.mkdir()  # Refuse concurrent writers for the same destination.
    try:
        writer = Pi0LeRobotDatasetWriter(stage, fps=options['fps'], robot_type=info['robot_type'],
            state_names=info['state_names'], action_names=info['action_names'], camera_keys=camera_keys(info),
            schema=contract.schema, arm_mode=contract.arm_mode, arm_side=contract.arm_side,
            action_source=contract.action_source, action_alignment=contract.action_alignment,
            action_offset=contract.action_offset, action_horizon=contract.action_horizon,
            image_resize_mode=info.get('image_resize_mode', 'unspecified'),
            coordinate_frame=info.get('coordinate_frame', 'slave_base'),
            source_frame=info.get('source_frame', 'slave_base'), arm_base_offset=info.get('arm_base_offset'))
        episodes = {int(row['episode_index']): row for row in (json.loads(line) for line in (source / 'meta/episodes.jsonl').read_text().splitlines() if line.strip())}
        tasks = {int(row['task_index']): row['task'] for row in (json.loads(line) for line in (source / 'meta/tasks.jsonl').read_text().splitlines() if line.strip())}
        maps = stage / 'meta/time_alignment'
        maps.mkdir()
        shutil.copy2(source / 'meta/info.json', maps / 'source_info.json')
        shutil.copy2(source / 'meta/episodes.jsonl', maps / 'source_episodes.jsonl')
        for path in sorted((source / 'data').glob('chunk-*/episode_*.parquet')):
            episode = int(path.stem.split('_')[-1])
            table = pq.read_table(path)
            plan = plan_episode(table, info, options)
            state_indices, action_indices = plan['state_indices'], plan['action_indices']
            states = np.asarray(table['observation.state'].to_pylist(), np.float32)
            original_actions = np.asarray(table['action'].to_pylist(), np.float32)
            if contract.action_offset:
                actions = states[action_indices].copy()
                if contract.schema == 'delivery' and ('same_step' in contract.action_alignment or 'commanded_gripper' in contract.action_source):
                    actions[:, 9::10] = original_actions[state_indices, 9::10]
            else:
                actions = original_actions[action_indices]
            images = {key: selected_images(source, info, table, episode, key, indices)
                      for key, indices in plan['image_indices'].items()}
            row = episodes[episode]
            task_index = int(table['task_index'][int(state_indices[0])].as_py())
            # Every output row can be traced to exact source state/action/video rows.
            np.savez_compressed(maps / f'{path.stem}.npz', state_indices=state_indices,
                action_indices=action_indices, grid=plan['grid'],
                **{f'image_indices_{key}': indices for key, indices in plan['image_indices'].items()})
            writer.append_episode(states=states[state_indices], actions=actions,
                timestamps=plan['grid'], state_timestamps=plan['state_timestamps'],
                action_timestamps=plan['action_timestamps'], image_timestamps=plan['image_timestamps'],
                images=images, task_name=row.get('task_name', 'aligned'), instruction=tasks[task_index],
                success=bool(row.get('success', True)), metadata={
                    **{key: row[key] for key in ('tags', 'split') if key in row},
                    'source_dataset': source.name,
                    'source_episode_index': episode, 'terminal_padding': False,
                    'alignment_max_skew_ms': options['max_skew_ms']})
            print(f'Aligned {path.stem}: {len(states)} -> {len(state_indices)} frames', flush=True)
        for name in ('dashboard_dataset_origin.json',):
            if (source / 'meta' / name).is_file():
                shutil.copy2(source / 'meta' / name, stage / 'meta' / name)
        if 'splits' in info:
            output_info = json.loads((stage / 'meta/info.json').read_text())
            output_info['splits'] = info['splits']
            atomic_json(stage / 'meta/info.json', output_info)
        errors = check_dataset(stage)
        if errors:
            raise ValueError('输出数据校验失败: ' + '; '.join(errors[:10]))
        if source_signature(source) != preview['source_signature'] or destination.exists():
            raise ValueError('发布前检测到源变化或目标已存在；没有替换任何数据集')
        report = {**preview, 'status': 'completed', 'output_dataset': destination.name,
                  'output_path': str(destination), 'validation': 'passed',
                  'norm_stats': 'recomputed_for_aligned_output',
                  'source_preserved': True}
        atomic_json(stage / 'meta/time_alignment.json', report)
        stage.rename(destination)
        return report
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        lock.rmdir()


def main():
    def terminated(_signal, _frame):
        raise KeyboardInterrupt('alignment task stopped')
    signal.signal(signal.SIGTERM, terminated)
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    result_path = args.request.parent / 'result.json'
    try:
        cgroup = Path('/proc/self/cgroup').read_text()
        atomic_json(args.request.parent / 'runtime.json', {'pid': os.getpid(), 'cgroup': cgroup})
        expected_scope = request.get('expected_scope')
        if expected_scope and not any(line.rstrip().endswith('/' + expected_scope) for line in cgroup.splitlines()):
            raise RuntimeError('worker 未进入预期独立 scope，已停止处理')
        if request['operation'] == 'inspect':
            result = inspect_dataset(request['source'], request['options'])
        else:
            preview = json.loads(Path(request['preview_report']).read_text())
            result = align_dataset(request['source'], request['destination'], preview)
        atomic_json(result_path, result)
    except BaseException as exc:
        atomic_json(result_path, {'status': 'failed', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
