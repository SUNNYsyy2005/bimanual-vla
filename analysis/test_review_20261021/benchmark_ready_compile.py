"""Offline paired benchmark on recorded splice states; never connects hardware.

The baseline is read from Git so both implementations use identical inputs
and run under the same CPU load. Actual online compilation is reported
separately in the deployment recording.
"""
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np

from bimanual_vla.deployment.chunk_planner import compile_chunk
from bimanual_vla.deployment.servo import ServoConfig
from review import J, read, stat


def main():
    name = 'bimanual_vla.deployment._benchmark_baseline'
    baseline = types.ModuleType(name)
    sys.modules[name] = baseline
    source = subprocess.check_output(
        ['git', 'show', 'd125d4e:bimanual_vla/deployment/chunk_planner.py'], text=True)
    exec(compile(source, '<baseline chunk planner d125d4e>', 'exec'), baseline.__dict__)
    path = next(Path('deployment_runs').glob('*T163923*'))
    samples = read(path / 'servo_commands.jsonl')
    control = read(path / 'trajectory.jsonl')
    models = {m['generation']: m for m in read(path / 'model_commands.jsonl')}
    cfg = json.loads((path / 'metadata.json').read_text())['runtime_configuration']
    config = ServoConfig(**{k: v for k, v in cfg['servo'].items() if k != 'backend'})
    limits = np.array(cfg['effective_joint_limits_rad']['left'] + cfg['effective_joint_limits_rad']['right'])
    lower, upper = np.zeros(14), np.full(14, .07)
    lower[J], upper[J] = limits[:, 0], limits[:, 1]
    plans = {s['plan_timing']['generation']: s['plan_timing'] for s in samples if s.get('plan_timing')}
    results = []
    for generation, timing in plans.items():
        start = timing['splice_monotonic']
        state = min(samples, key=lambda s: abs((s.get('planned_monotonic') or 0) - start))
        if abs(state['planned_monotonic'] - start) > 1e-6:
            continue
        m = models[generation]
        skip = next((r['command_queue_index'] for r in control
                     if r['command_generation'] == generation and r['command_sent']), None)
        if skip is None:
            continue
        targets = np.load(path / m['action_file'])['raw_actions'][skip:].astype(float)
        targets[:, 6::7] *= .07
        targets = np.clip(targets, lower, upper)
        indices = np.arange(skip, skip + len(targets))
        clocks = m['captured_monotonic'] + (indices + 1) / 20
        args = (state['command_qpos'], state['velocity'], state['acceleration'],
                lower, upper, config, targets, clocks, indices)
        timings = [[], []]
        for repeat in range(4):
            for which in (range(2) if repeat % 2 else reversed(range(2))):
                before = time.perf_counter()
                (baseline.compile_chunk, compile_chunk)[which](*args, start=start, generation=generation)
                timings[which].append((time.perf_counter() - before) * 1000)
        results.append(dict(generation=generation, baseline_ms=float(np.median(timings[0])),
                            optimized_ms=float(np.median(timings[1]))))
    result = dict(run=path.name, baseline_commit='d125d4e',
                  baseline_ms=stat([r['baseline_ms'] for r in results]),
                  optimized_ms=stat([r['optimized_ms'] for r in results]), samples=results)
    output = Path(__file__).with_name('ready_compile_benchmark.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'samples'}, indent=2))


if __name__ == '__main__':
    main()
