"""Run on the training node in a CPU Slurm job; never changes training files."""
import argparse
import errno
import json
import os
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume-export', type=Path)
    args = p.parse_args()
    if args.resume_export:
        dest = args.resume_export
        manifest = json.loads((dest / 'recovery.json').read_text())
    else:
        candidates = sorted(args.run.glob('checkpoints/global_step_*/recovery.json'),
                            key=lambda x: int(x.parent.name.split('_')[-1]), reverse=True)
        if not candidates:
            raise RuntimeError('No completed recovery checkpoint')
        marker = candidates[0]
        manifest = json.loads(marker.read_text())
        model_bytes = sum(size for name, size in manifest['files'].items() if name.startswith('model/'))
        # Preserve room for the trainer's next full model+optimizer recovery save.
        reserve = sum(manifest['files'].values()) + 10 * 2**30
        required = 3 * model_bytes + reserve
        args.output.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(args.output).free < required:
            raise RuntimeError(f'Export requires {required / 2**30:.1f} GiB free including training-save reserve; use another filesystem')
        dest = args.output / marker.parent.name
        dest.mkdir(parents=True, exist_ok=False)
        snapshot = dest / 'model'
        snapshot.mkdir()
        for name, size in manifest['files'].items():
            if not name.startswith('model/'):
                continue
            src = marker.parent / name
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, target)  # immutable inode survives training retention
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copyfile(src, target)
            if target.stat().st_size != size:
                raise RuntimeError(f'Incomplete model shard: {name}')
        (dest / 'recovery.json').write_text(json.dumps(manifest, indent=2))
        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
        dcp_to_torch_save(snapshot, dest / 'model.pt')
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    state = torch.load(dest / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
    state = state.get('state', state)
    state = state.get('model', state)
    index = json.loads((args.template / 'model.safetensors.index.json').read_text())
    weights = index['weight_map']
    if set(state) != set(weights):
        raise RuntimeError(f'Checkpoint keys differ: missing={set(weights)-set(state)}, extra={set(state)-set(weights)}')
    remaining_bytes = sum(t.numel() * t.element_size() for t in state.values())
    training_reserve = sum(manifest['files'].values()) if args.run.stat().st_dev == dest.stat().st_dev else 0
    if shutil.disk_usage(dest).free < remaining_bytes + training_reserve + 10 * 2**30:
        raise RuntimeError('Insufficient free space for HF export plus the next training recovery save')
    out = dest / 'hf_ckpt.tmp'
    out.mkdir(exist_ok=True)
    for src in args.template.iterdir():
        if src.is_file() and not src.name.endswith('.safetensors'):
            shutil.copy2(src, out / src.name)
    for shard in sorted(set(weights.values())):
        names = [k for k, v in weights.items() if v == shard]
        with safe_open(args.template / shard, framework='pt', device='cpu') as template:
            for name in names:
                if list(state[name].shape) != list(template.get_slice(name).get_shape()):
                    raise RuntimeError(f'Shape mismatch: {name}')
        save_file({k: state[k].contiguous() for k in names}, out / shard, metadata={'format': 'pt'})
    index['metadata']['total_size'] = sum(t.numel()*t.element_size() for t in state.values())
    (out / 'model.safetensors.index.json').write_text(json.dumps(index))
    shutil.copy2(args.run / 'robot_configs/piper_bimanual.yaml', out / 'piper_bimanual.yaml')
    shutil.copy2(args.run.parents[1] / 'data/piper-v2/norm_stats/piper_bimanual.json', out / 'piper_bimanual_norm.json')
    (out / 'export_source.json').write_text(json.dumps({'run': str(args.run), 'step': manifest['step'], 'kind': 'recovery_export'}, indent=2))
    out.rename(dest / 'hf_ckpt')
    del state
    (dest / 'model.pt').unlink()
    if (dest / 'model').exists():
        shutil.rmtree(dest / 'model')
    print(f'EXPORT_COMPLETE {dest / "hf_ckpt"}', flush=True)


if __name__ == '__main__':
    main()
