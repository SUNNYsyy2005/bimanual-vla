"""Convert a full Piper Orbax parameter checkpoint, using upstream mappings."""
import argparse
import ast
import gc
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from common import configure, make_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    configure(args.source)
    import numpy as np
    import torch
    from flax.nnx import traversals
    from openpi.models.model import restore_params
    from openpi.models import gemma
    from safetensors.torch import save_file
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'model.safetensors.index.json').exists():
        raise FileExistsError('Conversion already exists; use a different output directory')
    # Execute only the two upstream tensor mapping functions. Avoid importing
    # the training config and its unrelated dataset/training dependencies.
    converter = Path(args.source) / 'examples/convert_jax_model_to_pytorch.py'
    tree = ast.parse(converter.read_text())
    names = {'slice_paligemma_state_dict', 'slice_gemma_state_dict'}
    functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    scope = {'np': np, 'torch': torch}
    exec(compile(functions, str(converter), 'exec'), scope)
    print('Restoring full parameters on CPU', flush=True)
    params = restore_params(str(Path(args.checkpoint) / 'params'), restore_type=np.ndarray, dtype='float32')
    flat = traversals.flatten_mapping(params['PaliGemma'], sep='/')
    if any('lora' in key for key in flat):
        raise ValueError('LoRA checkpoint requires explicit merge; refusing to drop adapters')
    cfg = SimpleNamespace(vision_config=SimpleNamespace(hidden_size=1152, num_hidden_layers=27,
                              num_attention_heads=16, intermediate_size=4304, patch_size=14, projection_dim=2048),
                          text_config=SimpleNamespace(hidden_size=2048, num_hidden_layers=18,
                              num_attention_heads=8, head_dim=256, intermediate_size=16384))
    vlm, expert = scope['slice_paligemma_state_dict'](flat, cfg)
    expert = scope['slice_gemma_state_dict'](expert, gemma.get_config('gemma_300m'),
                    num_expert=1, checkpoint_dir='pi05', pi05=True)
    weights = {**vlm, **expert}
    for key in ('action_in_proj', 'action_out_proj', 'time_mlp_in', 'time_mlp_out'):
        for part, suffix in [('kernel', 'weight'), ('bias', 'bias')]:
            arr = params[key][part]
            if isinstance(arr, dict): arr = arr['value']
            weights[f'{key}.{suffix}'] = torch.from_numpy(np.asarray(arr).T if part == 'kernel' else np.asarray(arr))
    model = make_model()
    expected = model.state_dict()
    if set(weights) != set(expected):
        raise ValueError(f'Conversion key mismatch: missing={set(expected)-set(weights)}, extra={set(weights)-set(expected)}')
    for key, tensor in weights.items():
        if tensor.shape != expected[key].shape:
            raise ValueError(f'Shape mismatch {key}: {tensor.shape} != {expected[key].shape}')
    # Match upstream mixed precision; projection/norm parameters remain FP32.
    converted = {key: tensor.to(expected[key].dtype).contiguous() for key, tensor in weights.items()}
    del weights, params, flat, vlm, expert, model, expected
    gc.collect()
    shards, current, size = [], {}, 0
    for key, tensor in converted.items():
        if size > 512 * 1024**2:
            shards.append(current); current, size = {}, 0
        current[key] = tensor; size += tensor.numel() * tensor.element_size()
    if current: shards.append(current)
    index = {}
    for i, shard in enumerate(shards, 1):
        filename = f'model-{i:05d}-of-{len(shards):05d}.safetensors'
        save_file(shard, output / filename)
        index.update({key: filename for key in shard})
    assets = Path(args.checkpoint) / 'assets'
    if assets.exists(): shutil.copytree(assets, output / 'assets', dirs_exist_ok=True)
    (output / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': index}, indent=2))
    (output / 'conversion.json').write_text(json.dumps({'checkpoint':str(Path(args.checkpoint).resolve()),
        'pi05':True,'action_dim':32,'action_horizon':50,'parameters':len(index),
        'unused_vocabulary_heads_removed':True}, indent=2))
    print(f'Strict key/shape checks passed: {len(index)} parameters, {len(shards)} shards', flush=True)


if __name__ == '__main__':
    main()
