"""Standalone OpenPI PyTorch loading; no robot control imports."""
import os
from pathlib import Path
import sys


def configure(source):
    # JAX is used only for reading Orbax and model config, never GPU allocation.
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    sys.path.insert(0, str(Path(source) / 'src'))
    sys.path.insert(0, str(Path(source) / 'packages/openpi-client/src'))


def make_model():
    import torch
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    config = Pi0Config(pi05=True, action_dim=32, action_horizon=50,
                       pytorch_compile_mode=None)
    with torch.device('meta'):
        model = PI0Pytorch(config)
    # Sampling invokes the decoders, never either vocabulary prediction head.
    # Removing the unused expert head also prevents allocating random weights.
    model.paligemma_with_expert.paligemma.lm_head = torch.nn.Identity()
    model.paligemma_with_expert.gemma_expert.lm_head = torch.nn.Identity()
    return model.eval()


def load_model(checkpoint, mode='bf16'):
    import json
    import torch
    from safetensors import safe_open
    model = make_model()
    root = Path(checkpoint)
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    expected = model.state_dict()
    if set(index) != set(expected):
        raise ValueError(f'Weight key mismatch: missing={set(expected)-set(index)}, extra={set(index)-set(expected)}')
    quantized = []
    if mode == 'nf4':
        import bitsandbytes as bnb
        for name, module in list(model.named_modules()):
            if isinstance(module, torch.nn.Linear) and name.startswith('paligemma_with_expert.paligemma.model.language_model.'):
                parent_name, key = name.rsplit('.', 1)
                replacement = bnb.nn.Linear4bit(module.in_features, module.out_features,
                    bias=module.bias is not None, compute_dtype=torch.bfloat16,
                    compress_statistics=True, quant_type='nf4', device='meta')
                setattr(model.get_submodule(parent_name), key, replacement)
                quantized.append(name)
    # Load one shard and one parameter at a time to bound CPU/GPU staging memory.
    for filename in sorted(set(index.values())):
        with safe_open(root / filename, framework='pt', device='cpu') as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if tensor.shape != expected[name].shape:
                    raise ValueError(f'Shape mismatch for {name}: {tensor.shape} vs {expected[name].shape}')
                parent_name, key = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
                dtype = expected[name].dtype
                if parent_name in quantized and key == 'weight':
                    parent.weight = bnb.nn.Params4bit(tensor.to(torch.bfloat16), requires_grad=False,
                        compress_statistics=True, quant_type='nf4').to('cuda')
                else:
                    parent._parameters[key] = torch.nn.Parameter(tensor.to(device='cuda', dtype=dtype),
                                                                 requires_grad=False)
    del expected
    # Rebuild deterministic nonpersistent buffers after meta initialization.
    for name, buffer in list(model.named_buffers()):
        if buffer.device.type != 'cuda':
            parent_name, key = name.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
            if buffer.device.type == 'meta':
                if key == 'position_ids':
                    value = torch.arange(buffer.shape[-1], device='cuda')[None]
                elif key == 'inv_freq' and hasattr(parent, 'rope_init_fn'):
                    value, parent.attention_scaling = parent.rope_init_fn(parent.config, torch.device('cuda'))
                    parent.original_inv_freq = value
                else:
                    raise ValueError(f'No deterministic buffer initializer for {name}')
            else:
                value = buffer.to('cuda')
            parent._buffers[key] = value
    return model, quantized
