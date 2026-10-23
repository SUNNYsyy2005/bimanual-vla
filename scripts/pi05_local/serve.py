"""Serve the verified local Piper Pi05 checkpoint over OpenPI WebSocket."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import types

import numpy as np

from common import configure, load_model

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path('/home/user/models/pi05_5060')
EXPERIMENT = 'joint_trim_nopad_no_rtc_b4_20261022'
CAMERAS = {'cam_high': 'base_0_rgb', 'cam_left_wrist': 'left_wrist_0_rgb',
           'cam_right_wrist': 'right_wrist_0_rgb'}
DELTA_MASK = [True] * 6 + [False] + [True] * 6 + [False]


class PiperInputs:
    def __init__(self, state_dim=14, arm_side='both', training_arm_side=None):
        self.state_dim = state_dim
        self.arm_side = arm_side
        self.training_arm_side = training_arm_side or arm_side
        if state_dim == 7 and self.training_arm_side not in ('left', 'right'):
            raise ValueError('Invalid training camera side')
        self.cameras = CAMERAS if state_dim == 14 else {
            'cam_high': 'base_0_rgb',
            f'cam_{arm_side}_wrist': f'{self.training_arm_side}_wrist_0_rgb'}

    def __call__(self, data):
        state = np.asarray(data['state'], dtype=np.float32)
        if state.shape != (self.state_dim,) or not np.isfinite(state).all():
            raise ValueError(f'Expected finite {self.state_dim}D Piper state: joint radians, gripper opening fraction')
        if np.any(state[list(range(6, self.state_dim, 7))] < -0.01) or np.any(state[list(range(6, self.state_dim, 7))] > 1.01):
            raise ValueError('State grippers must use opening fractions between 0 and 1')
        if set(data['images']) != set(self.cameras):
            raise ValueError(f'Expected cameras {list(self.cameras)}')
        images = {}
        for name, key in self.cameras.items():
            image = np.asarray(data['images'][name])
            if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
                image = image.transpose(1, 2, 0)
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f'{name}: expected uint8 RGB HWC or CHW image')
            images[key] = image
        masks = {k: np.True_ for k in images}
        if self.state_dim == 7:
            absent = 'left_wrist_0_rgb' if self.training_arm_side == 'right' else 'right_wrist_0_rgb'
            images[absent] = np.zeros_like(images[f'{self.training_arm_side}_wrist_0_rgb'])
            masks[absent] = np.False_
        # Preserve training image token order, including the masked opposite wrist.
        images = {k: images[k] for k in CAMERAS.values()}
        result = dict(state=state.copy(), image=images, image_mask=masks)
        if 'actions' in data:
            actions = np.asarray(data['actions'], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != self.state_dim or not np.isfinite(actions).all():
                raise ValueError(f'Expected finite Tx{self.state_dim} absolute actions')
            result['actions'] = actions.copy()
        if 'prompt' in data:
            result['prompt'] = data['prompt']
        return result


class Float32Inputs:
    def __call__(self, data):
        # Match the JAX training loader's x64-disabled conversion boundary.
        for key in ('state', 'actions'):
            if key in data:
                data[key] = data[key].astype(np.float32)
        return data


class PiperOutputs:
    def __init__(self, action_dim=14):
        self.action_dim = action_dim

    def __call__(self, data):
        actions = np.asarray(data['actions'][:, :self.action_dim], dtype=np.float32)
        if actions.shape != (50, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f'Invalid model output; expected finite 50x{self.action_dim} absolute targets')
        return {'actions': actions}


def offload_word_embedding(model):
    """Keep the large vocabulary lookup on CPU, with exact BF16 row copies."""
    import torch
    embedding = model.paligemma_with_expert.paligemma.language_model.embed_tokens
    tokens = torch.tensor([[0, 1, 100, embedding.num_embeddings - 1]], device='cuda')
    with torch.inference_mode():
        expected = embedding(tokens)
        original = embedding.forward
        saved_mib = embedding.weight.numel() * embedding.weight.element_size() / 1024**2
        embedding.to('cpu')

        def forward(self, inputs):
            return original(inputs.to('cpu')).to(inputs.device)

        embedding.forward = types.MethodType(forward, embedding)
        assert torch.equal(expected, embedding(tokens)), 'CPU embedding lookup changed values'
    torch.cuda.empty_cache()
    logging.info('Word embedding lookup on CPU; %.0f MiB GPU weights released; BF16 unchanged', saved_mib)


def create_policy(args):
    import sentencepiece
    import torch
    from openpi import transforms
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.policy import Policy
    from openpi.shared.normalize import NormStats
    from bimanual_vla.data.image_resize import StretchResizeImages
    from bimanual_vla.deployment.rtc_policy import RTCConfig, build_rtc_policy
    from rtc_sampling import install_trained_sampler

    checkpoint = Path(args.checkpoint).resolve()
    provenance = json.loads((checkpoint / 'provenance.json').read_text())
    if provenance.get('checkpoint_step') != 15000 or not provenance.get('all_checks_passed'):
        raise ValueError('Expected the verified 15000-step checkpoint')
    contract = provenance['action_contract']
    settings = contract['finetuning']
    if settings['image_resize_mode'] != 'stretch':
        raise ValueError('This entry point requires stretch images')
    if (contract['arm_mode'], contract['schema'], contract['state_dim']) not in (
            ('bimanual', 'joint', 14), ('single', 'joint', 7)):
        raise ValueError('Expected a single-arm or bimanual joint-space Piper contract')
    dim = contract['state_dim']
    training_side = contract['arm_side']
    side = provenance.get('deployment_arm_side', training_side)
    if dim != 7 and side != training_side:
        raise ValueError('Arm-side relabeling is only supported for single-arm models')
    if dim == 7 and side not in ('left', 'right'):
        raise ValueError('Single-arm checkpoint must specify left or right')
    rtc = bool(settings['rtc_training'])
    delay = settings['rtc_training_max_delay'] if rtc else 0
    delta_mask = ([True] * 6 + [False]) * (dim // 7)
    inputs = PiperInputs(dim, side, training_side)
    prompt = args.prompt or provenance.get('default_prompt', 'put the bottle on the balck pad')
    norm_path = checkpoint / 'assets' / contract['dataset_id'] / 'norm_stats.json'
    raw = json.loads(norm_path.read_text())['norm_stats']
    stats = {k: NormStats(**{key: np.asarray(value) if value is not None else None
                             for key, value in values.items()}) for k, values in raw.items()}
    tokenizer = object.__new__(PaligemmaTokenizer)
    tokenizer._tokenizer = sentencepiece.SentencePieceProcessor(model_file=args.tokenizer)
    tokenizer._max_len = 200
    model, _ = load_model(str(checkpoint), 'bf16')
    if rtc:
        install_trained_sampler(model, delay)
    metadata = dict(contract, arm_side=side, training_arm_side=training_side, action_dim=dim, wire_action_dim=dim,
        robot_type='piper_bimanual' if dim == 14 else 'piper', model_variant='pi05', checkpoint=str(checkpoint),
        checkpoint_step=15000, experiment=provenance.get('experiment', checkpoint.parent.name), precision='bf16',
        camera_keys=list(inputs.cameras), image_resize_mode='stretch', image_transport='raw',
        action_horizon=50, action_hz=20.0, action_time_step_s=0.05,
        action_start_offset_steps=contract['model_action_start_offset'],
        action_semantics=contract['wire_action_semantics'],
        action_convention=contract['wire_action_convention'],
        model_gripper_semantics=contract['gripper_semantics'],
        transport='openpi_websocket_v1', default_prompt=prompt,
        rtc_supported=rtc, rtc_mode='trained' if rtc else 'off', rtc_backend='pytorch',
        rtc_algorithm='real_time_chunking_clean_prefix',
        rtc_training_max_delay=delay,
        rtc_physical_action_dim=dim, rtc_chunk_origin_reanchoring=True,
        rtc_temporal_consistency=False, denoising_steps=args.steps,
        policy_instance_id=f'{os.getpid()}-{time.time_ns()}',
        policy_instance_started_at=time.time(),
        rtc_capabilities=dict(max_execution_horizon=50, max_guidance_weight=5.0,
                              prefix_attention_schedules=['zeros', 'ones', 'linear', 'exp'],
                              client_planner_prefix=True))
    policy = Policy(model, is_pytorch=True, pytorch_device='cuda', metadata=metadata,
        sample_kwargs={'num_steps': args.steps},
        transforms=[inputs, transforms.InjectDefaultPrompt(prompt),
            transforms.DeltaActions(delta_mask), transforms.Normalize(stats, use_quantiles=True),
            StretchResizeImages(224, 224), transforms.TokenizePrompt(tokenizer, discrete_state_input=True),
            transforms.PadStatesAndActions(32), Float32Inputs()],
        output_transforms=[transforms.Unnormalize(stats, use_quantiles=True),
                           transforms.AbsoluteActions(delta_mask), PiperOutputs(dim)])
    # Policy.__init__ moves the model to CUDA, so offload after constructing it.
    offload_word_embedding(model)
    metadata['word_embedding_device'] = 'cpu'
    if not rtc:
        metadata['rtc_capabilities'] = {}
        metadata['rtc_algorithm'] = None
        return policy, metadata
    config = RTCConfig(mode='trained', training_max_delay=delay,
        physical_action_dim=dim, reanchor_action_mask=tuple(delta_mask),
        temporal_consistency=False)
    return build_rtc_policy(policy, config), metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='/home/user/src/openpi-5060')
    p.add_argument('--checkpoint', default=str(MODEL_ROOT / EXPERIMENT / '15000_pt'))
    p.add_argument('--tokenizer', default=str(MODEL_ROOT / 'paligemma_tokenizer.model'))
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8099)
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--prompt', default=None)
    args = p.parse_args()
    if not 1 <= args.steps <= 50:
        p.error('--steps must be between 1 and 50')
    configure(args.source)
    sys.path.insert(0, str(ROOT))
    import torch
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from bimanual_vla.deployment.transport import serve_policy_requests
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    torch.set_num_threads(8)
    policy, metadata = create_policy(args)

    class Server(WebsocketPolicyServer):
        async def _handler(self, websocket):
            await serve_policy_requests(self._policy, self._metadata, websocket)

    logging.info('Loaded %s step 15000 on %s; BF16, %s steps, RTC max delay %s',
                 EXPERIMENT, torch.cuda.get_device_name(), args.steps, metadata['rtc_training_max_delay'])
    logging.info('Serving ws://%s:%s ; HTTP /healthz', args.host, args.port)
    Server(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == '__main__':
    main()
