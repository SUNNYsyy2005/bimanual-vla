"""Offline train/test MSE on the local GPU; no robot or serving imports."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import types

import numpy as np

from common import configure, load_model

JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
CAMERAS = {'cam_high': 'base_0_rgb', 'cam_left_wrist': 'left_wrist_0_rgb',
           'cam_right_wrist': 'right_wrist_0_rgb'}


class Frames:
    def __init__(self, root, episodes, count, seed):
        import pyarrow.parquet as pq
        self.root = Path(root)
        self.episodes = {}
        self.images = {}
        candidates = []
        for episode in episodes:
            path = self.root / f'data/chunk-000/episode_{episode:06d}.parquet'
            table = pq.read_table(path)
            state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
            actions = np.asarray(table['action'].to_pylist(), dtype=np.float32)
            assert state.shape == actions.shape == (len(table), 14)
            np.testing.assert_array_equal(table['frame_index'].to_numpy(), np.arange(len(table)))
            self.episodes[episode] = (state, actions)
            candidates.extend((episode, frame) for frame in range(len(table)))
        self.available = len(candidates)
        indices = np.random.default_rng(seed).choice(len(candidates), min(count, len(candidates)), replace=False)
        self.selected = [candidates[i] for i in indices]

    def get(self, index):
        import cv2
        episode, frame = self.selected[index]
        if episode not in self.images:
            wanted = {f for e, f in self.selected if e == episode}
            decoded = {f: {} for f in wanted}
            for camera, model_key in CAMERAS.items():
                path = self.root / f'videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4'
                cap = cv2.VideoCapture(str(path))
                try:
                    for f in range(max(wanted) + 1):
                        assert cap.grab(), f'Video ended early: {path}, frame {f}'
                        if f in wanted:
                            ok, image = cap.retrieve()
                            assert ok and image.shape == (224, 224, 3), (path, f)
                            decoded[f][model_key] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                finally:
                    cap.release()
            self.images[episode] = decoded
        state, actions = self.episodes[episode]
        # Canonical action[t] is already the next-measured target. The training
        # loader's action_offset=1 queries t..t+49, repeating the terminal row.
        truth = actions[np.minimum(frame + np.arange(50), len(actions) - 1)].copy()
        return state[frame].copy(), truth, self.images[episode][frame]


def make_transforms(checkpoint, tokenizer):
    import sentencepiece
    from openpi import transforms
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.shared.normalize import NormStats
    candidates = sorted((Path(checkpoint) / 'assets').glob('*/norm_stats.json'))
    assert len(candidates) == 1 and candidates[0].parent.name == 'fix_nopad_trimstart_20261021'
    raw = json.loads(candidates[0].read_text())['norm_stats']
    stats = {key: NormStats(**{k: np.asarray(v) if v is not None else None for k, v in values.items()})
             for key, values in raw.items()}
    tok = object.__new__(PaligemmaTokenizer)
    tok._tokenizer = sentencepiece.SentencePieceProcessor(model_file=tokenizer)
    tok._max_len = 200
    pipeline = transforms.compose([
        transforms.DeltaActions([True] * 6 + [False] + [True] * 6 + [False]),
        transforms.Normalize(stats, use_quantiles=True),
        transforms.TokenizePrompt(tok, discrete_state_input=True),
        transforms.PadStatesAndActions(32)])
    scale = (stats['actions'].q99[:14] - stats['actions'].q01[:14] + 1e-6) / 2
    return pipeline, scale, hashlib.sha256(candidates[0].read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--samples', type=int, default=512)
    p.add_argument('--action-samples', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rtc-distribution', action='store_true',
                   help='Use the checkpoint training RTC delays and ground-truth prefixes; flow MSE only')
    args = p.parse_args()
    configure(args.source)
    import torch
    from openpi.models.model import Observation
    torch.set_num_threads(8)
    split = json.loads((Path(args.dataset) / 'meta/train_test_split.json').read_text())
    info = json.loads((Path(args.dataset) / 'meta/info.json').read_text())
    assert split['test_episodes'] == [12, 16] and split['action_offset'] == 1
    assert info['total_frames'] == 11957 and info['action_offset'] == 1
    assert not set(split['train_episodes']) & set(split['test_episodes'])
    prompt = json.loads((Path(args.dataset) / 'meta/tasks.jsonl').read_text().splitlines()[0])['task']
    pipeline, scale, norm_hash = make_transforms(args.checkpoint, args.tokenizer)
    provenance = json.loads((Path(args.checkpoint) / 'provenance.json').read_text())
    finetuning = provenance['action_contract']['finetuning']
    model, _ = load_model(args.checkpoint, 'bf16')
    # Upstream forward() requests train=True unconditionally. This isolated
    # evaluation instance always disables its crop/rotation/color augmentation.
    preprocess = model._preprocess_observation
    def without_augmentation(self, observation, *, train=True):
        return preprocess(observation, train=False)
    model._preprocess_observation = types.MethodType(without_augmentation, model)
    rtc_weights = None
    if args.rtc_distribution:
        from rtc_evaluation import install_per_token_time, flow_loss
        assert finetuning['rtc_training']
        rtc_weights = np.asarray(finetuning['rtc_training_delay_weights'], dtype=float)
        assert len(rtc_weights) == finetuning['rtc_training_max_delay'] + 1
        assert (rtc_weights >= 0).all() and rtc_weights.sum() > 0
        install_per_token_time(model)
        args.action_samples = 0
    report = dict(status='running', checkpoint=str(Path(args.checkpoint).resolve()),
                  dataset=str(Path(args.dataset).resolve()), gpu=torch.cuda.get_device_name(),
                  framework='PyTorch', precision='BF16 mixed, upstream projection/norm dtypes',
                  seed=args.seed, split_seed=split['seed'], rtc=args.rtc_distribution,
                  model_training_rtc=finetuning['rtc_training'], augmentation=False,
                  flow_metric='normalized velocity MSE, 14 physical dimensions, random Beta(1.5,1) flow times',
                  training_log_comparability=('training RTC delay distribution; no image augmentation' if args.rtc_distribution
                                              else 'delay zero; no image augmentation'),
                  action_offset=1, model_action_start_offset=1, terminal_targets='repeat final action row',
                  norm_stats_sha256=norm_hash, prompt=prompt, splits={})
    if rtc_weights is not None:
        report.update(rtc_delay_weights=rtc_weights.tolist(), rtc_max_delay=finetuning['rtc_training_max_delay'],
                      rtc_prefix='ground-truth actions, matching training; postfix-only MSE',
                      rtc_scope='training objective; does not measure asynchronous rollout performance')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for name in ['train', 'test']:
        frames = Frames(args.dataset, split[name + '_episodes'], args.samples, args.seed)
        losses, action_errors, holds = [], {5: [], 10: []}, []
        delays = (np.random.default_rng(args.seed + 10000).choice(len(rtc_weights), len(frames.selected),
                  p=rtc_weights / rtc_weights.sum()) if rtc_weights is not None else np.zeros(len(frames.selected), dtype=int))
        for index in range(len(frames.selected)):
            state, truth, images = frames.get(index)
            data = pipeline(dict(state=state.copy(), actions=truth.copy(), prompt=prompt))
            # JAX's training loader converts normalized NumPy float64 values
            # to float32 with x64 disabled. Match that boundary in PyTorch.
            data['state'] = data['state'].astype(np.float32)
            data['actions'] = data['actions'].astype(np.float32)
            obs = Observation(
                images={k: torch.from_numpy(v.copy()).permute(2, 0, 1)[None].cuda().float() / 255 * 2 - 1
                        for k, v in images.items()},
                image_masks={k: torch.ones(1, device='cuda', dtype=torch.bool) for k in images},
                state=torch.from_numpy(data['state'])[None].cuda(),
                tokenized_prompt=torch.from_numpy(data['tokenized_prompt'])[None].cuda(),
                tokenized_prompt_mask=torch.from_numpy(data['tokenized_prompt_mask'])[None].cuda())
            actions = torch.from_numpy(data['actions'])[None].cuda()
            torch.manual_seed(args.seed + index)
            noise = torch.randn_like(actions)
            flow_time = model.sample_time(1, torch.device('cuda'))
            with torch.inference_mode():
                if args.rtc_distribution:
                    if name == 'train' and index == 0:
                        ordinary = model(obs, actions, noise=noise, time=flow_time)
                        zero_delay = flow_loss(model, obs, actions, noise, flow_time, 0)
                        torch.testing.assert_close(zero_delay.mean(), ordinary.mean(), rtol=0.01, atol=1e-5)
                        difference = (zero_delay - ordinary).abs()
                        assert difference.mean().item() < 5e-4
                        report['delay_zero_equivalence'] = dict(
                            ordinary_mse=ordinary.mean().item(), rtc_zero_mse=zero_delay.mean().item(),
                            loss_mean_abs_difference=difference.mean().item(),
                            loss_max_abs_difference=difference.max().item())
                        print('delay_zero_equivalence', report['delay_zero_equivalence'], flush=True)
                        del ordinary, zero_delay, difference
                    loss = flow_loss(model, obs, actions, noise, flow_time, int(delays[index])).float().cpu().numpy()[0]
                else:
                    loss = model(obs, actions, noise=noise, time=flow_time).float().cpu().numpy()[0]
                assert np.isfinite(loss).all()
                losses.append(loss.mean(axis=0))
                if index < args.action_samples:
                    hold = pipeline(dict(state=state.copy(), actions=np.broadcast_to(state, truth.shape).copy(), prompt=prompt))['actions']
                    holds.append((hold[:, :14] - data['actions'][:, :14]) ** 2)
                    for steps in action_errors:
                        pred = model.sample_actions(torch.device('cuda'), obs, noise=noise.clone(), num_steps=steps)
                        pred = pred.float().cpu().numpy()[0, :, :14]
                        assert np.isfinite(pred).all()
                        action_errors[steps].append((pred - data['actions'][:, :14]) ** 2)
            if index % 32 == 0:
                print(f'{name}: {index + 1}/{len(frames.selected)}, flow_mse={np.mean(np.asarray(losses)[:, :14]):.6f}', flush=True)
        losses = np.asarray(losses, dtype=np.float64)
        means = losses.mean(axis=0)
        result = dict(episodes=split[name + '_episodes'], available_frames=frames.available,
                      samples=len(losses), frame_indexes=frames.selected, flow_mse=float(means[:14].mean()),
                      flow_mse_per_dim=means.tolist(),
                      flow_mse_sample_se=float(losses[:, :14].mean(axis=1).std(ddof=1) / np.sqrt(len(losses))),
                      flow_mse_joint=float(means[JOINTS].mean()), flow_mse_gripper=float(means[[6, 13]].mean()),
                      sampled_actions={})
        if args.rtc_distribution:
            result['rtc_delays'] = delays.tolist()
            result['rtc_by_delay'] = {
                str(d): dict(samples=int((delays == d).sum()),
                             flow_mse=float(losses[delays == d, :14].mean()))
                for d in range(len(rtc_weights)) if (delays == d).any()}
        episode_ids = np.asarray(frames.selected)[:, 0]
        result['per_episode_flow_mse'] = {
            str(e): dict(samples=int((episode_ids == e).sum()),
                         flow_mse=float(losses[episode_ids == e, :14].mean()))
            for e in split[name + '_episodes'] if (episode_ids == e).any()}
        hold_mse = float(np.mean(holds)) if holds else None
        for steps, errors in action_errors.items():
            if not errors:
                continue
            errors = np.asarray(errors)
            physical = errors * scale ** 2
            result['sampled_actions'][str(steps)] = dict(samples=len(errors), normalized_mse=float(errors.mean()),
                hold_baseline_normalized_mse=hold_mse, normalized_mse_per_dim=errors.mean(axis=(0, 1)).tolist(),
                joint_rmse_rad=float(np.sqrt(physical[..., JOINTS].mean())),
                gripper_rmse_fraction=float(np.sqrt(physical[..., [6, 13]].mean())))
        report['splits'][name] = result
        output.write_text(json.dumps(report, indent=2))
        print(name, json.dumps({k: v for k, v in result.items() if k != 'frame_indexes'}), flush=True)
        del frames
    report.update(status='completed', duration_s=time.monotonic() - started,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                  test_train_flow_ratio=report['splits']['test']['flow_mse'] / report['splits']['train']['flow_mse'])
    output.write_text(json.dumps(report, indent=2))
    print('RESULT=' + str(output), flush=True)


if __name__ == '__main__':
    main()
