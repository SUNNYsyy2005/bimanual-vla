"""Offline RTX 5060 benchmark on recorded Piper observations; no CAN access."""
import argparse
import json
from pathlib import Path
import statistics
import time

from common import configure, load_model


def observation(args):
    import cv2
    import numpy as np
    import sentencepiece
    import torch
    from openpi.models.model import Observation
    run = Path(args.recording)
    state = np.load(run / 'trajectory.npz')['qpos'][0].copy()
    state[[6, 13]] /= .07  # Recording stores metres; Piper model uses fractions.
    assets = Path(args.checkpoint) / 'assets'
    candidates = ([assets / args.asset_id / 'norm_stats.json'] if args.asset_id else
                  sorted(assets.glob('*/norm_stats.json')))
    if len(candidates) != 1 or not candidates[0].is_file():
        raise ValueError('Select the checkpoint norm stats with --asset-id')
    stats = json.loads(candidates[0].read_text())['norm_stats']
    lo, hi = np.asarray(stats['state']['q01']), np.asarray(stats['state']['q99'])
    normalized = 2 * (state - lo) / (hi - lo + 1e-6) - 1
    # OpenPI tokenizes the 14 physical dimensions before padding state to 32.
    token_state = normalized.copy()
    normalized = np.pad(normalized, (0, 32 - len(state))).astype('float32')
    sp = sentencepiece.SentencePieceProcessor(model_file=args.tokenizer)
    discrete = np.digitize(token_state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    prompt = args.prompt.strip().replace('_', ' ').replace('\n', ' ')
    tokens = sp.encode(f'Task: {prompt}, State: {" ".join(map(str, discrete))};\nAction: ', add_bos=True)
    tokens = tokens[:200]
    mask = [True] * len(tokens) + [False] * (200 - len(tokens))
    tokens += [0] * (200 - len(tokens))
    images = {}
    for camera, key in [('cam_high','base_0_rgb'), ('cam_left_wrist','left_wrist_0_rgb'),
                        ('cam_right_wrist','right_wrist_0_rgb')]:
        cap = cv2.VideoCapture(str(run / 'videos' / (camera + '.mp4')))
        ok, frame = cap.read(); cap.release()
        if not ok: raise ValueError(f'Cannot read {camera}')
        if frame.shape[:2] == (256, 256):
            frame = frame[55:200]  # Undo recorder padding before training stretch.
        rgb = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
        images[key] = torch.from_numpy(rgb.astype('float32') / 127.5 - 1).permute(2,0,1)[None].cuda()
    return Observation(images=images, image_masks={k:torch.ones(1,dtype=torch.bool,device='cuda') for k in images},
        state=torch.from_numpy(normalized)[None].cuda(),
        tokenized_prompt=torch.tensor([tokens], dtype=torch.long, device='cuda'),
        tokenized_prompt_mask=torch.tensor([mask], dtype=torch.bool, device='cuda'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--asset-id', help='Checkpoint normalization asset; auto-select if unique')
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--recording', required=True)
    p.add_argument('--mode', choices=['bf16','nf4'], default='bf16')
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--repeats', type=int, default=4)
    p.add_argument('--prompt', default='Place the bottles on the black pad in the center of the table.')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    configure(args.source)
    import numpy as np
    import torch
    torch.set_num_threads(8)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    report = {'mode':args.mode,'steps':args.steps,'rtc':False,'compile':False,
              'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
              'checkpoint':str(Path(args.checkpoint).resolve()), 'recording':args.recording,
              'input_kind':'lossy_recorded_observation_no_future_ground_truth'}
    started = time.perf_counter()
    try:
        model, quantized = load_model(args.checkpoint, args.mode)
        torch.cuda.synchronize()
        report['load_s'] = time.perf_counter() - started
        report['quantized_layers'] = len(quantized)
        report['loaded_allocated_mib'] = torch.cuda.memory_allocated() / 1024**2
        obs = observation(args)
        latencies, predictions = [], []
        for i in range(args.repeats + 1):
            torch.manual_seed(0)
            noise = torch.randn(1,50,32,device='cuda')
            torch.cuda.synchronize(); start = time.perf_counter()
            with torch.inference_mode():
                actions = model.sample_actions(torch.device('cuda'), obs, noise=noise, num_steps=args.steps)
            torch.cuda.synchronize(); latency = time.perf_counter() - start
            pred = actions.float().cpu().numpy()
            if pred.shape != (1,50,32) or not np.isfinite(pred).all():
                raise ValueError(f'Invalid output: {pred.shape}')
            latencies.append(latency); predictions.append(pred)
            print(f'{args.mode} request {i}: {latency:.3f}s, allocated peak {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB', flush=True)
        report.update(success=True,cold_s=latencies[0],warm_s=latencies[1:],
            warm_median_s=statistics.median(latencies[1:]),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
            output_shape=list(predictions[0].shape),all_finite=True)
        np.savez_compressed(output.with_suffix('.npz'), normalized_actions=np.concatenate(predictions))
    except Exception as exc:
        report.update(success=False,error=f'{type(exc).__name__}: {exc}',
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)
        raise
    finally:
        output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
