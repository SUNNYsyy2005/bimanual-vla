"""Register a family of PPO runs without copying curves or changing other sources."""
import argparse
import copy
import json
from pathlib import Path


def configure_ppo_monitor(config, *, root, source_id='h100_low_sr_ppo_20261022', ssh_host='login-server',
                          python='/DATA/disk0/sunny/miniconda3/envs/safe-rl-wm-train/bin/python'):
    result = copy.deepcopy(config)
    block = result.setdefault('rlinf_dashboard', {})
    source = dict(id=source_id, label='PPO-Lag · H100 · 低 SR 多任务', target='H100',
                  root=str(root), ssh_host=ssh_host, python=python, max_depth=2,
                  evaluation_cohort='fixed_validation', query_slurm=True)
    sources = block.setdefault('sources', [])
    sources[:] = [s for s in sources if s.get('id') != source_id]
    sources.append(source)
    return result


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--root', required=True)
    args=ap.parse_args()
    before=args.config.read_bytes()
    result=configure_ppo_monitor(json.loads(before),root=args.root)
    backup=args.config.with_suffix(args.config.suffix+'.before-ppo-monitor')
    if not backup.exists():
        backup.write_bytes(before)
        backup.chmod(0o600)
    temp=args.config.with_suffix(args.config.suffix+'.ppo.tmp')
    temp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.chmod(0o600);temp.replace(args.config)
    print('Registered PPO monitor source; existing sources and launch profiles preserved.')
