"""Run in the LingBot model Conda environment on a GPU workstation."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lingbot-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--port', type=int, default=8101)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--task', default='Place the bottles on the black pad in the center of the table.')
    args = p.parse_args()
    root, checkpoint = args.lingbot_root.resolve(), args.checkpoint.resolve()
    training_config = checkpoint.parent.parent.parent / 'lingbotvla_cli.yaml'
    if not training_config.is_file():
        raise FileNotFoundError(f'Missing original training config: {training_config}')
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    os.chdir(root)
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from deploy.websocket_policy_server import WebsocketPolicyServer
    from bimanual_vla.deployment.lingbot_piper import PiperPolicy, metadata, normalize_legacy_data_config
    config_target = root / 'configs/robot_configs/piper_bimanual.yaml'
    robot_config = (checkpoint / 'piper_bimanual.yaml').read_bytes()
    if config_target.exists() and config_target.read_bytes() != robot_config:
        raise RuntimeError('Existing Piper config differs; refusing to overwrite')
    config_target.write_bytes(robot_config)
    policy = LingbotVLAv2Server(str(checkpoint), robot_norm_path=str(checkpoint / 'piper_bimanual_norm.json'),
                               use_length=50, chunk_ret=True, use_bf16=True, use_compile=False)
    policy.vla.model.flexpi_params['num_steps'] = 4
    normalize_legacy_data_config(policy.data_config)
    policy.infer({'reset': True, 'robo_name': 'piper_bimanual'})
    source = json.loads((checkpoint / 'export_source.json').read_text())
    WebsocketPolicyServer(PiperPolicy(policy, args.task), host=args.host, port=args.port,
                          metadata=metadata(source=source)).serve_forever()


if __name__ == '__main__':
    main()
