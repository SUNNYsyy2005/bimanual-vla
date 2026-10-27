"""Launch: python -m bimanual_vla.deployment.lingbot_sync_client --help."""
from __future__ import annotations

import sys
import json
from pathlib import Path

SERIALS = {'left': '003D00395246570620323934', 'right': '003C00365246570620323934'}


def resolve_can(serial, root=Path('/sys/class/net')):
    matches = []
    for interface in root.glob('can*'):
        for parent in (interface.resolve(), *interface.resolve().parents):
            marker = parent / 'serial'
            if marker.is_file() and marker.read_text().strip() == serial:
                matches.append(interface.name)
                break
    if len(matches) != 1:
        raise RuntimeError(f'USB serial {serial}: expected one CAN interface, found {matches}')
    return matches[0]


def main():
    from bimanual_vla.deployment import client
    from bimanual_vla.deployment.lingbot_piper import DEFAULT_TASK
    args = sys.argv[1:]
    forbidden = ('--rtc-enabled', '--async-inference', '--preresize-policy-images',
                 '--arm-mode', '--arm-side', '--output-mode', '--policy-schema')
    if any(a.split('=')[0] in forbidden for a in args):
        raise SystemExit('This entry point requires bimanual joint control, synchronous inference, no RTC and padded256 images.')
    defaults = ['--host', '192.168.101.9', '--port', '8101', '--instruction', DEFAULT_TASK,
                '--sync-wait-steps', '25']
    if '--help' not in args and '-h' not in args:
        prefs_path = Path.home() / '.config/bimanual-vla/collect_gui_preferences.json'
        if prefs_path.exists():
            prefs = json.loads(prefs_path.read_text())
            for side in ('left', 'right'):
                flag = f'--cam-{side}-wrist-device'
                device = prefs.get(f'{side}_wrist_device')
                if device and not any(a.split('=')[0] == flag for a in args):
                    if not Path(device).exists():
                        raise RuntimeError(f'Saved {side} wrist camera is unavailable: {device}')
                    defaults += [flag, device]

        for side, serial in SERIALS.items():
            flag = f'--{side}-can'
            if not any(a.split('=')[0] == flag for a in args):
                interface = resolve_can(serial)
                print(f'Piper {side}: {interface}, USB serial={serial}', flush=True)
                defaults += [flag, interface]
    sys.argv = [sys.argv[0], *defaults, *args, '--arm-mode', 'bimanual', '--arm-side', 'both',
                '--output-mode', 'joint', '--no-rtc-enabled', '--no-async-inference',
                '--no-preresize-policy-images']
    validate = client.validate_policy_metadata
    def validate_piper(server_metadata, *a, **kw):
        if server_metadata.get('policy_family') != 'lingbot_piper' or server_metadata.get('rtc_supported') is not False:
            raise RuntimeError('Expected the dedicated LingBot Piper synchronous server')
        return validate(server_metadata, *a, **kw)
    client.validate_policy_metadata = validate_piper
    try:
        client.main()
    finally:
        client.validate_policy_metadata = validate


if __name__ == '__main__':
    main()
