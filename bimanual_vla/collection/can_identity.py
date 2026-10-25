"""Resolve stable USB-CAN serial selectors without reconfiguring interfaces."""
from pathlib import Path
import json
import os
import re

NET_ROOT = Path('/sys/class/net')
BINDINGS_PATH = Path('~/.config/bimanual-vla/can_bindings.json').expanduser()
BINDING_KEYS = ('can_selector', 'left_can_selector', 'right_can_selector')


def load_can_bindings(path: Path = BINDINGS_PATH) -> dict[str, str]:
    try:
        data = json.loads(Path(path).read_text())
        return {k: validate_can_selector(data[k]) for k in BINDING_KEYS if k in data}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def save_can_bindings(values: dict, path: Path = BINDINGS_PATH) -> None:
    data = {k: validate_can_selector(values[k]) for k in BINDING_KEYS if k in values}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def validate_can_selector(value: str) -> str:
    value = value.strip()
    if value.startswith('serial:'):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', value[7:]):
            raise ValueError('CAN serial selector must be serial:<USB serial number>')
    elif not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,14}', value):
        raise ValueError(f'invalid CAN selector: {value!r}')
    return value


def list_can_adapters(sysfs_root: Path = NET_ROOT) -> list[dict[str, str]]:
    adapters = []
    for interface in sorted(Path(sysfs_root).iterdir()):
        try:
            if (interface/'type').read_text().strip() != '280':
                continue
            device = (interface/'device').resolve(strict=True)
            serial = ''
            # Stop at the nearest USB device. Never borrow a hub's serial.
            for parent in (device, *device.parents):
                if (parent/'idVendor').is_file():
                    if (parent/'serial').is_file():
                        serial = (parent/'serial').read_text().strip()
                    break
            adapters.append(dict(interface=interface.name, serial=serial,
                                 device_path=str(device)))
        except (OSError, RuntimeError):
            continue  # Unplugged during discovery, or a non-hardware interface.
    return adapters


def resolve_can_selector(value: str, *, sysfs_root: Path = NET_ROOT) -> str:
    selector = validate_can_selector(value)
    if not selector.startswith('serial:'):
        return selector  # Legacy explicit interface; operational checks follow.
    serial = selector[7:]
    matches = [r['interface'] for r in list_can_adapters(sysfs_root) if r['serial'] == serial]
    if len(matches) != 1:
        raise ValueError(f'CAN serial {serial}: expected one interface, found {len(matches)}; '
                         'check adapter connection/serial (no interface-name fallback)')
    return matches[0]


def resolve_can_pair(left: str, right: str, *, sysfs_root: Path = NET_ROOT) -> tuple[str, str]:
    names = tuple(resolve_can_selector(s, sysfs_root=sysfs_root) for s in (left, right))
    if names[0] == names[1]:
        raise ValueError('Left and right CAN selectors resolve to the same physical interface')
    return names
