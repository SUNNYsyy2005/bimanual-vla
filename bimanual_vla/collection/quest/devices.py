"""Bounded ADB discovery shared by the GUI and Quest readers."""
from dataclasses import dataclass
import shutil
import subprocess


@dataclass(frozen=True)
class ADBDevice:
    serial: str
    state: str
    model: str = ''


def launch_teleop_app(serial):
    """Bring the installed official app forward without resetting it or the HMD."""
    if not serial.strip():
        raise ValueError('launch requires an explicitly selected ADB device')
    component = 'com.rail.oculus.teleop/.MainActivity'
    try:
        result = subprocess.run(
            ['adb', '-s', serial, 'shell', 'am', 'start', '-n', component],
            capture_output=True, text=True, timeout=5, check=True)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('VR app launch timed out; check the app and headset dialogs') from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError('Cannot launch VR app; check ADB authorization and APK installation') from exc
    # Android am can exit zero even when the package/activity does not exist.
    output = result.stdout + '\n' + result.stderr
    # Do not use -W: Quest system overlays can prevent ActivityManager's drawn
    # acknowledgement indefinitely. This only acknowledges the launch request;
    # the controller separately requires fresh poses/buttons and clutch release.
    if ('Error' in output or 'Exception' in output
            or not any(marker in output for marker in ('Starting: Intent', 'intent has been delivered'))):
        raise RuntimeError('VR app launch failed; install the official teleop APK: ' + output.strip()[:500])


def discover_devices():
    if not shutil.which('adb'):
        raise RuntimeError('adb not found; use an environment with Android Platform Tools installed')
    try:
        result = subprocess.run(['adb', 'devices', '-l'], capture_output=True,
                                text=True, timeout=5, check=True)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('ADB device scan timed out; check USB and refresh') from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError('ADB device scan failed; check adb and USB') from exc
    devices = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[0] == 'List' or fields[0].startswith('*'):
            continue
        state = 'no permissions' if fields[1:3] == ['no', 'permissions'] else fields[1]
        model = next((f.split(':', 1)[1] for f in fields[2:] if f.startswith('model:')), '')
        devices.append(ADBDevice(fields[0], state, model.replace('_', ' ')))
    return devices


def select_serial(devices, requested=''):
    requested = requested.strip()
    ready = [d for d in devices if d.state == 'device']
    if requested:
        matched = next((d for d in devices if d.serial == requested), None)
        if matched is None:
            raise RuntimeError(f'Selected device {requested} not found; refresh or select another device')
        if matched.state == 'device':
            return matched.serial
        devices = [matched]
    elif len(ready) == 1:
        return ready[0].serial
    elif len(ready) > 1:
        raise RuntimeError('Multiple ADB devices found; select the headset serial')
    if any(d.state == 'unauthorized' for d in devices):
        raise RuntimeError('VR unauthorized: allow USB debugging in the headset, then refresh')
    if any(d.state == 'no permissions' for d in devices):
        raise RuntimeError('ADB has no USB permissions; check device permissions')
    if devices:
        raise RuntimeError('Device not ready: '+', '.join(f'{d.serial} ({d.state})' for d in devices))
    raise RuntimeError('No ADB devices found; connect the headset and enable USB debugging')
