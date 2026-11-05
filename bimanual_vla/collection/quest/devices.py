"""Bounded ADB discovery shared by the GUI and Quest readers."""
from dataclasses import dataclass
import os
import shutil
import subprocess
import time


@dataclass(frozen=True)
class ADBDevice:
    serial: str
    state: str
    model: str = ''


def adb_executable() -> str:
    """Return the ADB binary used by every Quest operation.

    The GUI is often launched from a shell that has the Conda environment on
    PATH, while a deployment command may invoke the environment's Python by
    absolute path. Resolve the tool explicitly so both launch modes use the
    same ADB server and platform-tools installation.
    """
    configured = os.environ.get("BIMANUAL_VLA_ADB", "").strip()
    if configured:
        path = os.path.expanduser(configured)
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            raise RuntimeError(f"configured ADB executable is not runnable: {path}")
        return path
    found = shutil.which("adb")
    if found:
        # Keep the command name stable for callers/tests and let the shell
        # resolve the same ADB server selected by the user's environment.
        return "adb"
    raise RuntimeError(
        "adb not found; add Android platform-tools to PATH or set BIMANUAL_VLA_ADB"
    )


def launch_teleop_app(serial):
    """Bring the installed official app forward without resetting it or the HMD."""
    if not serial.strip():
        raise ValueError('launch requires an explicitly selected ADB device')
    component = 'com.rail.oculus.teleop/.MainActivity'
    try:
        result = subprocess.run(
            [os.environ.get("BIMANUAL_VLA_ADB", "adb"), '-s', serial,
             'shell', 'am', 'start', '-n', component],
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
    adb = adb_executable()
    try:
        result = subprocess.run([adb, 'devices', '-l'], capture_output=True,
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


def wait_for_serial(serial: str = "", *, timeout_s: float = 12.0, poll_s: float = 0.5) -> str:
    """Wait for the selected headset, or the sole authorized headset, to reappear."""
    requested = serial.strip()
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    last_error = ""
    while True:
        devices = discover_devices()
        try:
            return select_serial(devices, requested)
        except RuntimeError as exc:
            last_error = str(exc)
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Quest serial {requested} did not become ready within {timeout_s:.1f}s: {last_error}"
            )
        time.sleep(max(0.05, float(poll_s)))


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
