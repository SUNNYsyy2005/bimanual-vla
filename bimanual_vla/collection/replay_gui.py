"""Tk controls; each hardware replay owns a fresh SDK subprocess."""
from dataclasses import asdict, replace
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk
import uuid

from bimanual_vla.collection.replay import ReplayBlocked, ReplayConfig, load_replay_episode
from bimanual_vla.deployment.execution_defaults import DEFAULT_SERVO_HZ


def run_replay_worker(path, config, selectors, config_path, stop, progress):
    """Keep the SDK's per-CAN singleton out of the GUI and previous attempts."""
    config_path = Path(config_path)
    config_path.write_text(json.dumps(asdict(config), indent=2) + '\n', encoding='utf-8')
    command = [sys.executable, '-m', 'bimanual_vla.collection.replay', str(path),
               '--config-file', str(config_path),
               '--left-can', selectors.get('left', 'can0'),
               '--right-can', selectors.get('right', 'can1'),
               '--parent-pid', str(os.getpid()), '--allow-execution', '--exclusive-control']
    if stop.is_set():
        progress({'state': 'stopped', 'stop_errors': []})
        return
    process = subprocess.Popen(command, cwd=str(Path(__file__).resolve().parents[2]),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    finished = threading.Event()

    def request_stop():
        if process.poll() is None:
            try:
                # The CLI handler sets its stop event; never SIGKILL a writer.
                process.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass

    def watch_stop():
        while not finished.wait(.02):
            if stop.is_set():
                request_stop()
                return

    watcher = threading.Thread(target=watch_stop, name='replay-stop-watcher', daemon=True)
    watcher.start()
    failure, terminal = None, None
    recent_output = []
    try:
        progress({'state': 'worker_started', 'pid': process.pid})
        assert process.stdout is not None
        for line in process.stdout:
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or 'state' not in event:
                    raise ValueError('not a replay event')
            except ValueError:
                recent_output = (recent_output + [line.rstrip()])[-5:]
                progress({'state': 'worker_log', 'line': line.rstrip()})
                continue
            if event['state'] == 'failed':
                failure = event
                continue
            if event['state'] in ('stopped', 'completed'):
                terminal = event['state']
            progress(event)
        code = process.wait()
        if failure is not None:
            # Preserve the child's traceback in the log without reducing it to
            # the stale SDK cache message seen on earlier GUI retries.
            progress({'state': 'worker_failure_details', **{k: v for k, v in failure.items() if k != 'state'}})
            raise ReplayBlocked(failure['error'])
        if stop.is_set() and terminal is None:
            raise ReplayBlocked('Replay worker exited without confirming position hold; check the robot state')
        elif code != 0 or terminal is None:
            raise ReplayBlocked(f'Replay worker exited with code {code}: ' + '\n'.join(recent_output))
    finally:
        finished.set()
        request_stop()
        # Keep the GUI run active until the writer exits; never leave an
        # orphaned controller or allow a second run after a reader/log error.
        process.wait()
        watcher.join(timeout=.2)
        if process.stdout is not None:
            process.stdout.close()


class ReplayDialog:
    def __init__(self, parent, path, *, preview, selectors, blocking_reason,
                 activity_changed=lambda active: None, defaults=None, smoothed_hz=DEFAULT_SERVO_HZ):
        defaults = defaults or ReplayConfig()
        self.defaults = defaults
        self.smoothed_hz = smoothed_hz
        self.path, self.selectors, self.blocking_reason = Path(path), selectors, blocking_reason
        self.worker = None
        self.stop_event = threading.Event()
        self.events = queue.Queue()
        self.activity_changed = activity_changed
        self.activity_reported = False
        self.window = tk.Toplevel(parent)
        self.window.title(f'Replay episode — {self.path.name}')
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.window.columnconfigure(0, weight=1)
        self.form = ttk.Frame(self.window, padding=14)
        self.form.grid(sticky='nsew')
        self.form.columnconfigure(1, weight=1)
        self.episode = None
        try:
            self.episode = load_replay_episode(path)
            description = (f'{self.path.name} | {self.episode.arm_mode} / {", ".join(self.episode.sides)}\n'
                           f'Recorded representations: {", ".join(self.episode.formats)}')
        except Exception as exc:
            description = f'Hardware replay unavailable: {exc}\nVideo preview is still available.'
        ttk.Label(self.form, text=description, wraplength=620).grid(row=0, column=0, columnspan=3, sticky='w')
        ttk.Button(self.form, text='Video preview', command=preview).grid(row=1, column=0, sticky='w', pady=8)
        initial_mode = ('joint' if not self.episode or 'joint' in self.episode.formats
                        else self.episode.formats[0])
        self.mode = tk.StringVar(self.window, value=initial_mode)
        ttk.Label(self.form, text='Command representation').grid(row=2, column=0, sticky='w')
        self.mode_box = ttk.Combobox(self.form, textvariable=self.mode, state='readonly',
                                     values=self.episode.formats if self.episode else ())
        self.mode_box.grid(row=2, column=1, sticky='ew')
        self.joint_smoothing = tk.BooleanVar(self.window, value=defaults.joint_smoothing)
        self.eef_smoothing = tk.BooleanVar(self.window, value=defaults.eef_smoothing)
        self.joint_check = ttk.Checkbutton(self.form, text='Joint smoothing (Ruckig)', variable=self.joint_smoothing)
        self.joint_check.grid(row=3, column=0, columnspan=2, sticky='w', pady=(8, 0))
        self.eef_check = ttk.Checkbutton(self.form, text='End-effector smoothing (translation / rotation)',
                                         variable=self.eef_smoothing)
        self.eef_check.grid(row=4, column=0, columnspan=2, sticky='w')
        ttk.Label(self.form, text='EEF mode: recorded base-frame poses → optional pose smoothing → IK → optional joint smoothing.\n'
                  'With joint smoothing off, IK steps remain speed-limited; both arms and grippers wait together.\n'
                  'EEF smoothing alone does not bound joint speed. Feedback checks remain active.', wraplength=620).grid(
                      row=5, column=0, columnspan=3, sticky='w', pady=(3, 8))
        tabs = ttk.Notebook(self.form)
        tabs.grid(row=6, column=0, columnspan=3, sticky='ew')
        general, pose = ttk.Frame(tabs, padding=8), ttk.Frame(tabs, padding=8)
        tabs.add(general, text='Replay / joints')
        tabs.add(pose, text='End-effector smoothing')
        self.settings, self.entries = {}, []
        general_fields = (
            ('speed', 'Playback speed (0.05–1×)'), ('control_hz', 'Reference update rate (Hz)'),
            ('hz', 'Command publication rate (Hz)'),
            ('speed_pct', 'Firmware speed (%)'), ('max_joint_speed', 'Joint speed limit (rad/s)'),
            ('max_joint_acceleration', 'Joint acceleration (rad/s²)'), ('max_joint_jerk', 'Joint jerk (rad/s³)'),
            ('max_tracking_error', 'Joint tracking limit (rad)'), ('max_gripper_speed', 'Gripper speed (m/s)'),
            ('approach_joint_speed', 'First-pose joint speed (rad/s)'),
            ('approach_timeout', 'First-pose timeout (s)'),
            ('feedback_max_age', 'Maximum feedback age (s)'),
        )
        for row, (name, label) in enumerate(general_fields):
            ttk.Label(general, text=label).grid(row=row, column=0, sticky='w', padx=(0, 10))
            variable = tk.StringVar(self.window, value=str(getattr(defaults, name)))
            entry = ttk.Entry(general, textvariable=variable, width=12)
            entry.grid(row=row, column=1, sticky='e')
            self.settings[name] = variable
            self.entries.append(entry)
        self.enable_motors = tk.BooleanVar(self.window, value=True)
        self.enable_check = ttk.Checkbutton(general, variable=self.enable_motors,
            text='Enable motors while holding current pose (master must be isolated)')
        self.enable_check.grid(row=len(general_fields), column=0, columnspan=2, sticky='w', pady=(8, 0))
        self.auto_recover = tk.BooleanVar(self.window, value=defaults.auto_recover)
        self.recover_check = ttk.Checkbutton(general, variable=self.auto_recover,
            text='Automatically clear eligible cooled, disabled driver faults once before replay')
        self.recover_check.grid(row=len(general_fields) + 1, column=0, columnspan=2, sticky='w')
        self.pose_settings, self.pose_entries = {}, []
        for row, (name, label) in enumerate((
            ('max_linear_speed_m_s', 'Linear speed (m/s)'),
            ('max_linear_acceleration_m_s2', 'Linear acceleration (m/s²)'),
            ('max_angular_speed_rad_s', 'Angular speed (rad/s)'),
            ('max_angular_acceleration_rad_s2', 'Angular acceleration (rad/s²)'),
            ('response_time_s', 'Response time (s)'),
        )):
            ttk.Label(pose, text=label).grid(row=row, column=0, sticky='w', padx=(0, 10))
            variable = tk.StringVar(self.window, value=str(getattr(defaults.pose, name)))
            entry = ttk.Entry(pose, textvariable=variable, width=12)
            entry.grid(row=row, column=1, sticky='e')
            self.pose_settings[name] = variable
            self.pose_entries.append(entry)
        self.start_pose = tk.StringVar(self.window)
        ttk.Label(self.form, textvariable=self.start_pose, wraplength=620, justify='left').grid(
            row=7, column=0, columnspan=3, sticky='w', pady=8)
        self.exclusive = tk.BooleanVar(self.window, value=False)
        self.exclusive_check = ttk.Checkbutton(self.form, variable=self.exclusive,
            text='Master control is isolated; control is exclusive and the movement area is clear')
        self.exclusive_check.grid(row=8, column=0, columnspan=3, sticky='w')
        ttk.Label(self.form, text='When selected, enable holds the measured pose before moving smoothly to frame 0.\n'
                  'Stop / failure requests a hold at measured joints and leaves motors enabled.\n'
                  'Automatic fault clearing is limited to eligible startup thermal latches.\n'
                  'Emergency stops and motion failures require intervention; no automatic reset or restart.', wraplength=620).grid(
                      row=9, column=0, columnspan=3, sticky='w', pady=8)
        self.start_button = ttk.Button(self.form, text='Start hardware replay', command=self.start)
        self.start_button.grid(row=10, column=0, sticky='ew', padx=(0, 8))
        self.stop_button = ttk.Button(self.form, text='Stop replay (hold)', command=self.stop, state='disabled')
        self.stop_button.grid(row=10, column=1, sticky='ew')
        self.progress = ttk.Progressbar(self.form, maximum=100)
        self.progress.grid(row=11, column=0, columnspan=3, sticky='ew', pady=8)
        self.status = tk.StringVar(self.window, value='Ready — no hardware commands sent')
        ttk.Label(self.form, textvariable=self.status, wraplength=620).grid(row=12, column=0, columnspan=3, sticky='w')
        self.log_label = tk.StringVar(self.window)
        ttk.Label(self.form, textvariable=self.log_label, wraplength=620).grid(row=13, column=0, columnspan=3, sticky='w')
        self.mode.trace_add('write', lambda *_: self.refresh())
        self.joint_smoothing.trace_add('write', lambda *_: self.update_command_rate())
        self.eef_smoothing.trace_add('write', lambda *_: self.refresh())
        self.refresh()
        self.window.after(80, self.poll)

    @property
    def running(self):
        return self.worker is not None and self.worker.is_alive()

    def update_command_rate(self):
        if self.running:
            return
        try:
            control_hz = float(self.settings['control_hz'].get())
            hz = max(control_hz, self.smoothed_hz) if self.joint_smoothing.get() else control_hz
            self.settings['hz'].set(str(hz))
        except ValueError:
            # An entry may be temporarily empty while being edited.
            return

    def refresh(self):
        active = self.running
        for widget in [self.joint_check, self.exclusive_check, self.enable_check, self.recover_check, *self.entries]:
            widget.configure(state='disabled' if active else 'normal')
        self.mode_box.configure(state='disabled' if active else 'readonly')
        self.eef_check.configure(state='disabled' if active or self.mode.get() != 'eef' else 'normal')
        for entry in self.pose_entries:
            entry.configure(state='normal' if not active and self.mode.get() == 'eef' and self.eef_smoothing.get() else 'disabled')
        self.start_button.configure(state='normal' if self.episode and not active else 'disabled')
        self.stop_button.configure(state='normal' if active else 'disabled')
        if not self.episode:
            return
        mode = self.mode.get()
        width = 7 if mode == 'joint' else 10
        text = []
        import numpy as np
        for i, side in enumerate(self.episode.sides):
            first = self.episode.targets[mode][0, i * width:(i + 1) * width]
            if mode == 'joint':
                pose = 'joints (deg) ' + ', '.join(f'{v:.2f}' for v in np.rad2deg(first[:6]))
            else:
                pose = 'xyz (m) ' + ', '.join(f'{v:.4f}' for v in first[:3])
                from scipy.spatial.transform import Rotation
                from bimanual_vla.data.contract import rotation_matrix_from_state
                rotation = Rotation.from_matrix(rotation_matrix_from_state(first)).as_euler('xyz', degrees=True)
                pose += '; rpy (deg) ' + ', '.join(f'{v:.2f}' for v in rotation)
            text.append(f'{side} frame 0: {pose}; gripper {first[-1] * 1000:.2f} mm')
        self.start_pose.set('\n'.join(text))

    def start(self):
        if self.running or self.episode is None:
            return
        try:
            reason = self.blocking_reason()
            if reason:
                raise ValueError(reason)
            if not self.exclusive.get():
                raise ValueError('Isolate master control and confirm exclusive control before replay')
            values = {name: float(value.get()) for name, value in self.settings.items()}
            if not values['speed_pct'].is_integer():
                raise ValueError('Firmware speed must be an integer')
            values['speed_pct'] = int(values['speed_pct'])
            config = replace(self.defaults, mode=self.mode.get(), joint_smoothing=self.joint_smoothing.get(),
                eef_smoothing=self.eef_smoothing.get(), enable_before_replay=self.enable_motors.get(),
                auto_recover=self.auto_recover.get(), **values,
                pose=replace(self.defaults.pose, **{k: float(v.get()) for k, v in self.pose_settings.items()}))
            selectors = self.selectors(self.episode)
            log_root = Path('~/.local/state/bimanual-vla/replays').expanduser()
            log_root.mkdir(parents=True, exist_ok=True)
            log_path = log_root / (datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8] + '.jsonl')
            # Open the log before the worker can connect to or command hardware.
            log = log_path.open('x', encoding='utf-8', buffering=1)
        except Exception as exc:
            messagebox.showerror('Cannot replay', str(exc), parent=self.window)
            return
        self.log_label.set(f'Log: {log_path}')
        self.stop_event.clear()
        self.status.set('Checking hardware — no motion until preflight passes')
        self.progress['value'] = 0

        def emit(event):
            log.write(json.dumps({'time': datetime.now().isoformat(), **event}) + '\n')
            self.events.put(event)

        def work():
            try:
                emit({'state': 'configured', 'episode': str(self.path), 'selectors': selectors,
                      'config': asdict(config)})
                run_replay_worker(self.path, config, selectors, log_path.with_suffix('.config.json'),
                                  self.stop_event, emit)
            except Exception as exc:
                try:
                    emit({'state': 'failed', 'error': str(exc)})
                except Exception:
                    self.events.put({'state': 'failed', 'error': str(exc)})
            finally:
                log.close()
                self.events.put({'state': 'worker_done'})

        self.worker = threading.Thread(target=work, name='episode-hardware-replay', daemon=False)
        self.worker.start()
        self.activity_changed(True)
        self.activity_reported = True
        self.refresh()

    def poll(self):
        if not self.window.winfo_exists():
            return
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            state = event['state']
            if state == 'worker_done':
                self.refresh()
                continue
            if state in ('worker_log', 'worker_failure_details'):
                continue
            if 'frames' in event:
                self.progress['value'] = 100 * (event['index'] + 1) / event['frames']
            if state == 'failed':
                self.status.set('Replay blocked / failed: ' + event['error'])
            elif state == 'stopped':
                errors = event.get('stop_errors')
                self.status.set('Replay stopped; no disable/reset sent. ' + (
                    'Position hold not confirmed: ' + str(errors) if errors else
                    'Position hold requested; motor enable unchanged.'))
            elif state == 'holding_after_failure':
                self.status.set('Replay failed; position hold requested, no disable/reset sent. '
                                + str(event.get('stop_errors') or ''))
            elif state == 'completed':
                self.status.set('Replay completed and final pose verified; no return or reset sent')
            elif state == 'running':
                message = f'Running: frame {event["index"] + 1}/{event["frames"]}'
                if any(m['limited'] for m in event.get('ik_rate_limit', {}).values()):
                    message += ' — IK step limited (same online control path as inference)'
                self.status.set(message)
            elif state == 'approaching':
                self.status.set(f'Moving smoothly to first-frame pose: {event["elapsed_s"]:.1f}s — episode clock not started')
            elif state == 'first_pose_reached':
                self.status.set('First-frame pose reached and stable; starting episode replay')
            elif state == 'waiting_feedback':
                self.status.set('Waiting for fresh status, joint, gripper and six-motor feedback…')
            elif state == 'recovery_checking':
                self.status.set('Checking startup state for eligible automatic recovery…')
            elif state == 'recovery_not_needed':
                self.status.set('No driver fault recovery needed; continuing preflight')
            elif state == 'recovery_waiting_stationary':
                self.status.set('Checking disabled arms are stationary before clearing cooled thermal latches')
            elif state == 'recovering_driver_fault':
                self.status.set(f'Clearing eligible driver fault once: {event["side"]} J{event["joint"]}')
            elif state == 'recovery_completed':
                self.status.set('Driver faults cleared and healthy stationary feedback verified; continuing preflight')
            elif state == 'enabling_hold':
                self.status.set('Enabling motors while holding current pose — episode clock not started')
            elif state == 'enabled_hold_ready':
                self.status.set('Motor enable and CAN joint mode confirmed; preparing first-frame approach')
            elif state == 'aligning_start_boundary':
                self.status.set('Moving slightly negative initial J2 to its lower limit at 1% speed')
            else:
                self.status.set(state)
        if self.activity_reported and not self.running:
            self.activity_changed(False)
            self.activity_reported = False
        self.refresh()
        self.window.after(80, self.poll)

    def stop(self):
        self.stop_event.set()
        self.status.set('Stopping replay…')

    def close(self):
        if self.running:
            self.stop()
            return
        self.window.destroy()
