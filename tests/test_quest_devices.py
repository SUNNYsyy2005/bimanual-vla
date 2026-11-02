from types import SimpleNamespace
from unittest.mock import Mock, patch
import subprocess

import pytest

from bimanual_vla.collection.quest.devices import ADBDevice, discover_devices, select_serial


def test_device_listing_parses_usb_wifi_and_unusable_devices():
    output = '''List of devices attached
abc device usb:6-1.3 product:eureka model:Quest_3 device:eureka
192.168.1.2:5555 device model:Quest_2
blocked unauthorized usb:1-2
gone offline
denied no permissions (user in plugdev group)
'''
    with patch('shutil.which', return_value='/bin/adb'), \
            patch('subprocess.run', return_value=SimpleNamespace(stdout=output)) as run:
        devices = discover_devices()
    assert devices == [ADBDevice('abc', 'device', 'Quest 3'),
                       ADBDevice('192.168.1.2:5555', 'device', 'Quest 2'),
                       ADBDevice('blocked', 'unauthorized'), ADBDevice('gone', 'offline'),
                       ADBDevice('denied', 'no permissions')]
    assert run.call_args.args[0] == ['adb', 'devices', '-l']
    assert run.call_args.kwargs['timeout'] == 5


@pytest.mark.parametrize('error', [subprocess.TimeoutExpired('adb', 5),
                                 subprocess.CalledProcessError(1, 'adb'), OSError('missing')])
def test_discovery_failures_are_actionable(error):
    with patch('shutil.which', return_value='/bin/adb'), patch('subprocess.run', side_effect=error):
        with pytest.raises(RuntimeError, match='ADB'):
            discover_devices()


def test_missing_adb_does_not_run_commands():
    with patch('shutil.which', return_value=None), patch('subprocess.run') as run:
        with pytest.raises(RuntimeError, match='adb not found'):
            discover_devices()
        run.assert_not_called()


def test_single_authorized_device_selected_and_explicit_choice_preserved():
    devices = [ADBDevice('a', 'device'), ADBDevice('b', 'unauthorized')]
    assert select_serial(devices) == 'a'
    devices.append(ADBDevice('wifi:5555', 'device'))
    assert select_serial(devices, ' wifi:5555 ') == 'wifi:5555'


@pytest.mark.parametrize('devices,requested,reason', [
    ([], '', 'No ADB devices'),
    ([ADBDevice('a', 'device'), ADBDevice('b', 'device')], '', 'Multiple ADB devices'),
    ([ADBDevice('a', 'unauthorized')], '', 'allow USB debugging'),
    ([ADBDevice('a', 'offline')], 'a', 'offline'),
    ([ADBDevice('a', 'no permissions')], '', 'permissions'),
    ([ADBDevice('a', 'device')], 'missing', 'Selected device'),
    ([ADBDevice('a', 'device'), ADBDevice('b', 'unauthorized')], 'b', 'allow USB debugging'),
])
def test_ambiguous_or_unavailable_selection_does_not_fall_back(devices, requested, reason):
    with pytest.raises(RuntimeError, match=reason):
        select_serial(devices, requested)


def test_reader_pins_discovered_device_for_all_adb_commands():
    from bimanual_vla.collection.quest.input import QuestReader
    with patch('bimanual_vla.collection.quest.devices.discover_devices',
               return_value=[ADBDevice('vr', 'device')]), \
            patch('bimanual_vla.collection.quest.input.time.time', return_value=100.), \
            patch('subprocess.check_output', return_value='100.0') as probe, \
            patch('bimanual_vla.collection.quest.devices.launch_teleop_app') as launch, \
            patch('subprocess.Popen') as stream, patch('threading.Thread'):
        reader = QuestReader()
        reader.start()
        assert reader.serial == 'vr'
        launch.assert_called_once_with('vr')
        assert probe.call_args.args[0][:3] == ['adb', '-s', 'vr']
        assert stream.call_args.args[0][:3] == ['adb', '-s', 'vr']


@pytest.mark.parametrize('output', [
    'Starting: Intent { cmp=com.rail.oculus.teleop/.MainActivity }',
    'Warning: Activity not started, intent has been delivered to currently running top-most instance.',
])
def test_app_launch_uses_selected_device_and_never_force_stops(output):
    from bimanual_vla.collection.quest.devices import launch_teleop_app
    with patch('subprocess.run', return_value=SimpleNamespace(stdout=output, stderr='')) as run:
        launch_teleop_app('vr')
    assert run.call_args.args[0] == ['adb', '-s', 'vr', 'shell', 'am', 'start', '-n',
                                    'com.rail.oculus.teleop/.MainActivity']
    assert run.call_args.kwargs['timeout'] == 5


@pytest.mark.parametrize('output', ['Error type 3\nActivity class does not exist.',
                                  'Status: timeout', 'SecurityException: permission denied'])
def test_android_launch_failure_even_with_zero_exit_code_is_rejected(output):
    from bimanual_vla.collection.quest.devices import launch_teleop_app
    with patch('subprocess.run', return_value=SimpleNamespace(stdout=output, stderr='')):
        with pytest.raises(RuntimeError, match='launch failed'):
            launch_teleop_app('vr')


def test_reader_does_not_start_stream_after_app_launch_failure():
    from bimanual_vla.collection.quest.input import QuestReader
    with patch('bimanual_vla.collection.quest.devices.discover_devices',
               return_value=[ADBDevice('vr', 'device')]), \
            patch('bimanual_vla.collection.quest.devices.launch_teleop_app', side_effect=RuntimeError('APK missing')), \
            patch('subprocess.Popen') as stream:
        with pytest.raises(RuntimeError, match='APK missing'):
            QuestReader().start()
        stream.assert_not_called()


def test_readonly_reader_can_opt_out_of_launch():
    from bimanual_vla.collection.quest.input import QuestReader
    with patch('bimanual_vla.collection.quest.devices.discover_devices',
               return_value=[ADBDevice('vr', 'device')]), \
            patch('bimanual_vla.collection.quest.devices.launch_teleop_app') as launch, \
            patch('bimanual_vla.collection.quest.input.time.time', return_value=100.), \
            patch('subprocess.check_output', return_value='100.0'), \
            patch('subprocess.Popen'), patch('threading.Thread'):
        QuestReader(launch_app=False).start()
        launch.assert_not_called()


@pytest.mark.parametrize('requested,connected,expected', [('', False, 'a'),
                                                        ('b', False, 'b'), ('', True, None)])
def test_gui_scan_preserves_selection_and_does_not_change_active_connection(requested, connected, expected):
    from bimanual_vla.collection.gui import CollectorGUI
    ui = SimpleNamespace(_quest_scan_busy=True, quest_refresh_button=Mock(),
        quest_serial_entry=Mock(), quest_device_status_var=Mock(),
        quest_adb_serial_var=Mock(get=lambda: requested), piper=object() if connected else None,
        _inference_running=lambda: False, _replay_running=lambda: False)
    devices = [ADBDevice('a', 'device', 'Quest 3')]
    if requested:
        devices.append(ADBDevice('b', 'device', 'Quest 2'))
    CollectorGUI._finish_quest_devices(ui, devices, None)
    assert not ui._quest_scan_busy
    if expected is None:
        ui.quest_adb_serial_var.set.assert_not_called()
    else:
        ui.quest_adb_serial_var.set.assert_called_once_with(expected)


def test_gui_multi_device_scan_requests_choice():
    from bimanual_vla.collection.gui import CollectorGUI
    ui = SimpleNamespace(_quest_scan_busy=True, quest_refresh_button=Mock(),
        quest_serial_entry=Mock(), quest_device_status_var=Mock(),
        quest_adb_serial_var=Mock(get=lambda: ''))
    CollectorGUI._finish_quest_devices(ui, [ADBDevice('a', 'device'), ADBDevice('b', 'device')], None)
    ui.quest_adb_serial_var.set.assert_not_called()
    assert 'Multiple ADB devices' in ui.quest_device_status_var.set.call_args.args[0]
