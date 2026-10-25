from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from bimanual_vla.collection.can_identity import (list_can_adapters, resolve_can_pair, resolve_can_selector,
                                                save_can_bindings, load_can_bindings)


class CANIdentityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.net = self.root/'net'; self.net.mkdir()

    def adapter(self, name, serial, port):
        usb = self.root/'usb'/port; usb.mkdir(parents=True, exist_ok=True)
        (usb/'idVendor').write_text('1d50')
        if serial is not None:
            (usb/'serial').write_text(serial)
        dev = usb/(port+':1.0'); dev.mkdir(exist_ok=True)
        net = self.net/name; net.mkdir()
        (net/'type').write_text('280')
        (net/'device').symlink_to(dev)

    def test_serial_survives_interface_and_usb_port_changes(self):
        self.adapter('can0', 'LEFT', '3-1')
        self.adapter('can1', 'RIGHT', '3-2')
        self.assertEqual(resolve_can_pair('serial:LEFT', 'serial:RIGHT', sysfs_root=self.net), ('can0','can1'))
        (self.net/'can0').rename(self.net/'tmp')
        (self.net/'can1').rename(self.net/'can0')
        (self.net/'tmp').rename(self.net/'can1')
        self.assertEqual(resolve_can_pair('serial:LEFT', 'serial:RIGHT', sysfs_root=self.net), ('can1','can0'))
        (self.root/'usb'/'3-1').rename(self.root/'usb'/'9-4')
        (self.net/'can1'/'device').unlink()
        (self.net/'can1'/'device').symlink_to(self.root/'usb'/'9-4'/'3-1:1.0')
        self.assertEqual(resolve_can_selector('serial:LEFT', sysfs_root=self.net), 'can1')

    def test_missing_or_duplicate_serial_never_falls_back(self):
        self.adapter('can0', 'A', '3-1')
        with self.assertRaisesRegex(ValueError, 'found 0'):
            resolve_can_selector('serial:B', sysfs_root=self.net)
        self.adapter('can1', 'A', '3-2')
        with self.assertRaisesRegex(ValueError, 'found 2'):
            resolve_can_selector('serial:A', sysfs_root=self.net)

    def test_mixed_names_cannot_select_same_arm_twice(self):
        self.adapter('can0', 'A', '3-1')
        with self.assertRaisesRegex(ValueError, 'same physical'):
            resolve_can_pair('serial:A', 'can0', sysfs_root=self.net)

    def test_missing_device_serial_does_not_use_hub_serial(self):
        self.adapter('can0', None, '3-1')
        (self.root/'usb'/'idVendor').write_text('1234')
        (self.root/'usb'/'serial').write_text('HUB')
        self.assertEqual(list_can_adapters(self.net)[0]['serial'], '')
        with self.assertRaises(ValueError):
            resolve_can_selector('serial:HUB', sysfs_root=self.net)

    def test_invalid_selectors_are_rejected(self):
        for selector in ['serial:', 'serial:../foo', 'serial:A B', '../can0', 'can0;reboot']:
            with self.subTest(selector=selector), self.assertRaises(ValueError):
                resolve_can_selector(selector, sysfs_root=self.net)

    def test_binding_file_only_persists_hardware_selectors(self):
        path = self.root/'settings'/'can.json'
        save_can_bindings(dict(left_can_selector='serial:LEFT', right_can_selector='serial:RIGHT',
                               unrelated='not hardware'), path)
        self.assertEqual(load_can_bindings(path), dict(left_can_selector='serial:LEFT', right_can_selector='serial:RIGHT'))
        self.assertNotIn('unrelated', path.read_text())
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)


if __name__ == '__main__':
    unittest.main()
