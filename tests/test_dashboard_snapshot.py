import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from server_4090.dashboard_snapshot import Snapshot


class SnapshotTest(unittest.TestCase):
    def test_old_list_served_while_one_refresh_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'index.json';path.write_text(json.dumps(dict(data=['old'],updated_at=1)))
            started=threading.Event();release=threading.Event();calls=[]
            def build():calls.append(1);started.set();release.wait(3);return ['new']
            cache=Snapshot(path,build,ttl=30)
            try:
                before=time.monotonic();self.assertEqual(cache.get()['data'],['old']);self.assertLess(time.monotonic()-before,.1)
                self.assertTrue(started.wait(1))
                for _ in range(10):self.assertEqual(cache.get()['data'],['old'])
                self.assertEqual(len(calls),1)
            finally:release.set()
            self.wait_finished(cache)
            self.assertEqual(cache.get()['data'],['new'])

    def wait_finished(self,cache):
        for _ in range(100):
            with cache.lock:
                if not cache.running:return
            time.sleep(.01)
        self.fail('refresh did not finish')

    def test_invalidation_rejects_old_inflight_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            started=threading.Event();release=threading.Event()
            def build():started.set();release.wait(3);return ['stale']
            cache=Snapshot(Path(directory)/'index.json',build)
            cache.get();self.assertTrue(started.wait(1));cache.invalidate();release.set();self.wait_finished(cache)
            with cache.lock:self.assertIsNone(cache.data)

    def test_failure_preserves_last_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'index.json';path.write_text(json.dumps(dict(data=['old'],updated_at=1)))
            def fail():raise OSError('unavailable')
            cache=Snapshot(path,fail);cache.get();self.wait_finished(cache)
            result=cache.get();self.assertEqual(result['data'],['old']);self.assertIn('unavailable',result['error'])


if __name__=='__main__':unittest.main()
