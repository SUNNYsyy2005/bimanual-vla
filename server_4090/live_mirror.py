"""Bounded, persistent Dashboard copies of live metrics/logs; GET never waits on SSH."""
import copy
import hashlib
import json
import shutil
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor


class LiveMirror:
    def __init__(self, root, interval=3, workers=3):
        self.root=Path(root);self.interval=interval;self.lock=threading.Lock();self.entries={};self.blocked=set()
        self.pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='dashboard-live-sync')
        self.workers=workers;self.closed=threading.Event()
        threading.Thread(target=self._loop,daemon=True,name='dashboard-live-sync-clock').start()

    def _path(self,key):
        family=':'.join(key.split(':')[:2])
        return self.root/hashlib.sha256(family.encode()).hexdigest()/(hashlib.sha256(key.encode()).hexdigest()+'.json')

    def get(self,key,builder,seed=None):
        now=time.time()
        with self.lock:
            if ':'.join(key.split(':')[:2]) in self.blocked:raise FileNotFoundError('History deleted')
            if key not in self.entries:
                value=dict(data=seed or {},updated=0,error=None)
                try:
                    path=self._path(key)
                    if path.stat().st_size<=32*1024*1024:value=json.loads(path.read_text())
                except (OSError,ValueError):pass
                self.entries[key]={**value,'attempt':0,'running':False}
            row=self.entries[key];row.update(builder=builder,seen=now)
            result=copy.deepcopy(row['data'])
            result['sync']={'updated_at':row['updated'],'ready':bool(row['updated']),
                'refreshing':row['running'] or now-row['updated']>=self.interval,'error':row['error'],
                'interval_seconds':self.interval,'source':'dashboard_local_mirror'}
        self._tick()
        return result

    def _tick(self):
        with self.lock:
            now=time.time();slots=self.workers-sum(r['running'] for r in self.entries.values())
            due=sorted([(k,r) for k,r in self.entries.items() if not r['running'] and now-r['seen']<120
                and now-r['attempt']>=self.interval],key=lambda kv:kv[1]['attempt'])
            for key,row in due[:max(0,slots)]:
                row.update(running=True,attempt=now)
                self.pool.submit(self._refresh,key,row['builder'],row)
            for key,row in list(self.entries.items()):
                if not row['running'] and now-row['seen']>600:del self.entries[key]

    def forget(self,family):
        with self.lock:
            self.blocked.add(family)
            for key in list(self.entries):
                if key.startswith(family+':'):del self.entries[key]
            shutil.rmtree(self.root/hashlib.sha256(family.encode()).hexdigest(),ignore_errors=True)

    def _refresh(self,key,builder,entry):
        try:
            data=builder();updated=time.time()
            with self.lock:
                if self.entries.get(key) is not entry:return
                path=self._path(key);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
                temporary.write_text(json.dumps(dict(data=data,updated=updated,error=None)))
                temporary.replace(path)
                entry.update(data=data,updated=updated,error=None)
        except Exception as exc:
            with self.lock:
                if self.entries.get(key) is entry:entry['error']=str(exc)[:1000]
        finally:
            with self.lock:
                if self.entries.get(key) is entry:entry['running']=False

    def _loop(self):
        while not self.closed.wait(.5):self._tick()

    def close(self):
        self.closed.set();self.pool.shutdown(wait=True)
