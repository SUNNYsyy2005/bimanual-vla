"""Small stale-while-refresh snapshots for list endpoints, never log payloads."""
import copy
import json
from pathlib import Path
import threading
import time
import uuid


class Snapshot:
    def __init__(self,path,builder,ttl=30):
        self.path=Path(path);self.builder=builder;self.ttl=ttl
        self.lock=threading.Lock();self.running=False;self.attempt=0;self.error=None
        self.data=None;self.updated=0;self.generation=0
        try:
            if self.path.stat().st_size<=64*1024*1024:
                saved=json.loads(self.path.read_text(encoding='utf-8'))
                self.data=saved['data'];self.updated=float(saved['updated_at'])
        except (OSError,ValueError,KeyError,TypeError):pass

    def invalidate(self):
        with self.lock:self.attempt=0;self.updated=0;self.generation+=1

    def _refresh(self,generation):
        try:
            data=self.builder();updated=time.time()
            with self.lock:
                if generation!=self.generation:return
                self.path.parent.mkdir(parents=True,exist_ok=True)
                temporary=self.path.with_name('.'+self.path.name+'.'+uuid.uuid4().hex)
                temporary.write_text(json.dumps(dict(data=data,updated_at=updated)),encoding='utf-8');temporary.replace(self.path)
                self.data=data;self.updated=updated;self.error=None
        except Exception as exc:
            with self.lock:self.error=str(exc)[:1000]
        finally:
            with self.lock:self.running=False

    def get(self):
        with self.lock:
            now=time.time()
            if not self.running and now-self.updated>=self.ttl and now-self.attempt>=min(5,self.ttl):
                self.running=True;self.attempt=now
                threading.Thread(target=self._refresh,args=(self.generation,),daemon=True,name='dashboard-list-refresh').start()
            return dict(data=copy.deepcopy(self.data),updated_at=self.updated,refreshing=self.running,error=self.error,ready=self.data is not None)

    def find(self, ident):
        """Read only one cached record; selected views never rebuild the global list."""
        with self.lock:
            return copy.deepcopy(next((row for row in self.data or [] if row.get('id') == ident), None))
