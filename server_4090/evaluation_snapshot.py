"""Run the reviewed evaluator source on Slurm without overwriting remote repos."""
import base64
import json
from pathlib import Path
import zlib


def snapshot_command(command, source_dir):
    source_dir=Path(source_dir)
    sources={name:(source_dir/(name+'.py')).read_text() for name in ('openpi_single_arm','selected_resource','eval_heldout_loss')}
    packed=base64.b64encode(zlib.compress(json.dumps(sources).encode())).decode()
    bootstrap='''import base64,json,sys,types,zlib
from pathlib import Path
sources=json.loads(zlib.decompress(base64.b64decode(PACKED)))
helper=HELPER
sys.path.insert(0,str(Path(helper).parent))
for name in ('openpi_single_arm','selected_resource'):
    module=types.ModuleType(name);module.__file__=str(Path(helper).with_name(name+'.py'));sys.modules[name]=module
    exec(compile(sources[name],module.__file__,'exec'),module.__dict__)
sys.argv[0]=helper
exec(compile(sources['eval_heldout_loss'],helper,'exec'),dict(__name__='__main__',__file__=helper))
'''.replace('HELPER',repr(command[1])).replace('PACKED',repr(packed))
    return [command[0],'-c',bootstrap,*command[2:]]
