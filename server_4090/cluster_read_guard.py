"""Read-only Slurm admission checks. Never allocate just to check availability."""
import shlex
import subprocess


def admit_read(target, prefix):
    host=target['submit_host'];node=target['node']
    def query(args):
        result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8',host,shlex.join(args)],capture_output=True,text=True,timeout=15)
        if result.returncode:raise RuntimeError('无法核对节点状态；未提交远端索引任务')
        return result.stdout
    states=query(['sinfo','-h','-N','-n',node,'-o','%T']).split()
    if not states or any(state.lower() not in {'idle','mixed','allocated'} for state in states):
        raise RuntimeError(f'{node} 当前不可调度（{" / ".join(states) or "unknown"}）；保留固定索引，未提交作业')
    rows=query(['squeue','--me','-h','-o','%j|%T|%n']).splitlines()
    for row in rows:
        fields=row.split('|')
        if len(fields)==3 and fields[0].startswith(prefix) and node in fields[2].split(','):
            raise RuntimeError(f'{node} 已有同类索引作业 {fields[0]}（{fields[1]}）；不重复提交')
