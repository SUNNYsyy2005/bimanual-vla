"""Start, inspect, and stop the local policy without loading model libraries."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SERVE = ROOT / 'scripts/pi05_local/serve.py'
RUNTIME = ROOT / 'analysis/pi05_local_8099'
PID_FILE = RUNTIME / 'server.pid'
LAUNCH_FILE = RUNTIME / 'launch.json'
LOG_FILE = RUNTIME / 'server.log'


def current():
    """Accept the previous manually started service, but reject unrelated PIDs."""
    try:
        pid = int(PID_FILE.read_text().strip())
        proc = Path('/proc') / str(pid)
        command = (proc / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        if proc.stat().st_uid != os.getuid() or str(SERVE) not in command:
            return None
        launch = json.loads(LAUNCH_FILE.read_text())
        if int(launch['pid']) != pid or launch['command'] != command:
            return None
        port = command[command.index('--port') + 1] if '--port' in command else '8099'
        host = command[command.index('--host') + 1] if '--host' in command else '127.0.0.1'
        return dict(pid=pid, command=command, host=host, port=int(port))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def healthy(service):
    url = f"http://{service['host']}:{service['port']}/healthz"
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=1) as response:
            return response.status == 200 and response.read().strip() == b'OK'
    except OSError:
        return False


def show(service):
    if service is None:
        print('本地 policy 未运行。启动：./local_policy.sh')
        return
    state = '已就绪' if healthy(service) else '进程运行中，健康检查尚未通过'
    print(f"{state}：ws://{service['host']}:{service['port']}（PID {service['pid']}）")
    print(f'日志：{LOG_FILE}')


def start(args):
    service = current()
    if service:
        print('服务已在运行，未重复启动；修改参数请使用 restart。')
        show(service)
        return
    with socket.socket() as probe:
        try:
            probe.bind(('127.0.0.1', args.port))
        except OSError as exc:
            raise RuntimeError(f'{args.port} 端口已被占用，未改动现有进程。') from exc
    command = [sys.executable, '-u', str(SERVE), '--host', '127.0.0.1',
               '--port', str(args.port), '--steps', str(args.steps)]
    if args.checkpoint:
        command.extend(['--checkpoint', str(Path(args.checkpoint).expanduser().resolve())])
    with LOG_FILE.open('ab', buffering=0) as log:
        log.write(f'\n--- local_policy start {time.strftime("%Y-%m-%d %H:%M:%S")} ---\n'.encode())
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    PID_FILE.write_text(f'{process.pid}\n')
    LAUNCH_FILE.write_text(json.dumps(dict(pid=process.pid, command=command,
        bind=f'127.0.0.1:{args.port}'), indent=2))
    service = dict(pid=process.pid, host='127.0.0.1', port=args.port)
    print(f'正在后台加载模型（PID {process.pid}）…', flush=True)
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'启动失败，退出码 {process.returncode}。查看：./local_policy.sh logs')
        if healthy(service):
            show(service)
            return
        time.sleep(0.25)
    raise RuntimeError('45 秒内尚未就绪，进程仍在后台。查看：./local_policy.sh status 或 logs')


def stop():
    service = current()
    if service is None:
        show(None)
        return
    os.kill(service['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 10
    while current() is not None and time.monotonic() < deadline:
        time.sleep(0.1)
    if current() is not None:
        raise RuntimeError('已发送停止信号，服务仍在退出；请稍后查看 status。')
    PID_FILE.unlink(missing_ok=True)
    LAUNCH_FILE.unlink(missing_ok=True)
    print('本地 policy 已停止。')


def main():
    parser = argparse.ArgumentParser(description='本地 Piper Pi05 policy 启停；默认后台启动。')
    parser.add_argument('command', nargs='?', default='start',
                        choices=['start', 'stop', 'restart', 'status', 'logs', 'gui'])
    parser.add_argument('--port', type=int, default=8099, help='启动端口，默认 8099')
    parser.add_argument('--steps', type=int, default=5, help='去噪次数，默认 5')
    parser.add_argument('--checkpoint', help='覆盖默认 15k PyTorch 检查点路径')
    parser.add_argument('--lines', type=int, default=30, help='日志行数，默认 30')
    parser.add_argument('--follow', action='store_true', help='持续跟踪日志')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.steps <= 50 or args.lines <= 0:
        parser.error('port 必须为 1–65535，steps 为 1–50，lines 必须为正数')
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if args.command == 'gui':
        os.execv(sys.executable, [sys.executable, str(ROOT / 'scripts/pi05_local/panel.py')])
    if args.command == 'logs':
        if not LOG_FILE.exists():
            print('暂无服务日志。')
            return
        command = ['tail', '-n', str(args.lines)]
        if args.follow:
            command.append('-F')
        os.execvp('tail', [*command, str(LOG_FILE)])
    with (RUNTIME / 'manager.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.command in ('stop', 'restart'):
            stop()
        if args.command in ('start', 'restart'):
            start(args)
        elif args.command == 'status':
            show(current())


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError) as exc:
        print(f'错误：{exc}', file=sys.stderr)
        sys.exit(1)
