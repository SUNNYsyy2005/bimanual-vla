"""Desktop controls for the local policy service; opening this never loads a model."""
import fcntl
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import manage_server as manager

MODEL_ROOT = Path('/home/user/models/pi05_5060')
DEFAULT_MODEL = MODEL_ROOT / 'joint_trim_nopad_no_rtc_b4_20261022/15000_pt'
FONT = 'Noto Sans CJK SC'


class PolicyPanel:
    def __init__(self, root):
        self.root = root
        self.events = queue.Queue()
        self.closed = threading.Event()
        self.busy = False
        self.service = None
        self.model_valid = False
        self.log_position = None
        self.models = {}
        self.model = tk.StringVar(value='')
        self.port = tk.StringVar(value='8099')
        self.steps = tk.StringVar(value='5')
        self.address = tk.StringVar(value='ws://127.0.0.1:8099')
        self.state = tk.StringVar(value='正在检查')
        self.detail = tk.StringVar(value='检查本地服务状态…')
        self.gpu = tk.StringVar(value='正在读取显卡状态…')
        self.notice = tk.StringVar(value='服务保持停止，点击“启动服务”后才加载模型。')
        root.title('本地 Policy · Piper π0.5')
        root.geometry('1120x830')
        root.minsize(960, 760)
        root.configure(bg='#f2f5fa')
        root.option_add('*Font', (FONT, 10))
        self.build()
        self.discover_models()
        self.model.trace_add('write', lambda *_: self.update_model())
        self.port.trace_add('write', lambda *_: self.update_address())
        self.update_model()
        root.protocol('WM_DELETE_WINDOW', self.close)
        threading.Thread(target=self.monitor, daemon=True).start()
        root.after(100, self.tick)

    def build(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('TFrame', background='#ffffff')
        style.configure('TLabel', background='#ffffff', foreground='#344054', font=(FONT, 10))
        style.configure('Title.TLabel', font=(FONT, 13, 'bold'), foreground='#101828')
        style.configure('Muted.TLabel', foreground='#667085', font=(FONT, 9))
        style.configure('TButton', padding=(16, 9), font=(FONT, 10))
        style.configure('Primary.TButton', background='#2563eb', foreground='white', borderwidth=0)
        style.map('Primary.TButton', background=[('disabled', '#cbd5e1'), ('active', '#1d4ed8')],
                  foreground=[('disabled', '#64748b')])
        style.configure('TEntry', padding=7)
        style.configure('TCombobox', padding=7)

        hero = tk.Frame(self.root, bg='#101828', padx=28, pady=22)
        hero.pack(fill='x')
        tk.Label(hero, text='本地推理服务', font=(FONT, 22, 'bold'), fg='white', bg='#101828').pack(anchor='w')
        tk.Label(hero, text='Piper π0.5   /   模型管理与运行状态', fg='#a9bad4', bg='#101828',
                 font=(FONT, 10)).pack(anchor='w', pady=(5, 0))
        body = tk.Frame(self.root, bg='#f2f5fa', padx=24, pady=20)
        body.pack(fill='both', expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(2, weight=1)
        top = tk.Frame(body, bg='#f2f5fa')
        top.grid(row=0, column=0, sticky='ew')
        top.columnconfigure(0, weight=3)
        top.columnconfigure(1, weight=2)

        config = ttk.Frame(top, padding=20)
        config.grid(row=0, column=0, sticky='nsew', padx=(0, 14))
        config.columnconfigure(0, weight=1)
        ttk.Label(config, text='模型与推理配置', style='Title.TLabel').grid(row=0, column=0, sticky='w', columnspan=2)
        ttk.Label(config, text='模型目录', style='Muted.TLabel').grid(row=1, column=0, sticky='w', pady=(16, 5))
        self.model_box = ttk.Combobox(config, textvariable=self.model, state='readonly', width=55)
        self.model_box.grid(row=2, column=0, sticky='ew', padx=(0, 8))
        ttk.Button(config, text='选择…', command=self.browse).grid(row=2, column=1)
        self.model_info = ttk.Label(config, text='', style='Muted.TLabel', wraplength=510)
        self.model_info.grid(row=3, column=0, columnspan=2, sticky='w', pady=(8, 14))
        fields = ttk.Frame(config)
        fields.grid(row=4, column=0, columnspan=2, sticky='ew')
        ttk.Label(fields, text='本地端口').grid(row=0, column=0, sticky='w')
        ttk.Label(fields, text='去噪次数').grid(row=0, column=1, sticky='w', padx=(24, 0))
        ttk.Entry(fields, textvariable=self.port, width=12).grid(row=1, column=0, pady=(5, 0))
        ttk.Combobox(fields, textvariable=self.steps, values=['5', '10'], state='readonly', width=10).grid(
            row=1, column=1, padx=(24, 0), pady=(5, 0))
        ttk.Label(config, text='BF16 精度  ·  训练式 RTC  ·  双臂 Piper', style='Muted.TLabel').grid(
            row=5, column=0, columnspan=2, sticky='w', pady=(16, 0))

        status = ttk.Frame(top, padding=20)
        status.grid(row=0, column=1, sticky='nsew')
        ttk.Label(status, text='服务状态', style='Title.TLabel').pack(anchor='w')
        self.badge = tk.Label(status, textvariable=self.state, fg='#475467', bg='#eef2f6',
                              font=(FONT, 16, 'bold'), padx=14, pady=8)
        self.badge.pack(anchor='w', pady=(16, 10))
        ttk.Label(status, textvariable=self.detail, style='Muted.TLabel', wraplength=340).pack(anchor='w')
        address = ttk.Frame(status)
        address.pack(fill='x', pady=(18, 0))
        ttk.Entry(address, textvariable=self.address, state='readonly', width=27).pack(side='left', fill='x', expand=True)
        ttk.Button(address, text='复制', command=self.copy_address).pack(side='left', padx=(5, 0))
        ttk.Label(status, textvariable=self.gpu, style='Muted.TLabel', wraplength=340).pack(anchor='w', pady=(15, 0))

        bar = tk.Frame(body, bg='#f2f5fa')
        bar.grid(row=1, column=0, sticky='ew', pady=16)
        self.start_button = ttk.Button(bar, text='启动服务', style='Primary.TButton', command=lambda: self.operate('start'))
        self.start_button.pack(side='left')
        self.stop_button = ttk.Button(bar, text='停止服务', command=lambda: self.operate('stop'))
        self.stop_button.pack(side='left', padx=10)
        self.restart_button = ttk.Button(bar, text='应用配置并重启', command=lambda: self.operate('restart'))
        self.restart_button.pack(side='left')
        ttk.Label(bar, textvariable=self.notice, background='#f2f5fa', style='Muted.TLabel',
                  wraplength=460).pack(side='left', padx=(20, 0))

        logs = ttk.Frame(body, padding=(16, 12))
        logs.grid(row=2, column=0, sticky='nsew')
        heading = ttk.Frame(logs)
        heading.pack(fill='x', pady=(0, 10))
        ttk.Label(heading, text='运行日志', style='Title.TLabel').pack(side='left')
        ttk.Button(heading, text='打开日志目录', command=self.open_logs).pack(side='right')
        self.log = scrolledtext.ScrolledText(logs, height=11, bg='#101828', fg='#d4deed',
            insertbackground='white', font=('Noto Sans Mono CJK SC', 9), relief='flat', borderwidth=0,
            padx=12, pady=10, state='disabled', wrap='word')
        self.log.pack(fill='both', expand=True)
        ttk.Label(body, text='关闭窗口后，已启动的服务会继续在后台运行。', background='#f2f5fa',
                  style='Muted.TLabel').grid(row=3, column=0, sticky='w', pady=(12, 0))
        self.buttons()

    def discover_models(self):
        paths = sorted(MODEL_ROOT.glob('*/15000_pt/provenance.json'))
        for path in paths:
            self.models[f'{path.parent.parent.name} / {path.parent.name}'] = path.parent
        if DEFAULT_MODEL.exists():
            label = f'{DEFAULT_MODEL.parent.name} / {DEFAULT_MODEL.name}'
            self.models[label] = DEFAULT_MODEL
            self.model.set(label)
        self.model_box['values'] = list(self.models)

    def model_path(self):
        return self.models.get(self.model.get(), Path(self.model.get()))

    def update_model(self):
        self.model_valid = False
        try:
            data = json.loads((self.model_path() / 'provenance.json').read_text())
            settings = data['action_contract']['finetuning']
            if data['checkpoint_step'] != 15000 or not data['all_checks_passed']:
                raise ValueError('需要已校验的 Piper 15k 检查点')
            self.model_valid = True
            rtc_label = f"RTC 最大延迟 {settings['rtc_training_max_delay']} 步" if settings['rtc_training'] else '无 RTC'
            self.model_info.configure(text=f"训练步数 {data['checkpoint_step']:,}  ·  {rtc_label}  ·  {data['action_contract']['arm_mode']} / {data['action_contract']['arm_side']}\n{self.model_path()}")
        except (OSError, ValueError, KeyError) as exc:
            self.model_info.configure(text=f'模型不可用：{exc}')
        self.buttons()

    def update_address(self):
        if self.service is None:
            self.address.set(f'ws://127.0.0.1:{self.port.get()}')

    def browse(self):
        path = filedialog.askdirectory(title='选择 PyTorch 模型目录（15000_pt）', initialdir=str(MODEL_ROOT))
        if path:
            label = f'{Path(path).parent.name} / {Path(path).name}'
            self.models[label] = Path(path)
            self.model_box['values'] = list(self.models)
            self.model.set(label)

    def copy_address(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.address.get())
        self.notice.set('连接地址已复制。')

    def open_logs(self):
        subprocess.Popen(['xdg-open', str(manager.RUNTIME)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def buttons(self):
        self.start_button.configure(state='disabled' if self.busy or self.service or not self.model_valid else 'normal')
        self.stop_button.configure(state='disabled' if self.busy or not self.service else 'normal')
        self.restart_button.configure(state='disabled' if self.busy or not self.service or not self.model_valid else 'normal')

    def operate(self, command):
        if self.busy:
            return
        arguments = [sys.executable, str(Path(manager.__file__).resolve()), command]
        if command != 'stop':
            try:
                port, steps = int(self.port.get()), int(self.steps.get())
                if not 1 <= port <= 65535 or not 1 <= steps <= 50:
                    raise ValueError('端口应为 1–65535，去噪次数应为 1–50')
                if not (self.model_path() / 'provenance.json').is_file():
                    raise ValueError('请选择包含 provenance.json 的已转换模型目录')
            except ValueError as exc:
                messagebox.showerror('配置有误', str(exc), parent=self.root)
                return
            arguments += ['--port', str(port), '--steps', str(steps), '--checkpoint', str(self.model_path())]
        self.busy = True
        self.notice.set({'start': '正在加载模型…', 'stop': '正在停止服务…', 'restart': '正在应用配置并重启…'}[command])
        self.buttons()

        def work():
            try:
                process = subprocess.Popen(arguments, cwd=manager.ROOT, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    self.events.put(('log', '[管理] ' + line))
                self.events.put(('done', (command, process.wait())))
            except OSError as exc:
                self.events.put(('log', f'[管理] {exc}\n'))
                self.events.put(('done', (command, 1)))
        threading.Thread(target=work, daemon=True).start()

    def monitor(self):
        count = 0
        while not self.closed.is_set():
            service = manager.current()
            ready = bool(service and manager.healthy(service))
            self.events.put(('status', (service, ready)))
            if count % 3 == 0:
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,temperature.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2, check=True)
                    used, total, temperature = [float(x) for x in result.stdout.splitlines()[0].split(',')]
                    self.events.put(('gpu', f'整卡显存 {used / 1024:.1f} / {total / 1024:.1f} GiB   ·   {temperature:.0f}°C'))
                except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                    self.events.put(('gpu', '暂时无法读取显卡状态'))
            count += 1
            self.closed.wait(2)

    def append_log(self, text):
        at_bottom = self.log.yview()[1] > 0.97
        self.log.configure(state='normal')
        self.log.insert('end', text)
        if int(self.log.index('end-1c').split('.')[0]) > 1200:
            self.log.delete('1.0', '300.0')
        if at_bottom:
            self.log.see('end')
        self.log.configure(state='disabled')

    def read_logs(self):
        try:
            size = manager.LOG_FILE.stat().st_size
            if self.log_position is None or size < self.log_position:
                self.log_position = max(0, size - 18000)
                self.append_log('── 以下为近期服务日志 ──\n')
            with manager.LOG_FILE.open('rb') as stream:
                stream.seek(self.log_position)
                data = stream.read(65536)
                self.log_position = stream.tell()
            if data:
                self.append_log(data.decode(errors='replace'))
        except OSError:
            pass

    def tick(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == 'status':
                    self.service, ready = data
                    self.state.set('已就绪' if ready else ('正在加载' if self.service else '已停止'))
                    self.badge.configure(bg='#dcfce7' if ready else '#eef2f6', fg='#166534' if ready else '#475467')
                    if self.service:
                        self.address.set(f"ws://{self.service['host']}:{self.service['port']}")
                        self.detail.set(f"进程 PID {self.service['pid']}\n已就绪，可连接客户端。" if ready else f"进程 PID {self.service['pid']}\n等待模型加载与健康检查。")
                    else:
                        self.detail.set('模型未加载，推理服务未监听。')
                        self.update_address()
                    self.buttons()
                elif kind == 'gpu':
                    self.gpu.set(data)
                elif kind == 'log':
                    self.append_log(data)
                elif kind == 'done':
                    self.busy = False
                    command, code = data
                    self.notice.set(('操作完成。' if code == 0 else '操作未成功，请查看下方日志。'))
                    self.buttons()
        except queue.Empty:
            pass
        self.read_logs()
        if not self.closed.is_set():
            self.root.after(300, self.tick)

    def close(self):
        self.closed.set()
        self.root.destroy()


def main():
    manager.RUNTIME.mkdir(parents=True, exist_ok=True)
    with (manager.RUNTIME / 'panel.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('本地 Policy 管理窗口已打开。')
            return
        root = tk.Tk()
        PolicyPanel(root)
        root.mainloop()


if __name__ == '__main__':
    main()
