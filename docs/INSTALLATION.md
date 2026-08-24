# Installation

This guide separates the robot workstation from the OpenPI Policy/training
server. They have different Python and CUDA requirements and should not share
one environment.

## 1. Supported Setup

| Component | Tested platform | Python |
|---|---|---:|
| Collection GUI, data tools, RTC client | Ubuntu 22.04, x86_64 | 3.10.20 |
| Dashboard and OpenPI Policy server | Ubuntu 22.04, NVIDIA GPU | 3.11.15 |
| RoboTwin/cuRobo evaluation | Ubuntu 22.04, CUDA 12.1 | 3.10/3.11 |

The commands below assume a Conda-compatible installation such as Miniconda,
Mambaforge, or Anaconda. Windows and macOS are not supported for Piper CAN
control.

## 2. System Packages

Install the robot-workstation tools on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  can-utils \
  ffmpeg \
  git \
  iproute2 \
  libgl1 \
  libglib2.0-0 \
  python3-tk \
  v4l-utils
```

The Linux kernel must expose the USB-CAN adapters through SocketCAN. The
official Piper USB-CAN adapter should normally appear with the `gs_usb` driver.

## 3. Clone the Repository

```bash
git clone https://github.com/SUNNYsyy2005/bimanual-vla.git
cd bimanual-vla
```

## 4. Robot Workstation Environment

Create the environment:

```bash
conda create -n dual_arm python=3.10.20 -y
conda activate dual_arm
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

`requirements.txt` covers:

- the collection GUI and camera pipeline;
- Piper SocketCAN control;
- raw NPZ validation and LeRobot v2.1 conversion;
- dataset upload and Dashboard management dependencies;
- the robot-side OpenPI WebSocket/RTC client.

The public OpenPI client is pinned to `0.1.2`; its WebSocket API is compatible
with the source-installed `0.1.0` client used on the verified workstation.
NumPy is intentionally constrained to `1.26.4` because both client versions
declare `numpy<2`.

Verify the Python environment:

```bash
python - <<'PY'
import cv2
import flask
import numpy
import pandas
import pyarrow
import scipy
from PIL import Image
from openpi_client import websocket_client_policy
from piper_sdk import C_PiperInterface_V2

print("Python dependencies: OK")
PY

python -m pip check
bin/bimanual-vla --help
```

Check the non-GUI command entry points without connecting hardware:

```bash
for command in \
  collect-output teleop-bimanual teleop-single home-single \
  data-check data-export data-upload data-validate data-view rtc-client
do
  bin/bimanual-vla "$command" --help >/dev/null
done
echo "Non-GUI CLI entry points: OK"
```

`collect-gui` starts the Tk application directly, so it is intentionally not
included in the `--help` loop. Validate it with `bash start_gui.sh` in a desktop
session.

## 5. CAN Setup

Connect and power the Piper arms before activating their CAN interfaces.
Identify the adapters:

```bash
lsusb -t
ip -br link show type can
```

Activate a single arm at 1 Mbit/s:

```bash
sudo ip link set can0 down 2>/dev/null || true
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 up
ip -s -details link show can0
timeout 2 candump -L can0
```

For the current bimanual GUI, repeat the commands for `can1`. Verify the
physical left/right mapping before enabling or commanding either arm; interface
numbers can change when USB adapters are reconnected.

The repository can also use the Piper SDK activation helper when that SDK
checkout is available:

```bash
bash ~/dual_ARM_project/piper_sdk/piper_sdk/can_activate.sh can0 1000000
```

## 6. Camera Setup

List the V4L2 devices and formats:

```bash
v4l2-ctl --list-devices

for device in /dev/video*; do
  printf '%s\n' "$device"
  v4l2-ctl -d "$device" --get-fmt-video 2>/dev/null || true
done
```

The current rig uses:

- Intel RealSense D435i for `cam_high`;
- one Intel RealSense D405 for `cam_left_wrist`;
- one Intel RealSense D405 for `cam_right_wrist`.

The software prefers stable `/dev/v4l/by-path` selectors. Confirm all three GUI
previews after reconnecting cameras instead of relying on old `/dev/videoN`
numbers. Depth and grayscale nodes are not valid RGB inputs.

## 7. Start the Collection GUI

```bash
conda activate dual_arm
bash start_gui.sh
```

The GUI can be opened without connected hardware, but collection requires fresh
CAN feedback and valid camera frames. See
`docs/collection/GUI_OPERATION_GUIDE.md` for the operating workflow.

## 8. Data-Only Installation Check

The validation and conversion commands do not require a connected robot:

```bash
bin/bimanual-vla data-validate --help
bin/bimanual-vla data-export --help
bin/bimanual-vla data-check --help
bin/bimanual-vla data-upload --help
```

Run the automated test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

## 9. OpenPI Policy and Training Server

The Policy server is a separate environment. Do not install the root
`requirements.txt` over a working OpenPI/JAX environment.

The current Dashboard expects an OpenPI checkout that contains the Piper
training configs and exposes these paths in `server_4090/config.json`:

```json
{
  "openpi_repo": "/path/to/openpi",
  "openpi_python": "/path/to/openpi-python",
  "dataset_root": "/path/to/lerobot",
  "assets_base_dir": "/path/to/openpi/assets",
  "checkpoint_base_dir": "/path/to/openpi/checkpoints",
  "base_checkpoint": "/path/to/pi05_base"
}
```

The verified OpenPI source revision is:

```text
repository: https://github.com/Physical-Intelligence/openpi.git
revision:   15a9616a00943ada6c20a0f158e3adb39df2ccac
Python:     3.11
```

Install the base OpenPI environment using its own upstream `uv` workflow:

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
git checkout 15a9616a00943ada6c20a0f158e3adb39df2ccac

curl -LsSf https://astral.sh/uv/install.sh | sh
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

The upstream checkout alone does not provide this project's custom Piper
training configs. Install or synchronize the validated Piper/OpenPI integration
used by your deployment before starting norm, training, or Policy tasks.

Download the `pi0.5` base checkpoint from the `bimanual-vla` repository:

```bash
python -m scripts.models.download_openpi_checkpoint \
  --checkpoint gs://openpi-assets/checkpoints/pi05_base \
  --source auto \
  --workers 16 \
  --chunks-per-file 16
```

Create the local Dashboard config:

```bash
cp server_4090/config.example.json server_4090/config.json
```

Edit only the local `config.json`; do not commit machine-specific paths or
credentials. Start the service in the foreground for the first validation:

```bash
bash server_4090/run_server_foreground.sh
```

Then check:

```bash
curl -fsS http://127.0.0.1:8090/healthz
```

For the established 4 x RTX 4090 host and user-level systemd deployment, use:

```bash
bash deploy_4090_server.sh
```

This command requires a working SSH host alias and matching remote paths. It
does not install CUDA or create the OpenPI environment.

## 10. H100/H200 Training

H100/H200 jobs use the dedicated OpenPI environment and must run through
Slurm. Before submission, verify:

```bash
hostname
pwd
resources
myquota
```

Do not train, run inference, or process large datasets on `login-server`.
Submit the preparation and training jobs with `sbatch`, or use the Dashboard
API described in `server_4090/API_USAGE.md`.

## 11. RoboTwin and cuRobo Evaluation

RoboTwin, SAPIEN, and cuRobo use a separate CUDA 12.1-compatible environment.
They are intentionally excluded from `requirements.txt`. Follow
`docs/deployment/SERVER_PATHS_ENV_TRAIN_EVAL.md` for the currently validated
environment and rollout commands.

## 12. Common Problems

### `ModuleNotFoundError: piper_sdk`

Confirm the active environment and reinstall the pinned SDK:

```bash
conda activate dual_arm
python -m pip install piper_sdk==0.6.1
```

### `No such device: can0`

Check that the adapter uses `gs_usb`, then reactivate SocketCAN:

```bash
lsusb -t
ip -br link show type can
```

### Camera opens the wrong stream

Use `v4l2-ctl --list-devices` and select an RGB/YUYV/MJPEG node. Do not select
`Z16` depth or `GREY` infrared nodes.

### Tkinter is missing

```bash
sudo apt-get install -y python3-tk
```

Recreate the Conda environment after installing Tk if the active Python still
cannot import `tkinter`.

### OpenPI CUDA/JAX errors

Do not mix the robot workstation environment with OpenPI or cuRobo. Confirm the
host-specific Python, CUDA toolkit, driver, and checkpoint format described in
the deployment documentation.

## 13. Security Notes

- Never commit Dashboard tokens, passwords, SSH keys, or VPN configuration.
- Keep `server_4090/config.json` machine-local when it contains private paths.
- Start the RTC client without `--allow-execution` for the first Policy check.
- Real motion also requires a non-expired Dashboard `EXECUTE` authorization.
- Verify the physical CAN-to-arm mapping before every deployment.
