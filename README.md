<div align="center">

# bimanual-vla

**End-to-end Piper robot data collection, OpenPI training, and real-time policy deployment**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](docs/INSTALLATION.md) [![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-E95420?logo=ubuntu&logoColor=white)](docs/INSTALLATION.md) [![LeRobot](https://img.shields.io/badge/LeRobot-v2.1-FFD21E)](https://github.com/huggingface/lerobot) [![OpenPI](https://img.shields.io/badge/OpenPI-pi0%20%7C%20pi0.5-6C5CE7)](https://github.com/Physical-Intelligence/openpi) [![Control](https://img.shields.io/badge/Robot_Control-20_Hz-2E8B57)](#real-robot-deployment)

[**Code**](https://github.com/SUNNYsyy2005/bimanual-vla) &nbsp;&middot;&nbsp; [**Documentation**](docs/README.md) &nbsp;&middot;&nbsp; [**Demo**](#demo) &nbsp;&middot;&nbsp; [**Installation**](docs/INSTALLATION.md)

</div>

<p align="center">
  <img src="assets/hero.jpg" width="100%" alt="Bimanual Piper robot platform">
</p>

## Overview

`bimanual-vla` connects the complete real-robot learning loop in one repository:
collect synchronized Piper demonstrations, validate and export LeRobot datasets,
fine-tune OpenPI `pi0` / `pi0.5` policies, and deploy action chunks through a
latency-aware and safety-gated execution stack.

```text
Piper + RGB Cameras  ->  Robot Demonstrations  ->  LeRobot Dataset
        ->  OpenPI LoRA Fine-tuning  ->  Policy Server
        ->  Real-Time Chunking  ->  Safety Layer  ->  Piper Execution
```

The workflow supports single-arm and bimanual Piper operation, joint-space and
end-effector (`delivery`) data contracts, and GUI/CLI workflows. Deployment
behavior depends on the active Policy contract and hardware configuration.

## Highlights

- **End-to-end real-robot pipeline** from demonstrations to closed-loop policy execution.
- **Multi-camera Piper collection** with GUI operation, teleoperation, replay, and episode management.
- **LeRobot-compatible data** with versioned contracts, validation, conversion, and upload tools.
- **OpenPI training stack** for `pi0` / `pi0.5` LoRA fine-tuning and optional multi-GPU FSDP.
- **Real-time, safety-aware deployment** with model-side RTC, configurable inference scheduling, 20 Hz robot control, and fail-closed execution.

## System Overview

<p align="center">
  <img src="assets/architecture_overview.png" width="100%" alt="bimanual-vla system architecture">
</p>

The public workflow is organized into three stages. Detailed data contracts,
validation rules, transport fields, and server operations live in
[the documentation](docs/README.md) instead of the project homepage.

| Stage | Input | Core components | Output |
|---|---|---|---|
| Data collection | Piper feedback, RGB views, language instruction | GUI or teleoperation, synchronized recording, episode validation | Raw robot episodes and LeRobot v2.1 datasets |
| VLA training | Images, robot state, actions, task text | OpenPI `pi0` / `pi0.5`, LoRA, optional FSDP | Versioned Policy checkpoints |
| Real-time deployment | Live observation and Policy checkpoint | Asynchronous WebSocket inference, model-side RTC, action queues, safety gates | Validated Piper commands at the configured control rate |

## Hardware Setup

### Robot and Sensors

| Component | Configuration |
|---|---|
| Robot | Piper robotic arm, single-arm or bimanual |
| Kinematics | 6 revolute joints per arm plus gripper |
| End effector | Piper gripper, approximately `0.00-0.07 m` opening range |
| Control | Piper SDK over SocketCAN at `1 Mbit/s` |
| Overhead camera | Intel RealSense D435i, third-person RGB view |
| Wrist cameras | Intel RealSense D405, one RGB view per active wrist |
| Collection / control rate | `20 Hz` |
| Camera source rate | Typically `30 FPS` |

Depth is available on the camera hardware but is not part of the current
training contract. Stable `/dev/v4l/by-path` camera selectors are recommended
because `/dev/videoN` indices can change after USB reconnection.

### Compute

| Role | Typical platform | Responsibility |
|---|---|---|
| Robot workstation | Ubuntu 22.04, x86_64 | CAN control, camera capture, GUI, RTC client |
| Policy and Dashboard server | NVIDIA RTX 4090 workstation | Dataset management, Policy inference, telemetry |
| Training cluster | NVIDIA H100 / H200 with Slurm | Multi-GPU fine-tuning and evaluation |

ROS, ROS2, gRPC, and EtherCAT are not required by the current stack.

## Data Collection

<p align="center">
  <img src="assets/collection_gui.jpg" width="100%" alt="Piper data collection GUI">
</p>

The collection application provides device discovery, live multi-camera
preview, task metadata, robot feedback, episode controls, replay, and direct
conversion/upload. It supports:

- single-arm and bimanual collection;
- master/slave teleoperation or output-arm feedback recording;
- overhead plus left/right wrist RGB streams;
- `joint` and `delivery` data schemas;
- success/failure labeling and guarded episode publication.

Start the GUI from the robot workstation:

```bash
conda activate dual_arm
bash start_gui.sh
```

For command-line bimanual teleoperation and recording:

```bash
bin/bimanual-vla teleop-bimanual --record --schema joint
```

See the [GUI operation guide](docs/collection/GUI_OPERATION_GUIDE.md) and
[data collection guide](docs/collection/DATA_COLLECTION_GUIDE.md) before
connecting hardware.

## Dataset Pipeline

```text
Raw NPZ Episodes  ->  Contract Validation  ->  LeRobot v2.1
       ->  Dataset Inspection / Merge  ->  OpenPI Training Dataset
```

Each episode combines the signals needed by a vision-language-action policy:

| Signal | Content |
|---|---|
| RGB observations | Overhead view and one or two wrist views |
| Robot state | Joint state or absolute end-effector state |
| Actions | Joint targets or end-effector targets / model deltas |
| Language | Natural-language task instruction |
| Metadata | Timestamps, schema, arm mode, contract version, success label |

Supported v3 action contracts are explicit and versioned:

| Mode | Schema | Observation | Raw action | Model action |
|---|---|---:|---:|---:|
| Single arm | Joint | 7D | 7D | 7D |
| Bimanual | Joint | 14D | 14D | 14D |
| Single arm | Delivery | 10D | 10D | 7D |
| Bimanual | Delivery | 20D | 20D | 14D |

Bimanual vectors always use `left + right` ordering, and the normalized gripper
convention is `0 = closed`, `1 = open`.

```bash
# Validate raw episodes
bin/bimanual-vla data-validate \
  --input-dir episodes_piper_v21 \
  --target-fps 20

# Export successful episodes to LeRobot v2.1
bin/bimanual-vla data-export \
  --input-dir episodes_piper_v21 \
  --repo-id piper/piper_v1 \
  --root lerobot_datasets/piper_v1 \
  --fps 20

# Validate the exported dataset
bin/bimanual-vla data-check lerobot_datasets/piper_v1
```

The authoritative field definitions are documented in the
[Piper data contract](docs/collection/PIPER_DATA_CONTRACT.md).

## VLA Training

The training backend integrates OpenPI `pi0` and `pi0.5` with LoRA fine-tuning.
The Dashboard manages dataset selection, persistent train/test splits,
normalization statistics, training jobs, checkpoints, and held-out evaluation.
FSDP can distribute supported runs across multiple GPUs; H100/H200 jobs are
submitted through Slurm.

```yaml
Input:
  Images: overhead RGB + wrist RGB views
  State: joint 7D/14D or end-effector 10D/20D
  Language: natural-language task instruction

Output:
  Action chunk: 50 x 7D or 50 x 14D
  Execution target: schema-dependent decoded action with safety checks
```

Download an OpenPI base checkpoint with the repository helper:

```bash
python -m scripts.models.download_openpi_checkpoint \
  --checkpoint gs://openpi-assets/checkpoints/pi05_base \
  --source auto \
  --workers 16 \
  --chunks-per-file 16
```

Training and Policy serving use a separate OpenPI Python 3.11 environment. Do
not install the robot workstation requirements over a working JAX/OpenPI
environment. Follow the [installation guide](docs/INSTALLATION.md#9-openpi-policy-and-training-server)
for the pinned upstream revision and server configuration.

## Real-Robot Deployment

```text
Robot Workstation                         Policy Server
-----------------                        -----------------
RGB cameras + Piper state  --WebSocket-> OpenPI pi0 / pi0.5
RTC session and timing      --WebSocket-> Model-side RTC
Action queue + safety       <-WebSocket-- Timestamped action chunk
        |
        +--> 20 Hz validated Piper commands
```

Start the Dashboard/Policy stack on a configured inference server, then run the
robot client in **shadow mode** first:

```bash
export BIMANUAL_VLA_POLICY_HOST="<policy-server-host>"

bin/bimanual-vla rtc-client \
  --host "$BIMANUAL_VLA_POLICY_HOST" \
  --port 8000 \
  --arm-mode bimanual \
  --arm-side both \
  --left-can can0 \
  --right-can can1 \
  --cam-high-device auto \
  --cam-left-wrist-device auto \
  --cam-right-wrist-device auto \
  --instruction "pick up the cube" \
  --hz 4 \
  --control-hz 20 \
  --rtc-enabled
```

> [!CAUTION]
> Real motion requires both the local `--allow-execution` flag and a non-expired
> Dashboard `EXECUTE` authorization for the same Policy. A stale camera, CAN or
> Policy stream, a contract mismatch, or any failed safety check blocks commands.

The execution layer validates action age and shape, workspace bounds, joint and
gripper changes, IK feasibility, Piper state, and authorization on every control
cycle. RTC is applied inside model denoising; it is not client-side interpolation.
See the [RTC deployment guide](docs/deployment/RTC_CLIENT_GUIDE.md).

### Inference timing and execution controls

Policy inference is asynchronous and separate from the robot's 20 Hz control
loop. The client supports two scheduling modes:

- `--inference-trigger-mode periodic` (default): attempt a new inference at
  `--hz` (default 4 Hz), with at most one request in flight. Actual throughput
  depends on end-to-end latency.
- `--inference-trigger-mode chunk_step --inference-trigger-step 10`: launch
  the next request when the accepted chunk reaches the specified source step.
  In this mode `--hz` governs initialization/recovery retries rather than
  the normal per-chunk trigger.

The server performs RTC prefix guidance during model denoising. Additional
client-side blending is separate and should be tuned deliberately, not treated
as equivalent to model-side RTC.

The execution path includes configurable joint trajectory shaping and safety
interlocks:

- `--trajectory-shaping` is enabled by default, with independently
  configurable low-pass, tracking, speed, acceleration, jerk and lookahead
  stages. Use `--no-trajectory-shaping` for controlled comparisons.
- `--trajectory-spike-suppression` is **disabled by default**; it targets
  isolated acceleration spikes in short execution horizons and can be tested
  separately from trajectory shaping.
- `--blend-profile smootherstep` controls the optional action-boundary blend.
- `--gripper-open-lookahead-steps` can anticipate opening while keeping
  gripper filtering and safety checks active.
- `--reject-external-control-streams` guards against simultaneous
  high-rate Piper control; `--auto-return` is enabled by default for a
  monitored return to the startup pose on normal shutdown.

**Important:** smoothing and motion limits trade tracking responsiveness for
command continuity. Aggressive filtering or lookahead can cause lag or
overshoot; disabling shaping can expose high-frequency joint jitter. Validate
settings in shadow mode and then in a supervised, low-risk hardware test before
normal deployment. Neither mode guarantees the absence of overshoot.

Monitoring and deployment recordings can distinguish raw Policy predictions
from commands actually sent to the robot, including velocity, acceleration,
tracking error, chunk boundaries, and timing. See the
[RTC deployment guide](docs/deployment/RTC_CLIENT_GUIDE.md) for parameters,
diagnostic definitions, and safety prerequisites.

## Demo

<p align="center">
  <img src="assets/demo.gif" width="720" alt="Bimanual Piper real-robot demonstration">
</p>

Real-robot manipulation demonstration. Playback is accelerated for README viewing; it does not represent the live robot control rate.

## Dashboard

<p align="center">
  <img src="assets/dashboard.png" width="100%" alt="Training and deployment Dashboard">
</p>

The Dashboard supports dataset/episode management, normalization, LoRA/FSDP
training workflows, Policy lifecycle controls, live telemetry, and trajectory
analysis. Diagnostics distinguish Policy output from executed commands, and
include chunk-boundary and timing views when the underlying recordings contain
the required signals.

## Installation

The robot workstation is tested with Ubuntu 22.04 and Python 3.10:

```bash
git clone https://github.com/SUNNYsyy2005/bimanual-vla.git
cd bimanual-vla

conda create -n dual_arm python=3.10.20 -y
conda activate dual_arm
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

bin/bimanual-vla --help
```

System packages, SocketCAN activation, RealSense selection, OpenPI setup,
Dashboard configuration, Slurm, verification, and troubleshooting are covered
in the [complete installation guide](docs/INSTALLATION.md).

## Project Layout

```text
.
|-- assets/                    # README images and demos
|-- bin/bimanual-vla          # Unified command entry point
|-- bimanual_vla/
|   |-- collection/           # GUI, cameras, teleoperation, robot sessions
|   |-- data/                 # Contracts, validation, export, upload, replay
|   `-- deployment/           # RTC client, Policy adapter, safety execution
|-- server_4090/              # Dashboard backend, frontend, Policy services
|-- docs/                     # Installation, collection, and deployment guides
|-- scripts/                  # Model, maintenance, analysis, and smoke tools
|-- jobs/                     # Slurm and analysis jobs
|-- tests/                    # Automated test suite
`-- requirements.txt          # Robot workstation dependencies
```

Runtime datasets, checkpoints, telemetry, and deployment recordings are not
source code and are excluded from Git.

## Documentation

| Topic | Guide |
|---|---|
| Installation and hardware | [Installation](docs/INSTALLATION.md) |
| Collection GUI | [GUI operation guide](docs/collection/GUI_OPERATION_GUIDE.md) |
| Data collection | [Data collection guide](docs/collection/DATA_COLLECTION_GUIDE.md) |
| Dataset fields and semantics | [Piper data contract](docs/collection/PIPER_DATA_CONTRACT.md) |
| 7D/10D action design | [OpenPI action design](docs/collection/PI05_PIPER_7D_10D_DATA_ACTION_DESIGN.md) |
| Real-robot inference | [RTC client guide](docs/deployment/RTC_CLIENT_GUIDE.md) |
| Dashboard backend | [Dashboard operations](server_4090/README.md) |
| Dashboard API | [API reference](server_4090/API_USAGE.md) |

## Testing

```bash
python -m unittest discover -s tests -v
```

Hardware and Policy smoke tests are available under `scripts/smoke/`. Run them
only in the matching robot or inference-server environment.

## Acknowledgements

This project builds on [OpenPI](https://github.com/Physical-Intelligence/openpi),
[LeRobot](https://github.com/huggingface/lerobot), and the Piper robot SDK.
