# PEIRASTIC_control

`PEIRASTIC_control` is a standalone Franka Panda control stack built on top of
`libfranka`.

It contains:

- `franka-interface/`: the real-time C++ control node that runs on the robot PC
- `peirastic/`: the Python package used on the desktop PC to send commands and read robot state
- `proto/`: the shared protobuf command/state definitions

## Architecture

```text
Desktop Python (peirastic.FrankaInterface)  <----ZMQ---->  NUC C++ (franka-interface)  ->  Robot
```

The desktop PC does not connect to the robot directly. It sends control
messages through ZMQ. The NUC runs the real-time `libfranka` loop and executes
the selected controller.

## Supported Controllers

- `OSC_POSE`
- `OSC_POSITION`
- `OSC_YAW`
- `JOINT_POSITION`
- `JOINT_IMPEDANCE`
- `CARTESIAN_VELOCITY`

`CARTESIAN_VELOCITY` is the only Cartesian control mode. It supports both
velocity commands and `RUCKIG_POSE` absolute pose tracking with jerk-limited
retargeting in the real-time loop.

## NUC Build and Run

Build on a local Linux filesystem when possible. Some removable filesystems do
not support the symlinks created by shared libraries.

```bash
cd /path/to/PEIRASTIC_control
make
```

This builds `bin/franka-interface`.

If the source tree is stored on a removable filesystem, use an out-of-tree
build directory on a local Linux filesystem:

```bash
cmake -S . -B /tmp/peirastic_build -DCMAKE_BUILD_TYPE=Release -DBUILD_FRANKA=1 -DBUILD_PEIRASTIC=1
cmake --build /tmp/peirastic_build -j4 --target franka-interface py_protobuf
```

Start the control node on the NUC:

```bash
./bin/franka-interface config/local-host.yml
```

## Desktop Installation

Create and activate a Python environment on the desktop PC:

```bash
cd /path/to/PEIRASTIC_control
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

If you use the SpaceMouse USB driver, `hidapi` is installed from
`requirements.txt`.

## Desktop Configuration

Edit `config/local-host.yml` and verify at least:

- `ROBOT.IP`
- `NUC.IP`
- `NUC.SUB_PORT`
- `NUC.PUB_PORT`
- `NUC.GRIPPER_SUB_PORT`
- `NUC.GRIPPER_PUB_PORT`

The desktop must be able to reach the NUC over the configured ports.

## Desktop Usage

### Python API

```python
from peirastic import config_root
from peirastic.franka_interface import FrankaInterface
from peirastic.utils.config_utils import get_default_controller_config

robot = FrankaInterface(f"{config_root}/local-host.yml", use_visualizer=False)

cfg = get_default_controller_config("CARTESIAN_VELOCITY")
action = [0.50, 0.00, 0.30, 0.0, 0.0, 0.0, -1.0]

robot.control(
    controller_type="CARTESIAN_VELOCITY",
    action=action,
    controller_cfg=cfg,
)
```

For delta commands, set `is_delta: true` in the controller config.

### SpaceMouse Teleoperation

Run SpaceMouse teleoperation from the desktop:

```bash
peirastic.spacemouse_control --interface-cfg local-host.yml --controller-type OSC_POSE
```

Available controller types for the SpaceMouse script:

- `OSC_POSE`
- `OSC_POSITION`
- `OSC_YAW`
- `CARTESIAN_VELOCITY`

Example for Cartesian pose tracking through velocity mode:

```bash
peirastic.spacemouse_control --interface-cfg local-host.yml --controller-type CARTESIAN_VELOCITY
```

The SpaceMouse helper uses delta pose updates with `RUCKIG_POSE` under
`CARTESIAN_VELOCITY`.

## Notes

- This repository is self-contained and does not require `deoxys_control` at runtime.
- `Ruckig` is vendored under `third_party/ruckig` and compiled locally as part of the C++ stack.
- The current admittance prototype still lives in Python scripts and is not yet moved into the NUC real-time loop.
# PEIRASTIC_control
