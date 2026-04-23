# PEIRASTIC_control

`PEIRASTIC_control` is a self-contained Franka Panda control stack built on `libfranka`.

It contains:

- `franka-interface/`: the real-time C++ control node that runs on the robot PC (NUC)
- `peirastic/`: the Python package that runs on the desktop PC and talks to the NUC over ZMQ
- `proto/`: shared protobuf command/state definitions

**Full documentation:** see `peirastic/README.md` (desktop API, scripts, NetFT, admittance, and detailed tuning notes).

## Architecture

```text
Desktop Python (peirastic.FrankaInterface)  <----ZMQ---->  NUC C++ (franka-interface)  ->  Robot
```

The desktop does not connect to the robot directly. It streams control messages over ZMQ. The NUC runs the real-time `libfranka` loop and executes the selected controller.

## Supported controllers (NUC)

- `OSC_POSE`
- `OSC_POSITION`
- `OSC_YAW`
- `JOINT_POSITION`
- `JOINT_IMPEDANCE`
- `CARTESIAN_VELOCITY`

`CARTESIAN_VELOCITY` is the **Cartesian-velocity** command path (via `libfranka` Cartesian velocity control). The OSC family is **operational-space torque control** (task-space impedance) and is not the same interface as `CARTESIAN_VELOCITY`.

In `CARTESIAN_VELOCITY` mode, you can send velocity-style commands, and you can also use `RUCKIG_POSE` for smooth absolute pose tracking (translation uses Ruckig smoothing; see the C++ callback for orientation handling details).

## NUC build and run

Build on a local Linux filesystem when possible. Some removable filesystems do not support the symlinks created by shared libraries.

```bash
cd /path/to/PEIRASTIC_control
make
```

This builds `bin/franka-interface` and `bin/gripper-interface` at the **repository root** (via `make install` into the tree).

If the source tree is stored on a removable filesystem, use an out-of-tree build directory on a local Linux filesystem:

```bash
cmake -S . -B /tmp/peirastic_build -DCMAKE_BUILD_TYPE=Release -DBUILD_FRANKA=1 -DBUILD_PEIRASTIC=1
cmake --build /tmp/peirastic_build -j4 --target franka-interface py_protobuf
```

### Start the real-time node (single machine)

From the repository root (so `./bin/...` resolves correctly):

```bash
cd /path/to/PEIRASTIC_control
./bin/franka-interface config/local-host.yml
```

Optional gripper process:

```bash
./bin/gripper-interface config/local-host.yml
```

### Two-machine deployment

For a two-machine setup, start from `config/charmander.yml` and set `ROBOT.IP`, `NUC.IP`, and `PC.IP` to the robot, NUC, and desktop addresses, then run the same binaries with that file.

## Desktop installation

```bash
cd /path/to/PEIRASTIC_control
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## Desktop configuration

`peirastic` defaults to `config/local-host.yml` for most CLI entry points. Edit the YAML and verify at least:

- `ROBOT.IP`
- `NUC.IP` / `PC.IP` (as appropriate for your topology)
- `NUC.SUB_PORT`
- `NUC.PUB_PORT`
- `NUC.GRIPPER_SUB_PORT`
- `NUC.GRIPPER_PUB_PORT`

The desktop must be able to reach the NUC over the configured ports.

## Desktop usage (minimal)

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

### SpaceMouse teleoperation (CLI)

```bash
peirastic.spacemouse_control --interface-cfg local-host.yml --controller-type OSC_POSE
```

`CARTESIAN_VELOCITY` in the SpaceMouse example script is configured as a **delta** command path (not the same thing as `move_pose()`-style `RUCKIG_POSE` absolute tracking).

## Notes

- This repository is self-contained and does not require `deoxys_control` at runtime.
- `Ruckig` is vendored under `third_party/ruckig` and built as part of the C++ stack.
- The admittance example (`peirastic.admittance_target_follow`) is a **Python** outer loop and is not yet moved into the NUC real-time loop.
