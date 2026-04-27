# PEIrastic

`peirastic` is the desktop-side Python package in `PEIRASTIC_control` (import
and CLI name stay **lowercase**). It sends control commands over ZMQ to the
real-time C++ `franka-interface` process. The high-level API includes:

- **Joint reset** and **joint-mode control** (`JOINT_POSITION`, `JOINT_IMPEDANCE`)
- **Cartesian** target following (velocity commands, `RUCKIG_POSE` / `move_pose` pose tracking)
- **OSC** target following (`OSC_POSE`)
- **Admittance** (Python outer loop, OSC inner loop, NetFT wrench)
- **Dynamic mode switching** (swap controller types at runtime; session-scoped `control_session` / `session_hard_reset` to avoid half-initialized or stale interpolator state)
- **NetFT** calibration, identification, and named joint targets

This README focuses on the **single-machine** setup, where Python and the C++
control node run on the same host. A **two-machine** deployment is also
supported: put `franka-interface` (and the gripper process, if used) on the
robot-side computer and run `peirastic` on another machine on the same LAN, by
pointing the interface YAML at the correct `ROBOT` / `NUC` IPs and ZMQ ports.
The examples below use localhost-oriented configs for clarity.

## What lives here

- `peirastic/franka_interface/`: Python API (`FrankaInterface`)
- `peirastic/scripts/`: command-line examples and utility scripts
- `peirastic/netft_calib/`: NetFT calibration, identification, and named joint states

## Installation

**One tree, two builds:** `make` produces the C++ `franka-interface`; `pip
install -e .` installs `peirastic` from the same root. No stand-alone prebuilt
C++ package in this doc. On two machines, run `make` on the host that runs
`franka-interface`, and `pip` on the host that runs Python. All commands below
are from the `PEIRASTIC_control` repository root (how you obtain the tree is
your choice). Out-of-tree `cmake` options: root `README.md`.

### 1. System requirements

- Python 3 with `venv`
- CMake / a C++ toolchain
- Protobuf runtime and `protoc`
- Eigen3
- Poco
- ZeroMQ / `libzmq`

ROS is **not** required for the basic Python API; it **is** required for
SpaceNav examples, NetFT tools, and the admittance script.

### 2. Build the C++ control node

Compiles **`franka-interface`** —the
**libfranka** + ZMQ processes your Python will talk to. **Step 3** adds the
`peirastic` client; run the binaries from this `make` to actually move the arm.

```bash
cd /path/to/PEIRASTIC_control
make
```

`bin/franka-interface` and `bin/gripper-interface` are created under the **repo
root** .

### 3. Install the Python package

From the same **repository root** as in step 2:

```bash
cd /path/to/PEIRASTIC_control
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Useful entry points after install (more in `setup.py`):

- `peirastic.get_controller_info` / `peirastic.get_controller_list`
- `peirastic.reset_joints`
- `peirastic.spacenav_mode_switch_test`
- `peirastic.netft_data_acquisition`

### 4. Configuration (single-machine, recommended)

Use `config/local-host.yml`. Check at least:

- `ROBOT.IP`
- `NUC.IP`
- `NUC.SUB_PORT`
- `NUC.PUB_PORT`
- `NUC.GRIPPER_SUB_PORT`
- `NUC.GRIPPER_PUB_PORT`

### 5. Start the real-time node

From the same **repository root** (so `./bin/...` resolves correctly):

```bash
cd /path/to/PEIRASTIC_control
./bin/franka-interface config/local-host.yml
```

If you use the gripper process (same `cd` as above):

```bash
./bin/gripper-interface config/local-host.yml
```

For a two-machine deployment, copy `config/local-host.yml`, set `ROBOT.IP`, `NUC.IP`, and `PC.IP` to the robot, NUC, and desktop addresses, then run the same commands with that file.

## Quick start

### Minimal Python control

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

### Inspect available controllers

```bash
peirastic.get_controller_list
peirastic.get_controller_info
```

## Control modes

The stack exposes the following controller families.
At the API level, they should be understood as target-following modes: the
target may come from a script, policy, perception module, teleop device, or any
other upstream source.

### OSC modes

- `OSC_POSE`
This controller uses task-space impedance and, by default, simple pose
interpolation such as `LINEAR_POSE` rather than `RUCKIG_POSE`.

Default config files:

- `config/osc-pose-controller.yml`

### Joint-space modes

- `JOINT_POSITION`
- `JOINT_IMPEDANCE`

Default config files:

- `config/joint-position-controller.yml`
- `config/joint-impedance-controller.yml`

### Cartesian tracking mode

- `CARTESIAN_VELOCITY`

This is the Cartesian controller type exposed by the NUC-side stack.
It can be used in two ways:

- direct Cartesian velocity commands
- absolute pose tracking with `RUCKIG_POSE`

Default config file:

- `config/cartesian-velocity-controller.yml`

### Admittance mode

The standalone admittance mode is implemented as a **Python outer
loop** with an **OSC inner loop**.

Runtime config file:

- `config/spacenav-admittance-controller.yml`

## Joint reset examples

### CLI reset script

```bash
peirastic.reset_joints --interface-cfg local-host.yml
```

This script sends `JOINT_POSITION` commands until the target is reached.

### Python: move to a specific joint target

```python
from peirastic import config_root
from peirastic.franka_interface import FrankaInterface

robot = FrankaInterface(f"{config_root}/local-host.yml", use_visualizer=False)

ok = robot.move_joints(
    [0.0, 0.0, 0.0, -1.8, 0.0, 1.8, 0.0],
    timeout=30.0,
)
print("success:", ok)
```

### Python: reset to a named joint target

```python
from peirastic import config_root
from peirastic.franka_interface import FrankaInterface

robot = FrankaInterface(f"{config_root}/local-host.yml", use_visualizer=False)

ok = robot.reset_joints("ready")
print("success:", ok)
```

Built-in named joint targets include:

- `ready`
- `ftcalib_jgroup`

If needed, you can provide your own YAML mapping for named states.

## Cartesian and OSC target following

### Cartesian: absolute pose tracking

`move_pose()` wraps absolute tracking through `CARTESIAN_VELOCITY`.

```python
import numpy as np
from peirastic import config_root
from peirastic.franka_interface import FrankaInterface
from peirastic.utils.config_utils import get_default_controller_config

robot = FrankaInterface(f"{config_root}/local-host.yml", use_visualizer=False)
cfg = get_default_controller_config("CARTESIAN_VELOCITY")

target_pose = np.array([
    0.50, 0.00, 0.30,   # position
    0.0, 0.0, 0.0, 1.0  # quaternion
])

ok = robot.move_pose(target_pose, controller_cfg=cfg, timeout=20.0)
print("success:", ok)
```

### OSC: task-space targets (delta or absolute)

`control("OSC_POSE", action, ...)` uses `controller_cfg["is_delta"]` (from YAML, or
set in code). **Delta** (default in `osc-pose-controller.yml`): `action` is a
small step on top of the current end-effector pose. **Absolute**: set
`cfg["is_delta"] = False`; `action` is `x,y,z` and axis-angle `ax,ay,az` in the
**base frame** (same `Goal` layout as the NUC `OSCImpedanceController`).

For both OSC and `CARTESIAN_VELOCITY`, `FrankaInterface.control()` forwards
`action` in controller-native physical units. The Python API no longer applies
hidden `action_scale` multiplication inside `control()`. If you want joystick /
teleop normalization or per-step clipping, do it explicitly in your own script.

```python
import numpy as np
from peirastic import config_root
from peirastic.franka_interface import FrankaInterface
from peirastic.utils.config_utils import get_default_controller_config

robot = FrankaInterface(f"{config_root}/local-host.yml", use_visualizer=False)
cfg = get_default_controller_config("OSC_POSE")
# anisotropic Kp (per axis); same structure in `osc-pose-controller.yml`
cfg["Kp"]["translation"] = [900.0, 600.0, 300.0]
cfg["Kp"]["rotation"] = [1000.0, 700.0, 400.0]

delta_action = np.array([0.0, 0.0, -0.01, 0.0, 0.0, 0.0, -1.0], dtype=float)
robot.control("OSC_POSE", delta_action, controller_cfg=cfg)

# Absolute: set `is_delta: false` in `osc-pose-controller.yml`, or
cfg_abs = get_default_controller_config("OSC_POSE")
cfg_abs["is_delta"] = False
# ... Kp, etc. ...
# abs_action: base [x,y,z, ax,ay,az, nullspace]; then:
# robot.control("OSC_POSE", abs_action, controller_cfg=cfg_abs)
```

Treat OSC and Cartesian tracking as generic target-following interfaces, not
tied to a specific input device.

## Where to tune parameters

### Network and control loop

- `config/local-host.yml`: robot IP, ZMQ ports, policy/state/traj rate
- `config/control_config.yml`: safety limits, torque limits, speed limits, ZMQ idle behavior

### Cartesian tracking

Tune here:

- `config/cartesian-velocity-controller.yml`

Main fields:

- `traj_interpolator_cfg.traj_interpolator_type`
- `traj_interpolator_cfg.time_fraction`
- `traj_interpolator_config.max_velocity`
- `traj_interpolator_config.max_acceleration`
- `traj_interpolator_config.max_jerk`
- `Kp.translation`
- `Kp.rotation`
- `Kd`
- `action_scale.translation` (script-side helper; not applied automatically by `FrankaInterface.control()`)
- `action_scale.rotation` (script-side helper; not applied automatically by `FrankaInterface.control()`)

### OSC

Tune here:

- `config/osc-pose-controller.yml`

Main fields:

- `Kp.translation`
- `Kp.rotation`
- `residual_mass_vec`
- `traj_interpolator_cfg`
- `action_scale` (script-side helper; not applied automatically by `FrankaInterface.control()`)
- `state_estimator_cfg`

### Joint-space controllers

Tune here:

- `config/joint-position-controller.yml`
- `config/joint-impedance-controller.yml`

### Admittance

`config/spacenav-admittance-controller.yml` (consumed by `spacenav_mode_switch_test.py`):

- **Link / rate**: `interface_cfg`, `control_freq`
- **Start pose**: `init_joint_angles`
- **NetFT calib file**: `netft_calibration.env_key`, `netft_calibration.default_path`
- **Tool frame**: `tool.eef_offset`, `tool.eef_offset_rpy`
- **Teleop inputs**: `teleop.linear_scale`, `teleop.angular_scale`, deadzones and override thresholds
- **Admittance loop** (`admittance` block): `force_target_z`, `mass`, `damping_down`,
  `damping_up`, `force_derivative_gain`, `accel_limit`, `force_deadband`, `force_alpha`,
  `max_admittance_position`, `max_admittance_velocity`, `bias_wait`
- **Contact switching** (`contact` block): make/break thresholds and release delay
- **Guards** (`safety` block): `max_position_error`, `max_rotation_error`
- **OSC inner loop** (`osc` block): `kp_translation`, `kp_rotation`, `residual_mass_vec`,
  `action_scale` (translation / rotation, used by the script as a per-cycle clip),
  `traj_interpolator_cfg` (`traj_interpolator_type`,
  `time_fraction`), `state_estimator_cfg` (estimator type and `alpha_q`, `alpha_dq`, `alpha_eef`, `alpha_eef_vel`)

## Special case: `spacenav_mode_switch_test` and OSC `Kp`

`spacenav_mode_switch_test.py` is a **teleop-style** demo: it overwrites
`Kp` with isotropic values from `--osc-translation-stiffness` /
`--osc-rotation-stiffness`, not the per-axis entries in the YAML. For
anisotropic `Kp`, use the general path in
[OSC: task-space targets (delta or absolute)](#osc-task-space-targets-delta-or-absolute)
instead of this script, or remove the overwrite and set `osc_cfg["Kp"]` yourself.

## Admittance mode

### What it is

The main standalone implementation is:

- `peirastic/scripts/spacenav_mode_switch_test.py`

It accepts:

- SpaceNav teleop input from ROS
- NetFT force input from ROS `/netft_data`
- calibrated force compensation from the configured NetFT YAML
- contact-triggered switching into admittance mode
- OSC for the inner robot command loop

### Important behavior note

The script stays in normal teleop until calibrated force crosses the contact
threshold. In contact, the admittance loop regulates only the tool-Z direction;
tangential motion and orientation remain controlled by SpaceNav.

### Calibration file

The script reads the NetFT calibration YAML from:

- environment variable: `PEIRASTIC_NETFT_CALIB_YAML`
- otherwise default: `~/.local/share/peirastic/netft/config/netft_calib_result.yaml`

The YAML is loaded directly by the Python script; no separate `rosparam load`
step is required.

### Runtime configuration

The standalone admittance runtime parameters live in:

- `config/spacenav-admittance-controller.yml`

### How to run

Prerequisites:

- `franka-interface` running
- `roscore` running
- `/netft_data` available
- calibration YAML generated

Integrated SpaceNav example:

```bash
peirastic.spacenav_mode_switch_test \
  --interface-cfg local-host.yml \
  --admittance-controller-cfg spacenav-admittance-controller.yml
```

## Dynamic mode switching

To run **any** supported `controller_type` in sequence, you switch in **Python** by
calling `FrankaInterface.control()` with a new `controller_type` and the matching
`controller_cfg` from `get_default_controller_config(...)`. You are not limited
to a specific pair of modes; use `peirastic.get_controller_info` to list what
the NUC build exposes.

When you change the active controller, call `robot.bump_control_session()` once
before the first `control()` in the new mode (see the example) so the NUC can
drop stale trajectory state from the old mode. The C++ node now also resets its
internal timing / interpolator runtime state on a new control session, so the
first command in the new mode starts from a clean interpolator state.

Example:

```python
import numpy as np
from peirastic import config_root
from peirastic.franka_interface import FrankaInterface
from peirastic.utils.config_utils import get_default_controller_config

robot = FrankaInterface(f"{config_root}/local-host.yml", use_visualizer=False)
cfg_osc = get_default_controller_config("OSC_POSE")
cfg_cart = get_default_controller_config("CARTESIAN_VELOCITY")
# ... tune cfg_* ...

# Mode A — call repeatedly in your own loop (policy, timer, etc.):
a_osc = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=float)  # placeholder
robot.control("OSC_POSE", a_osc, controller_cfg=cfg_osc)

# Hand off to mode B — bump once, then use the new controller:
robot.bump_control_session()
a_cart = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=float)  # 6D + gripper, placeholder
robot.control("CARTESIAN_VELOCITY", a_cart, controller_cfg=cfg_cart)
```

A SpaceNav **demo** that also toggles modes (optional, input-device specific):

```bash
peirastic.spacenav_mode_switch_test --interface-cfg local-host.yml --launch-controller
```

It is only a reference script, not the architecture; it calls the same
`bump_control_session` path when changing modes.

## Appendix A: force calibration and parameter identification

The NetFT tools write generated data outside the source tree by default. The
default runtime workspace is:

```text
~/.local/share/peirastic/netft
```

You can override it with:

```bash
export PEIRASTIC_NETFT_WORKSPACE=/your/custom/workspace
```

### 1. Acquire calibration data

Example:

```bash
peirastic.netft_data_acquisition ready \
  --interface-cfg local-host.yml \
  --seed-joint-cmds-from-package \
  --with-roslaunch \
  --netft-ip <NETFT_IP> \
  --ros-setup /path/to/open_ws/devel/setup.bash
```

What this uses:

- named joint targets: `ready` / `ftcalib_jgroup`
- waypoint file: `<workspace>/config/joint_cmds.txt`
- acquisition log: `<workspace>/log/write_data_identify.txt`
- joint log: `<workspace>/log/position_data.txt`

### 2. Run identification

```bash
peirastic.netft_identification ~/.local/share/peirastic/netft
```

This writes:

```text
<workspace>/config/netft_calib_result.yaml
```

The output YAML contains:

- `netft_calib_param.Fx0`
- `netft_calib_param.Fy0`
- `netft_calib_param.Fz0`
- `netft_calib_param.Mx0`
- `netft_calib_param.My0`
- `netft_calib_param.Mz0`
- `netft_calib_param.Lx`
- `netft_calib_param.Ly`
- `netft_calib_param.Lz`
- `netft_calib_param.mcx`
- `netft_calib_param.mcy`
- `netft_calib_param.mcz`

### 3. Use the identified calibration

Either:

- export `PEIRASTIC_NETFT_CALIB_YAML=/path/to/netft_calib_result.yaml`

or keep the generated file at the default runtime location so the admittance
script can load it automatically.

### 4. Optional validation

You can use:

```bash
peirastic.netft_pub_calib_result --interface-cfg local-host.yml
```

to publish calibrated force results for inspection.

## Appendix B: useful files

- `config/local-host.yml`
- `config/control_config.yml`
- `config/cartesian-velocity-controller.yml`
- `config/osc-pose-controller.yml`
- `config/joint-position-controller.yml`
- `config/joint-impedance-controller.yml`
- `config/spacenav-admittance-controller.yml`
- `peirastic/franka_interface/franka_interface.py`
- `peirastic/scripts/spacenav_mode_switch_test.py`
- `peirastic/scripts/spacenav_cartesian_min.py`
- `peirastic/scripts/netft_data_acquisition.py`
- `peirastic/scripts/netft_identification.py`
- `peirastic/scripts/netft_pub_calib_result.py`
- `peirastic/netft_calib/identification_core.py`
