#!/usr/bin/env python3
"""SpaceNav teleop with runtime switching between Cartesian tracking and OSC."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import List

import numpy as np
import rosgraph
import rospy
from geometry_msgs.msg import Twist, WrenchStamped
from sensor_msgs.msg import Joy

from peirastic import ROOT_PATH, config_root
from peirastic.franka_interface import FrankaInterface
from peirastic.utils.config_utils import get_default_controller_config, verify_controller_config
from peirastic.utils.admittance_utils import (
    calibrate_tool_z_force,
    canonicalize_quaternion,
    ft_sensor_frame,
    integrate_body_rotation,
    load_netft_calib_param,
)
from peirastic.utils.runtime_paths import get_default_netft_calib_yaml
from peirastic.utils import transform_utils
from peirastic.utils.yaml_config import YamlConfig

INIT_JOINT_ANGLES = [0.0, 0.0, 0.0, -1.8, 0.0, 1.8, 0.0]

MODE_CARTESIAN = "CARTESIAN_TRACKING"
MODE_OSC = "OSC_POSE"
MODE_ADMITTANCE = "ADMITTANCE"
MODES = [MODE_CARTESIAN, MODE_OSC, MODE_ADMITTANCE]

_lock = threading.Lock()
_sn_lin = np.zeros(3, dtype=np.float64)
_sn_ang = np.zeros(3, dtype=np.float64)
_raw_ft = np.zeros(6, dtype=np.float64)
_sn_lin_bias = np.zeros(3, dtype=np.float64)
_sn_ang_bias = np.zeros(3, dtype=np.float64)
_last_twist_time = 0.0
_force_ready = False
_reset_requested = False
_toggle_requested = False
_btn0_prev = 0
_btn1_prev = 0


def _twist_cb(msg: Twist) -> None:
    global _sn_lin, _sn_ang, _last_twist_time
    with _lock:
        _sn_lin = np.array([msg.linear.x, msg.linear.y, msg.linear.z], dtype=np.float64)
        _sn_ang = np.array([msg.angular.x, msg.angular.y, msg.angular.z], dtype=np.float64)
        _last_twist_time = time.monotonic()


def _netft_cb(msg: WrenchStamped) -> None:
    global _raw_ft, _force_ready
    with _lock:
        _raw_ft = np.array(
            [
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
                msg.wrench.torque.x,
                msg.wrench.torque.y,
                msg.wrench.torque.z,
            ],
            dtype=np.float64,
        )
        _force_ready = True


def _joy_cb(msg: Joy) -> None:
    global _reset_requested, _toggle_requested, _btn0_prev, _btn1_prev
    buttons = msg.buttons
    cur0 = int(buttons[0]) if len(buttons) > 0 else 0
    cur1 = int(buttons[1]) if len(buttons) > 1 else 0
    with _lock:
        if cur0 == 1 and _btn0_prev == 0:
            _toggle_requested = True
        if cur1 == 1 and _btn1_prev == 0:
            _reset_requested = True
        _btn0_prev = cur0
        _btn1_prev = cur1


def _resolve_config_path(cfg_name: str) -> str:
    return cfg_name if cfg_name.startswith("/") else f"{config_root}/{cfg_name}"


def _load_controller_cfg(controller_type: str, cfg_path: str):
    if cfg_path:
        cfg = verify_controller_config(YamlConfig(_resolve_config_path(cfg_path)).as_easydict())
    else:
        cfg = get_default_controller_config(controller_type)
    return cfg


def _existing_franka_interface_processes() -> List[str]:
    try:
        result = subprocess.run(
            ["pgrep", "-af", "franka-interface"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        return []
    return [
        line
        for line in result.stdout.splitlines()
        if "pgrep" not in line and "franka-interface" in line
    ]


def _launch_local_controller(
    interface_cfg: str,
    control_cfg: str,
    log_path: str,
    allow_existing: bool,
) -> int:
    existing = _existing_franka_interface_processes()
    if existing and not allow_existing:
        raise RuntimeError(
            "franka-interface is already running. Reuse it by removing "
            "`--launch-controller`, or stop the stale process before launching a new one:\n"
            + "\n".join(existing)
        )

    repo_root = Path(ROOT_PATH).resolve().parent
    launcher = repo_root / "scripts" / "run_franka_interface.sh"
    if not launcher.exists():
        raise FileNotFoundError(f"Launcher not found: {launcher}")

    cmd = [str(launcher), interface_cfg]
    if control_cfg:
        cmd.append(control_cfg)

    log_file = Path(os.path.expanduser(log_path)).resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "ab") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return int(proc.pid)


def _canonicalize_quaternion(quat, reference_quat=None):
    quat = np.asarray(quat, dtype=np.float64).reshape(4).copy()
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    quat /= norm
    if reference_quat is not None:
        reference_quat = np.asarray(reference_quat, dtype=np.float64).reshape(4)
        if float(np.dot(quat, reference_quat)) < 0.0:
            quat *= -1.0
    elif quat[3] < 0.0:
        quat *= -1.0
    return quat


def _integrate_absolute_target(
    target_pos,
    target_quat,
    sn_lin,
    sn_ang,
    dt: float,
    linear_scale: float,
    angular_scale: float,
):
    next_pos = np.asarray(target_pos, dtype=np.float64).copy()
    next_quat = np.asarray(target_quat, dtype=np.float64).copy()

    next_pos += sn_lin * linear_scale * dt
    drot = np.array([sn_ang[0], -sn_ang[1], -sn_ang[2]], dtype=np.float64) * angular_scale * dt
    if np.linalg.norm(drot) > 0.0:
        delta_quat = transform_utils.axisangle2quat(drot)
        next_quat = transform_utils.quat_multiply(next_quat, delta_quat).astype(np.float64)
        next_quat = _canonicalize_quaternion(next_quat, reference_quat=target_quat)

    return next_pos, next_quat


def _build_absolute_action(target_pos, target_quat):
    canonical_quat = _canonicalize_quaternion(target_quat)
    return np.concatenate([np.asarray(target_pos, dtype=np.float64), canonical_quat, [-1.0]])


def _clip_vector_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if max_norm <= 0.0 or norm <= max_norm or norm <= 1e-12:
        return vec
    return vec * (max_norm / norm)


def _build_osc_delta_action(
    current_pose: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    max_pos_delta: float,
    max_rot_delta: float,
):
    current_pos = current_pose[:3, 3].astype(np.float64).copy()
    current_quat = _canonicalize_quaternion(
        transform_utils.mat2quat(current_pose[:3, :3]).astype(np.float64).copy()
    )

    delta_pos = np.asarray(target_pos, dtype=np.float64) - current_pos
    delta_pos = _clip_vector_norm(delta_pos, max_pos_delta)

    delta_quat = transform_utils.quat_distance(
        _canonicalize_quaternion(target_quat, reference_quat=current_quat),
        current_quat,
    ).astype(np.float64)
    delta_quat = _canonicalize_quaternion(delta_quat)
    delta_rot = transform_utils.quat2axisangle(delta_quat).astype(np.float64)
    delta_rot = _clip_vector_norm(delta_rot, max_rot_delta)

    return np.concatenate([delta_pos, delta_rot, [-1.0]])


def _step_command_target(
    command_pos: np.ndarray,
    command_quat: np.ndarray,
    desired_pos: np.ndarray,
    desired_quat: np.ndarray,
    max_pos_step: float,
    max_rot_step: float,
):
    next_pos = command_pos + _clip_vector_norm(desired_pos - command_pos, max_pos_step)

    desired_quat = _canonicalize_quaternion(desired_quat, reference_quat=command_quat)
    delta_quat = transform_utils.quat_distance(desired_quat, command_quat).astype(np.float64)
    delta_quat = _canonicalize_quaternion(delta_quat)
    delta_rot = transform_utils.quat2axisangle(delta_quat).astype(np.float64)
    delta_angle = float(np.linalg.norm(delta_rot))
    if max_rot_step <= 0.0 or delta_angle <= max_rot_step or delta_angle <= 1e-12:
        next_quat = desired_quat
    else:
        step_fraction = max_rot_step / delta_angle
        next_quat = transform_utils.quat_slerp(
            command_quat.astype(np.float64),
            desired_quat.astype(np.float64),
            step_fraction,
        ).astype(np.float64)
        next_quat = _canonicalize_quaternion(next_quat, reference_quat=command_quat)

    pos_error = float(np.linalg.norm(desired_pos - next_pos))
    rot_error = float(
        np.linalg.norm(
            transform_utils.quat2axisangle(
                _canonicalize_quaternion(
                    transform_utils.quat_distance(desired_quat, next_quat).astype(np.float64)
                )
            ).astype(np.float64)
        )
    )
    return next_pos, next_quat, pos_error, rot_error


def _capture_pose_target(robot: FrankaInterface):
    current_pose = robot.last_eef_pose
    if current_pose is None:
        return None, None
    target_pos = current_pose[:3, 3].astype(np.float64).copy()
    target_quat = _canonicalize_quaternion(
        transform_utils.mat2quat(current_pose[:3, :3]).astype(np.float64).copy()
    )
    return target_pos, target_quat


def _current_tool_pose(robot: FrankaInterface, eef_offset: np.ndarray, eef_offset_rot: np.ndarray):
    current_pose = robot.last_eef_pose
    if current_pose is None:
        return None, None, None

    link8_rot = current_pose[:3, :3]
    link8_pos = current_pose[:3, 3].astype(np.float64).copy()
    tool_pos = link8_pos + link8_rot @ eef_offset
    tool_rot = link8_rot @ eef_offset_rot
    tool_quat = canonicalize_quaternion(transform_utils.mat2quat(tool_rot).astype(np.float64))
    return tool_pos.astype(np.float64), tool_quat, link8_rot


def _build_admittance_osc_cfg(runtime_cfg):
    cfg = get_default_controller_config("OSC_POSE")
    cfg["is_delta"] = True
    cfg["Kp"] = {
        "translation": [float(v) for v in runtime_cfg.osc.kp_translation],
        "rotation": [float(v) for v in runtime_cfg.osc.kp_rotation],
    }
    cfg["action_scale"] = {
        "translation": float(runtime_cfg.osc.action_scale.translation),
        "rotation": float(runtime_cfg.osc.action_scale.rotation),
    }
    cfg["residual_mass_vec"] = [float(v) for v in runtime_cfg.osc.residual_mass_vec]
    cfg["traj_interpolator_cfg"] = {
        "traj_interpolator_type": runtime_cfg.osc.traj_interpolator_cfg.traj_interpolator_type,
        "time_fraction": float(runtime_cfg.osc.traj_interpolator_cfg.time_fraction),
    }
    cfg["state_estimator_cfg"] = {
        "is_estimation": bool(runtime_cfg.osc.state_estimator_cfg.is_estimation),
        "state_estimator_type": runtime_cfg.osc.state_estimator_cfg.state_estimator_type,
        "alpha_q": float(runtime_cfg.osc.state_estimator_cfg.alpha_q),
        "alpha_dq": float(runtime_cfg.osc.state_estimator_cfg.alpha_dq),
        "alpha_eef": float(runtime_cfg.osc.state_estimator_cfg.alpha_eef),
        "alpha_eef_vel": float(runtime_cfg.osc.state_estimator_cfg.alpha_eef_vel),
    }
    return cfg


def _load_netft_calibration(calib_yaml: str):
    calib_yaml = os.path.abspath(os.path.expanduser(calib_yaml))
    if not os.path.isfile(calib_yaml):
        raise FileNotFoundError(f"Calibration file not found: {calib_yaml}")

    cfg = YamlConfig(calib_yaml).as_easydict()
    calib_param = cfg.netft_calib_param if "netft_calib_param" in cfg else cfg
    return load_netft_calib_param(calib_param)


def _capture_admittance_force_bias(
    robot: FrankaInterface,
    controller_cfg,
    eef_offset: np.ndarray,
    eef_offset_rot: np.ndarray,
    calib_params,
    sensor_rot_z_rad: float,
    sensor_tz: float,
    ftf: np.ndarray,
    alpha_force: float,
    bias_wait: float,
    control_freq: float,
):
    hold_action = np.concatenate([np.zeros(6, dtype=np.float64), [-1.0]])
    for _ in range(30):
        robot.control("OSC_POSE", hold_action, controller_cfg=controller_cfg)
        time.sleep(0.02)

    force_samples = []
    filtered_fz = None
    dt = 1.0 / control_freq
    deadline = time.monotonic() + max(0.0, bias_wait)
    while time.monotonic() < deadline and not rospy.is_shutdown():
        _, _, link8_rot = _current_tool_pose(robot, eef_offset, eef_offset_rot)
        with _lock:
            raw_ft = _raw_ft.copy()
            have_force = _force_ready
        if have_force and link8_rot is not None:
            link8_rpy = transform_utils.mat2euler(link8_rot, axes="sxyz")
            sensor_rpy = ft_sensor_frame(link8_rpy, sensor_rot_z_rad, sensor_tz)
            fz_cal = calibrate_tool_z_force(raw_ft, sensor_rpy, calib_params, ftf)
            filtered_fz = (
                fz_cal if filtered_fz is None else alpha_force * filtered_fz + (1.0 - alpha_force) * fz_cal
            )
            force_samples.append(fz_cal)
        robot.control("OSC_POSE", hold_action, controller_cfg=controller_cfg)
        time.sleep(dt)

    if not force_samples:
        return 0.0, 0.0, 0
    force_bias = float(np.mean(force_samples))
    return force_bias, force_bias, len(force_samples)


def _wait_for_force_data(timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline and not rospy.is_shutdown():
        with _lock:
            if _force_ready:
                return True
        time.sleep(0.02)
    with _lock:
        return bool(_force_ready)


def _poll_spacenav_state():
    global _reset_requested, _toggle_requested
    with _lock:
        sn_lin = _sn_lin.copy()
        sn_ang = _sn_ang.copy()
        last_twist_time = _last_twist_time
        do_reset = _reset_requested
        do_toggle = _toggle_requested
        _reset_requested = False
        _toggle_requested = False
    return sn_lin, sn_ang, do_reset, do_toggle, last_twist_time


def _capture_spacenav_bias():
    global _sn_lin_bias, _sn_ang_bias
    with _lock:
        _sn_lin_bias = _sn_lin.copy()
        _sn_ang_bias = _sn_ang.copy()
    return _sn_lin_bias.copy(), _sn_ang_bias.copy()


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface-cfg", default="local-host.yml")
    parser.add_argument("--cartesian-controller-cfg", default="")
    parser.add_argument("--osc-controller-cfg", default="")
    parser.add_argument(
        "--admittance-controller-cfg",
        default="spacenav-admittance-controller.yml",
        help="Admittance runtime YAML under config/ or an absolute path.",
    )
    parser.add_argument(
        "--netft-calib-yaml",
        default=os.environ.get("PEIRASTIC_NETFT_CALIB_YAML", get_default_netft_calib_yaml()),
        help="NetFT calibration YAML read directly by this script.",
    )
    parser.add_argument(
        "--netft-topic",
        default="/netft_data",
        help="Raw NetFT WrenchStamped topic. This script calibrates the raw wrench inline.",
    )
    parser.add_argument(
        "--force-wait-timeout",
        type=float,
        default=3.0,
        help="Seconds to wait for NetFT data before enabling admittance mode.",
    )
    parser.add_argument("--control-freq", type=float, default=250.0)
    parser.add_argument("--linear-scale", type=float, default=0.08)
    parser.add_argument("--angular-scale", type=float, default=0.18)
    parser.add_argument("--traj-time-fraction", type=float, default=0.4)
    parser.add_argument("--linear-deadzone", type=float, default=0.01)
    parser.add_argument("--angular-deadzone", type=float, default=0.01)
    parser.add_argument("--spacenav-timeout", type=float, default=5.0)
    parser.add_argument("--state-timeout", type=float, default=30.0)
    parser.add_argument("--init-timeout", type=float, default=120.0)
    parser.add_argument("--reset-on-start", action="store_true")
    parser.add_argument(
        "--button1-reset-origin",
        action="store_true",
        help="Make SpaceNav button1 physically reset to INIT_JOINT_ANGLES.",
    )
    parser.add_argument("--launch-controller", action="store_true")
    parser.add_argument(
        "--control-cfg",
        default="spacenav-debug-control.yml",
        help="Control safety YAML under config/ used only with --launch-controller.",
    )
    parser.add_argument(
        "--allow-existing-controller",
        action="store_true",
        help="Allow --launch-controller even if a franka-interface process is already running.",
    )
    parser.add_argument("--controller-log", default="/tmp/peirastic-franka-interface.log")
    parser.add_argument("--terminate-at-exit", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--command-max-pos-step",
        type=float,
        default=0.008,
        help="Shared per-cycle desired->command translation step in meters.",
    )
    parser.add_argument(
        "--command-max-rot-step",
        type=float,
        default=0.08,
        help="Shared per-cycle desired->command rotation step in radians.",
    )
    parser.add_argument(
        "--osc-max-pos-delta",
        type=float,
        default=0.010,
        help="Per-cycle OSC delta translation clip in meters.",
    )
    parser.add_argument(
        "--osc-max-rot-delta",
        type=float,
        default=0.08,
        help="Per-cycle OSC delta rotation clip in radians.",
    )
    parser.add_argument(
        "--osc-translation-stiffness",
        type=float,
        default=700.0,
        help="Per-axis OSC translational stiffness used by this safety test script.",
    )
    parser.add_argument(
        "--osc-rotation-stiffness",
        type=float,
        default=500.0,
        help="Per-axis OSC rotational stiffness used by this safety test script.",
    )
    parser.add_argument(
        "--admittance-debug-interval",
        type=float,
        default=0.25,
        help="Seconds between admittance debug prints; set <=0 to disable.",
    )
    parser.add_argument("--debug-input", action="store_true")
    return parser.parse_args()


def _info(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    args = _parse_args()
    interface_cfg = _resolve_config_path(args.interface_cfg)
    control_cfg = _resolve_config_path(args.control_cfg) if args.control_cfg else ""

    cartesian_cfg = _load_controller_cfg("CARTESIAN_VELOCITY", args.cartesian_controller_cfg)
    cartesian_cfg["is_delta"] = False
    cartesian_cfg["traj_interpolator_cfg"]["time_fraction"] = max(
        0.1, float(args.traj_time_fraction)
    )

    osc_cfg = _load_controller_cfg("OSC_POSE", args.osc_controller_cfg)
    osc_cfg["is_delta"] = True
    osc_cfg["action_scale"]["translation"] = 1.0
    osc_cfg["action_scale"]["rotation"] = 1.0
    osc_cfg["Kp"]["translation"] = [float(args.osc_translation_stiffness)] * 3
    osc_cfg["Kp"]["rotation"] = [float(args.osc_rotation_stiffness)] * 3

    admittance_runtime = YamlConfig(
        _resolve_config_path(args.admittance_controller_cfg)
    ).as_easydict()
    admittance_cfg = _build_admittance_osc_cfg(admittance_runtime)
    eef_offset = np.array(admittance_runtime.tool.eef_offset, dtype=np.float64)
    eef_offset_rot = transform_utils.euler2mat(
        np.array(admittance_runtime.tool.eef_offset_rpy, dtype=np.float64)
    )
    force_target_z = float(admittance_runtime.admittance.force_target_z)
    admittance_mass = float(admittance_runtime.admittance.mass)
    damping_down = float(admittance_runtime.admittance.damping_down)
    damping_up = float(admittance_runtime.admittance.damping_up)
    force_derivative_gain = float(admittance_runtime.admittance.force_derivative_gain)
    accel_limit = float(admittance_runtime.admittance.accel_limit)
    force_deadband = float(admittance_runtime.admittance.force_deadband)
    force_alpha = float(admittance_runtime.admittance.force_alpha)
    max_admittance_position = float(admittance_runtime.admittance.max_admittance_position)
    max_admittance_velocity = float(admittance_runtime.admittance.max_admittance_velocity)
    bias_wait = float(admittance_runtime.admittance.bias_wait)
    contact_make = float(admittance_runtime.contact.make_threshold)
    contact_break_slow = float(admittance_runtime.contact.break_slow_threshold)
    contact_break_fast = float(admittance_runtime.contact.break_fast_threshold)
    contact_break_delay = float(admittance_runtime.contact.break_delay)
    user_pull_up_threshold = float(admittance_runtime.teleop.user_pull_up_threshold)
    user_override_threshold = float(admittance_runtime.teleop.user_override_threshold)
    max_position_error = float(admittance_runtime.safety.max_position_error)
    max_rotation_error = float(admittance_runtime.safety.max_rotation_error)
    kp_z_free = float(admittance_runtime.osc.kp_z.free_space)
    kp_z_contact = float(admittance_runtime.osc.kp_z.contact)
    kp_z_alpha = float(admittance_runtime.osc.kp_z.smoothing_alpha)

    _info("Checking ROS master...")
    if not rosgraph.is_master_online():
        print("ROS master is not reachable. Start roscore first.", file=sys.stderr, flush=True)
        return 1

    _info("Connecting to ROS node...")
    rospy.init_node("spacenav_mode_switch_test", anonymous=False)
    rospy.Subscriber("/spacenav/twist", Twist, _twist_cb, queue_size=1)
    rospy.Subscriber("/spacenav/joy", Joy, _joy_cb, queue_size=1)
    rospy.Subscriber(args.netft_topic, WrenchStamped, _netft_cb, queue_size=1, tcp_nodelay=True)

    try:
        calib_params, sensor_rot_z_rad, sensor_tz, ftf = _load_netft_calibration(
            args.netft_calib_yaml
        )
    except Exception as exc:
        print(f"Failed to load NetFT calibration for admittance mode: {exc}", file=sys.stderr)
        return 2

    try:
        _info("Waiting for /spacenav/twist messages...")
        rospy.wait_for_message("/spacenav/twist", Twist, timeout=args.spacenav_timeout)
    except rospy.ROSException:
        print("Timeout: no /spacenav/twist messages. Start roscore and spacenav_node first.", file=sys.stderr)
        return 3

    if args.launch_controller:
        try:
            _info("Launching local franka-interface...")
            pid = _launch_local_controller(
                interface_cfg,
                control_cfg,
                args.controller_log,
                args.allow_existing_controller,
            )
        except Exception as exc:
            print(f"Failed to launch local franka-interface: {exc}", file=sys.stderr)
            return 4
        _info(f"Launched local franka-interface (pid={pid}).")
        time.sleep(1.0)

    _info("Connecting to franka-interface...")
    robot = FrankaInterface(
        interface_cfg,
        control_freq=args.control_freq,
        use_visualizer=False,
        automatic_gripper_reset=False,
    )

    active_mode = MODE_CARTESIAN
    steps = 0
    last_debug_time = 0.0
    last_admittance_debug_time = 0.0
    force_send = True
    dt = 1.0 / args.control_freq

    try:
        _info("Waiting for robot state...")
        if not robot.wait_for_state(timeout=args.state_timeout):
            print("Timeout: no robot state from local franka-interface.", file=sys.stderr)
            return 4

        if args.reset_on_start:
            _info("Resetting robot to INIT_JOINT_ANGLES...")
            if not robot.move_joints(INIT_JOINT_ANGLES, timeout=args.init_timeout):
                print("Failed to reach INIT_JOINT_ANGLES.", file=sys.stderr)
                return 5
        else:
            _info("Skipping startup joint reset; using current robot pose as teleop reference.")

        bias_lin, bias_ang = _capture_spacenav_bias()
        desired_pos, desired_quat = _capture_pose_target(robot)
        if desired_pos is None or desired_quat is None:
            print("Robot pose is unavailable after startup initialization.", file=sys.stderr, flush=True)
            return 6
        command_pos = desired_pos.copy()
        command_quat = desired_quat.copy()

        admittance_pos, admittance_quat, _ = _current_tool_pose(
            robot, eef_offset, eef_offset_rot
        )
        if admittance_pos is None or admittance_quat is None:
            print("Robot tool pose is unavailable after startup initialization.", file=sys.stderr)
            return 7
        admittance_axis_base = transform_utils.quat2mat(admittance_quat)[:, 2].copy()
        admittance_axis_base /= max(float(np.linalg.norm(admittance_axis_base)), 1e-12)
        admittance_anchor_normal = float(np.dot(admittance_pos, admittance_axis_base))
        admittance_offset = 0.0
        admittance_vel = 0.0
        filtered_fz = None
        force_bias = 0.0
        fz_prev = 0.0
        is_in_contact = False
        break_contact_timer = 0.0
        current_kp_z = kp_z_free

        force_bias, filtered_fz, force_sample_count = _capture_admittance_force_bias(
            robot,
            admittance_cfg,
            eef_offset,
            eef_offset_rot,
            calib_params,
            sensor_rot_z_rad,
            sensor_tz,
            ftf,
            force_alpha,
            bias_wait,
            args.control_freq,
        )
        if force_sample_count > 0:
            _info(
                f"NetFT force bias captured: Fz={force_bias:.3f} N "
                f"from {force_sample_count} samples."
            )
        else:
            _info(
                f"No samples on {args.netft_topic} during bias capture; "
                "admittance mode waits for force data."
            )

        _info(
            "Captured SpaceNav neutral bias: "
            f"lin={np.round(bias_lin, 6).tolist()} ang={np.round(bias_ang, 6).tolist()}"
        )
        _info(
            "Ready: button0 cycles CARTESIAN_TRACKING/OSC_POSE/ADMITTANCE; "
            "button1 re-anchors current pose unless --button1-reset-origin is set."
        )

        rate = rospy.Rate(args.control_freq)

        while not rospy.is_shutdown():
            sn_lin, sn_ang, do_reset, do_toggle, last_twist_time = _poll_spacenav_state()
            raw_sn_lin = sn_lin - _sn_lin_bias
            raw_sn_ang = sn_ang - _sn_ang_bias
            sn_lin = raw_sn_lin.copy()
            sn_ang = raw_sn_ang.copy()

            if do_toggle:
                active_mode = MODES[(MODES.index(active_mode) + 1) % len(MODES)]
                current_pos, current_quat = _capture_pose_target(robot)
                tool_pos, tool_quat, _ = _current_tool_pose(robot, eef_offset, eef_offset_rot)
                if current_pos is not None and current_quat is not None:
                    desired_pos = current_pos.copy()
                    desired_quat = current_quat.copy()
                    command_pos = current_pos.copy()
                    command_quat = current_quat.copy()
                if tool_pos is not None and tool_quat is not None:
                    admittance_pos = tool_pos.copy()
                    admittance_quat = tool_quat.copy()
                    admittance_axis_base = transform_utils.quat2mat(tool_quat)[:, 2].copy()
                    admittance_axis_base /= max(float(np.linalg.norm(admittance_axis_base)), 1e-12)
                    admittance_anchor_normal = float(np.dot(admittance_pos, admittance_axis_base))
                    admittance_offset = 0.0
                    admittance_vel = 0.0
                    break_contact_timer = 0.0
                    current_kp_z = kp_z_free
                robot.bump_control_session()
                # Ensure the next target controller message is not dropped by policy-rate
                # throttling right after a bursty mode switch, and re-run the NO_CONTROL
                # handshake the next time controller_type actually changes in control().
                robot.reset_control_rate_limiter()
                robot.force_next_control_preprocess()
                if active_mode == MODE_ADMITTANCE:
                    if not _wait_for_force_data(args.force_wait_timeout):
                        _info(
                            "Admittance mode needs force data, but no WrenchStamped messages "
                            f"arrived on {args.netft_topic}. Start netft_utils/netft_node "
                            "so it publishes /netft_data, then switch to ADMITTANCE again."
                        )
                        active_mode = MODE_OSC
                        force_send = True
                        _info("Switched back to OSC_POSE because NetFT data is unavailable.")
                        continue
                    _info("Initializing admittance mode: holding current pose and capturing force bias.")
                    force_bias, filtered_fz, force_sample_count = _capture_admittance_force_bias(
                        robot,
                        admittance_cfg,
                        eef_offset,
                        eef_offset_rot,
                        calib_params,
                        sensor_rot_z_rad,
                        sensor_tz,
                        ftf,
                        force_alpha,
                        bias_wait,
                        args.control_freq,
                    )
                    tool_pos, tool_quat, _ = _current_tool_pose(
                        robot, eef_offset, eef_offset_rot
                    )
                    if tool_pos is not None and tool_quat is not None:
                        admittance_pos = tool_pos.copy()
                        admittance_quat = tool_quat.copy()
                        admittance_axis_base = transform_utils.quat2mat(tool_quat)[:, 2].copy()
                        admittance_axis_base /= max(float(np.linalg.norm(admittance_axis_base)), 1e-12)
                        admittance_anchor_normal = float(np.dot(admittance_pos, admittance_axis_base))
                        admittance_offset = 0.0
                    admittance_vel = 0.0
                    fz_prev = 0.0
                    is_in_contact = False
                    break_contact_timer = 0.0
                    current_kp_z = kp_z_free
                    _info(
                        "Admittance initialized: "
                        f"target_fz={force_target_z:.3f}N "
                        f"bias_fz={force_bias:.3f}N "
                        f"samples={force_sample_count} "
                        f"make_threshold={contact_make:.3f}N."
                    )
                force_send = True
                _info(f"Switched control mode to {active_mode}.")

            if do_reset:
                if not args.button1_reset_origin:
                    desired_pos, desired_quat = _capture_pose_target(robot)
                    if desired_pos is None or desired_quat is None:
                        print("Robot pose is unavailable for re-anchor.", file=sys.stderr)
                        return 8
                    command_pos = desired_pos.copy()
                    command_quat = desired_quat.copy()
                    tool_pos, tool_quat, _ = _current_tool_pose(robot, eef_offset, eef_offset_rot)
                    if tool_pos is not None and tool_quat is not None:
                        admittance_pos = tool_pos.copy()
                        admittance_quat = tool_quat.copy()
                        admittance_axis_base = transform_utils.quat2mat(tool_quat)[:, 2].copy()
                        admittance_axis_base /= max(float(np.linalg.norm(admittance_axis_base)), 1e-12)
                        admittance_anchor_normal = float(np.dot(admittance_pos, admittance_axis_base))
                        admittance_offset = 0.0
                        admittance_vel = 0.0
                        is_in_contact = False
                        break_contact_timer = 0.0
                        current_kp_z = kp_z_free
                    robot.bump_control_session()
                    robot.reset_control_rate_limiter()
                    robot.force_next_control_preprocess()
                    force_send = True
                    _info("Re-anchored target to current pose; no joint reset.")
                    rate.sleep()
                    continue

                if not robot.move_joints(INIT_JOINT_ANGLES, timeout=args.init_timeout):
                    print("Failed to reset to INIT_JOINT_ANGLES.", file=sys.stderr)
                    return 7
                desired_pos, desired_quat = _capture_pose_target(robot)
                if desired_pos is None or desired_quat is None:
                    print("Robot pose is unavailable after reset.", file=sys.stderr)
                    return 8
                command_pos = desired_pos.copy()
                command_quat = desired_quat.copy()
                tool_pos, tool_quat, _ = _current_tool_pose(robot, eef_offset, eef_offset_rot)
                if tool_pos is not None and tool_quat is not None:
                    admittance_pos = tool_pos.copy()
                    admittance_quat = tool_quat.copy()
                    admittance_axis_base = transform_utils.quat2mat(tool_quat)[:, 2].copy()
                    admittance_axis_base /= max(float(np.linalg.norm(admittance_axis_base)), 1e-12)
                    admittance_anchor_normal = float(np.dot(admittance_pos, admittance_axis_base))
                    admittance_offset = 0.0
                    admittance_vel = 0.0
                    is_in_contact = False
                    break_contact_timer = 0.0
                    current_kp_z = kp_z_free
                force_send = True
                rate.sleep()
                continue

            if np.linalg.norm(sn_lin) < args.linear_deadzone:
                sn_lin[:] = 0.0
            if np.linalg.norm(sn_ang) < args.angular_deadzone:
                sn_ang[:] = 0.0

            now = time.monotonic()
            if args.debug_input and now - last_debug_time >= 0.25:
                input_age = max(0.0, now - last_twist_time) if last_twist_time > 0.0 else float("inf")
                _info(
                    "SpaceNav input: "
                    f"mode={active_mode} "
                    f"raw_lin={np.linalg.norm(raw_sn_lin):.6f} "
                    f"raw_ang={np.linalg.norm(raw_sn_ang):.6f} "
                    f"filtered_lin={np.linalg.norm(sn_lin):.6f} "
                    f"filtered_ang={np.linalg.norm(sn_ang):.6f} "
                    f"msg_age={input_age:.3f}s"
                )
                last_debug_time = now

            input_active = np.linalg.norm(sn_lin) > 1e-12 or np.linalg.norm(sn_ang) > 1e-12
            if active_mode != MODE_ADMITTANCE and input_active:
                desired_pos, desired_quat = _integrate_absolute_target(
                    desired_pos,
                    desired_quat,
                    sn_lin,
                    sn_ang,
                    dt,
                    args.linear_scale,
                    args.angular_scale,
                )
                force_send = True

            current_pose = robot.last_eef_pose
            if current_pose is None:
                rate.sleep()
                continue

            admittance_have_force = False
            admittance_f_error = 0.0
            admittance_action_z = 0.0
            if active_mode == MODE_CARTESIAN:
                command_pos, command_quat, target_pos_error, target_rot_error = _step_command_target(
                    command_pos,
                    command_quat,
                    desired_pos,
                    desired_quat,
                    args.command_max_pos_step,
                    args.command_max_rot_step,
                )
                target_pending = target_pos_error > 1e-5 or target_rot_error > 1e-4
                if force_send or target_pending or args.max_steps == 0 or steps < args.max_steps:
                    action = _build_absolute_action(command_pos, command_quat)
                    robot.control("CARTESIAN_VELOCITY", action, controller_cfg=cartesian_cfg)
                    steps += 1
            elif active_mode == MODE_OSC:
                command_pos, command_quat, target_pos_error, target_rot_error = _step_command_target(
                    command_pos,
                    command_quat,
                    desired_pos,
                    desired_quat,
                    args.command_max_pos_step,
                    args.command_max_rot_step,
                )
                target_pending = target_pos_error > 1e-5 or target_rot_error > 1e-4
                action = _build_osc_delta_action(
                    current_pose,
                    command_pos,
                    command_quat,
                    args.osc_max_pos_delta,
                    args.osc_max_rot_delta,
                )
                delta_norm = max(np.linalg.norm(action[:3]), np.linalg.norm(action[3:6]))
                if force_send or target_pending or delta_norm > 1e-5:
                    robot.control("OSC_POSE", action, controller_cfg=osc_cfg)
                    steps += 1
            else:
                tool_pos, tool_quat, link8_rot = _current_tool_pose(
                    robot, eef_offset, eef_offset_rot
                )
                if tool_pos is None or tool_quat is None or link8_rot is None:
                    rate.sleep()
                    continue

                with _lock:
                    raw_ft = _raw_ft.copy()
                    have_force = _force_ready
                admittance_have_force = have_force

                if have_force:
                    link8_rpy = transform_utils.mat2euler(link8_rot, axes="sxyz")
                    sensor_rpy = ft_sensor_frame(link8_rpy, sensor_rot_z_rad, sensor_tz)
                    fz_cal = calibrate_tool_z_force(raw_ft, sensor_rpy, calib_params, ftf)
                    filtered_fz = force_alpha * filtered_fz + (1.0 - force_alpha) * fz_cal

                fz_comp = float(filtered_fz - force_bias)
                user_pull_up = sn_lin[2] > user_pull_up_threshold
                user_override = sn_lin[2] > user_override_threshold

                admittance_pos[0] += sn_lin[0] * args.linear_scale * dt
                admittance_pos[1] += sn_lin[1] * args.linear_scale * dt
                admittance_pos[2] += sn_lin[2] * args.linear_scale * dt
                admittance_quat = integrate_body_rotation(
                    admittance_quat,
                    np.array([sn_ang[0], -sn_ang[1], -sn_ang[2]], dtype=np.float64)
                    * args.angular_scale,
                    dt,
                )
                if not is_in_contact:
                    admittance_axis_base = transform_utils.quat2mat(admittance_quat)[:, 2].copy()
                    axis_norm = float(np.linalg.norm(admittance_axis_base))
                    if axis_norm > 1e-12:
                        admittance_axis_base /= axis_norm
                    admittance_anchor_normal = float(
                        np.dot(admittance_pos, admittance_axis_base)
                    )

                if have_force and not is_in_contact and fz_comp < contact_make and not user_override:
                    is_in_contact = True
                    admittance_anchor_normal = float(
                        np.dot(admittance_pos, admittance_axis_base)
                    )
                    admittance_offset = 0.0
                    admittance_vel = 0.0
                    break_contact_timer = 0.0
                    _info(f"Contact detected: Fz={fz_comp:.3f} N, admittance active.")
                elif is_in_contact:
                    if user_override:
                        is_in_contact = False
                        admittance_pos = tool_pos.copy()
                        admittance_offset = 0.0
                        admittance_vel = 0.0
                        break_contact_timer = 0.0
                        _info("Contact released by SpaceNav Z override.")
                    elif have_force and fz_comp > contact_break_fast:
                        is_in_contact = False
                        admittance_pos = tool_pos.copy()
                        admittance_offset = 0.0
                        admittance_vel = 0.0
                        break_contact_timer = 0.0
                        _info(f"Contact released: Fz={fz_comp:.3f} N.")
                    elif have_force and fz_comp > contact_break_slow:
                        break_contact_timer += dt
                        if break_contact_timer > contact_break_delay:
                            is_in_contact = False
                            admittance_pos = tool_pos.copy()
                            admittance_offset = 0.0
                            admittance_vel = 0.0
                            break_contact_timer = 0.0
                            _info(f"Contact released after delay: Fz={fz_comp:.3f} N.")
                    else:
                        break_contact_timer = 0.0

                fz_dot = (fz_comp - fz_prev) / dt
                fz_prev = fz_comp

                if is_in_contact:
                    target_kp_z = kp_z_contact
                    f_error = fz_comp - force_target_z
                    if abs(f_error) < force_deadband:
                        f_error = 0.0
                    if user_pull_up:
                        f_error = 0.0
                    admittance_f_error = f_error
                    damping_eff = damping_up if admittance_vel > 0.0 else damping_down
                    admittance_acc = (
                        f_error
                        - force_derivative_gain * fz_dot
                        - damping_eff * admittance_vel
                    ) / admittance_mass
                    admittance_acc = float(np.clip(admittance_acc, -accel_limit, accel_limit))
                else:
                    target_kp_z = kp_z_free
                    admittance_acc = 0.0
                    admittance_vel = 0.0
                    admittance_offset = 0.0
                    admittance_acc = float(np.clip(admittance_acc, -accel_limit, accel_limit))

                admittance_vel += admittance_acc * dt
                admittance_vel = float(
                    np.clip(admittance_vel, -max_admittance_velocity, max_admittance_velocity)
                )

                admittance_offset += admittance_vel * dt
                admittance_offset = float(
                    np.clip(
                        admittance_offset,
                        -max_admittance_position,
                        max_admittance_position,
                    )
                )
                if is_in_contact:
                    nominal_normal = float(np.dot(admittance_pos, admittance_axis_base))
                    tangent_target = (
                        admittance_pos - nominal_normal * admittance_axis_base
                    )
                    commanded_admittance_pos = (
                        tangent_target
                        + (admittance_anchor_normal + admittance_offset)
                        * admittance_axis_base
                    )
                else:
                    commanded_admittance_pos = admittance_pos.copy()

                current_kp_z = (1.0 - kp_z_alpha) * current_kp_z + kp_z_alpha * target_kp_z
                admittance_cfg["Kp"]["translation"] = [
                    float(admittance_runtime.osc.kp_translation[0]),
                    float(admittance_runtime.osc.kp_translation[1]),
                    float(current_kp_z),
                ]

                pos_error = commanded_admittance_pos - tool_pos
                if is_in_contact:
                    normal_component = (
                        float(np.dot(pos_error, admittance_axis_base))
                        * admittance_axis_base
                    )
                    tangent_component = pos_error - normal_component
                    tangent_norm = float(np.linalg.norm(tangent_component))
                    if tangent_norm > max_position_error:
                        tangent_component *= max_position_error / tangent_norm
                        commanded_admittance_pos = (
                            tool_pos + tangent_component + normal_component
                        )
                        pos_error = commanded_admittance_pos - tool_pos
                else:
                    pos_norm = float(np.linalg.norm(pos_error))
                    if pos_norm > max_position_error:
                        commanded_admittance_pos = (
                            tool_pos + pos_error * (max_position_error / pos_norm)
                        )
                        pos_error = commanded_admittance_pos - tool_pos

                delta_quat = transform_utils.quat_distance(
                    canonicalize_quaternion(admittance_quat, reference_quat_xyzw=tool_quat),
                    tool_quat,
                ).astype(np.float64)
                rot_error = transform_utils.quat2axisangle(
                    canonicalize_quaternion(delta_quat)
                ).astype(np.float64)
                rot_norm = float(np.linalg.norm(rot_error))
                if rot_norm > max_rotation_error:
                    rot_error = rot_error * (max_rotation_error / rot_norm)

                max_pos_step = float(admittance_cfg["action_scale"]["translation"])
                max_rot_step = float(admittance_cfg["action_scale"]["rotation"])
                action_pos = np.clip(pos_error, -max_pos_step, max_pos_step)
                admittance_action_z = float(action_pos[2])
                action_rot = np.clip(rot_error, -max_rot_step, max_rot_step)
                action = np.concatenate([action_pos, action_rot, [-1.0]])
                robot.control("OSC_POSE", action, controller_cfg=admittance_cfg)
                steps += 1
                if (
                    args.admittance_debug_interval > 0.0
                    and now - last_admittance_debug_time >= args.admittance_debug_interval
                ):
                    _info(
                        "Admittance debug: "
                        f"force_ready={int(admittance_have_force)} "
                        f"contact={int(is_in_contact)} "
                        f"fz={fz_comp:.3f}N "
                        f"target_fz={force_target_z:.3f}N "
                        f"f_err={admittance_f_error:.3f}N "
                        f"adm_vel={admittance_vel:.4f}m/s "
                        f"adm_z={admittance_offset:.4f}m "
                        f"action_z={admittance_action_z:.4f}m "
                        f"kp_z={current_kp_z:.1f}"
                    )
                    last_admittance_debug_time = now
            force_send = False

            if args.max_steps > 0 and steps >= args.max_steps:
                break
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        if args.terminate_at_exit:
            robot.control(
                "CARTESIAN_VELOCITY",
                np.concatenate([np.zeros(6), [1.0]]),
                controller_cfg=cartesian_cfg,
                termination=True,
            )
        robot.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
