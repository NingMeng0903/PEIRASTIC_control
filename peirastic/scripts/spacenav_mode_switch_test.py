#!/usr/bin/env python3
"""SpaceNav teleop with runtime switching between Cartesian tracking and OSC."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import rosgraph
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

from peirastic import ROOT_PATH, config_root
from peirastic.franka_interface import FrankaInterface
from peirastic.utils.config_utils import get_default_controller_config, verify_controller_config
from peirastic.utils import transform_utils
from peirastic.utils.yaml_config import YamlConfig

INIT_JOINT_ANGLES = [0.0, 0.0, 0.0, -1.8, 0.0, 1.8, 0.0]

MODE_CARTESIAN = "CARTESIAN_TRACKING"
MODE_OSC = "OSC_POSE"

_lock = threading.Lock()
_sn_lin = np.zeros(3, dtype=np.float64)
_sn_ang = np.zeros(3, dtype=np.float64)
_sn_lin_bias = np.zeros(3, dtype=np.float64)
_sn_ang_bias = np.zeros(3, dtype=np.float64)
_last_twist_time = 0.0
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


def _launch_local_controller(interface_cfg: str, log_path: str) -> int:
    repo_root = Path(ROOT_PATH).resolve().parent
    launcher = repo_root / "scripts" / "run_franka_interface.sh"
    if not launcher.exists():
        raise FileNotFoundError(f"Launcher not found: {launcher}")

    log_file = Path(os.path.expanduser(log_path)).resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "ab") as f:
        proc = subprocess.Popen(
            [str(launcher), interface_cfg],
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
    parser.add_argument("--control-freq", type=float, default=200.0)
    parser.add_argument("--linear-scale", type=float, default=0.08)
    parser.add_argument("--angular-scale", type=float, default=0.18)
    parser.add_argument("--traj-time-fraction", type=float, default=0.5)
    parser.add_argument("--linear-deadzone", type=float, default=0.01)
    parser.add_argument("--angular-deadzone", type=float, default=0.01)
    parser.add_argument("--spacenav-timeout", type=float, default=5.0)
    parser.add_argument("--state-timeout", type=float, default=30.0)
    parser.add_argument("--init-timeout", type=float, default=120.0)
    parser.add_argument("--reset-on-start", action="store_true")
    parser.add_argument("--launch-controller", action="store_true")
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
        default=0.002,
        help="Per-cycle OSC delta translation clip in meters.",
    )
    parser.add_argument(
        "--osc-max-rot-delta",
        type=float,
        default=0.02,
        help="Per-cycle OSC delta rotation clip in radians.",
    )
    parser.add_argument(
        "--osc-translation-stiffness",
        type=float,
        default=200.0,
        help="Per-axis OSC translational stiffness used by this safety test script.",
    )
    parser.add_argument(
        "--osc-rotation-stiffness",
        type=float,
        default=200.0,
        help="Per-axis OSC rotational stiffness used by this safety test script.",
    )
    parser.add_argument("--debug-input", action="store_true")
    return parser.parse_args()


def _info(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    args = _parse_args()
    interface_cfg = _resolve_config_path(args.interface_cfg)

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

    _info("Checking ROS master...")
    if not rosgraph.is_master_online():
        print("ROS master is not reachable. Start roscore first.", file=sys.stderr, flush=True)
        return 1

    _info("Connecting to ROS node...")
    rospy.init_node("spacenav_mode_switch_test", anonymous=False)
    rospy.Subscriber("/spacenav/twist", Twist, _twist_cb, queue_size=1)
    rospy.Subscriber("/spacenav/joy", Joy, _joy_cb, queue_size=1)

    try:
        _info("Waiting for /spacenav/twist messages...")
        rospy.wait_for_message("/spacenav/twist", Twist, timeout=args.spacenav_timeout)
    except rospy.ROSException:
        print("Timeout: no /spacenav/twist messages. Start roscore and spacenav_node first.", file=sys.stderr)
        return 2

    if args.launch_controller:
        try:
            _info("Launching local franka-interface...")
            pid = _launch_local_controller(interface_cfg, args.controller_log)
        except Exception as exc:
            print(f"Failed to launch local franka-interface: {exc}", file=sys.stderr)
            return 3
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
    force_send = True

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

        _info(
            "Captured SpaceNav neutral bias: "
            f"lin={np.round(bias_lin, 6).tolist()} ang={np.round(bias_ang, 6).tolist()}"
        )
        _info(
            "Ready: button0 toggles CARTESIAN_TRACKING/OSC_POSE; "
            "button1 triggers joint reset."
        )

        dt = 1.0 / args.control_freq
        rate = rospy.Rate(args.control_freq)

        while not rospy.is_shutdown():
            sn_lin, sn_ang, do_reset, do_toggle, last_twist_time = _poll_spacenav_state()
            raw_sn_lin = sn_lin - _sn_lin_bias
            raw_sn_ang = sn_ang - _sn_ang_bias
            sn_lin = raw_sn_lin.copy()
            sn_ang = raw_sn_ang.copy()

            if do_toggle:
                active_mode = MODE_OSC if active_mode == MODE_CARTESIAN else MODE_CARTESIAN
                robot.bump_control_session()
                # Ensure the next target controller message is not dropped by policy-rate
                # throttling right after a bursty mode switch, and re-run the NO_CONTROL
                # handshake the next time controller_type actually changes in control().
                robot.reset_control_rate_limiter()
                robot.force_next_control_preprocess()
                force_send = True
                _info(f"Switched control mode to {active_mode}.")

            if do_reset:
                if not robot.move_joints(INIT_JOINT_ANGLES, timeout=args.init_timeout):
                    print("Failed to reset to INIT_JOINT_ANGLES.", file=sys.stderr)
                    return 7
                desired_pos, desired_quat = _capture_pose_target(robot)
                if desired_pos is None or desired_quat is None:
                    print("Robot pose is unavailable after reset.", file=sys.stderr)
                    return 8
                command_pos = desired_pos.copy()
                command_quat = desired_quat.copy()
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

            if np.linalg.norm(sn_lin) > 1e-12 or np.linalg.norm(sn_ang) > 1e-12:
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

            command_pos, command_quat, target_pos_error, target_rot_error = _step_command_target(
                command_pos,
                command_quat,
                desired_pos,
                desired_quat,
                args.command_max_pos_step,
                args.command_max_rot_step,
            )
            target_pending = target_pos_error > 1e-5 or target_rot_error > 1e-4

            if active_mode == MODE_CARTESIAN:
                if force_send or target_pending or args.max_steps == 0 or steps < args.max_steps:
                    action = _build_absolute_action(command_pos, command_quat)
                    robot.control("CARTESIAN_VELOCITY", action, controller_cfg=cartesian_cfg)
                    steps += 1
            else:
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
