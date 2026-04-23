#!/usr/bin/env python3
"""Minimal single-machine SpaceNav -> CARTESIAN_VELOCITY pose tracking teleop."""

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

_lock = threading.Lock()
_sn_lin = np.zeros(3, dtype=np.float64)
_sn_ang = np.zeros(3, dtype=np.float64)
_sn_lin_bias = np.zeros(3, dtype=np.float64)
_sn_ang_bias = np.zeros(3, dtype=np.float64)
_last_twist_time = 0.0
_reset_requested = False
_btn1_prev = 0


def _twist_cb(msg: Twist) -> None:
    global _sn_lin, _sn_ang, _last_twist_time
    with _lock:
        _sn_lin = np.array([msg.linear.x, msg.linear.y, msg.linear.z], dtype=np.float64)
        _sn_ang = np.array([msg.angular.x, msg.angular.y, msg.angular.z], dtype=np.float64)
        _last_twist_time = time.monotonic()


def _joy_cb(msg: Joy) -> None:
    global _reset_requested, _btn1_prev
    if len(msg.buttons) < 2:
        return
    cur = int(msg.buttons[1])
    with _lock:
        if cur == 1 and _btn1_prev == 0:
            _reset_requested = True
        _btn1_prev = cur


def _resolve_config_path(cfg_name: str) -> str:
    return cfg_name if cfg_name.startswith("/") else f"{config_root}/{cfg_name}"


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
        # Match deoxys/spacenav_teleop_publisher: apply incremental rotation in
        # the current tool frame (q_next = q_current * delta_q), not base frame.
        next_quat = transform_utils.quat_multiply(next_quat, delta_quat).astype(np.float64)
        next_quat = _canonicalize_quaternion(next_quat, reference_quat=target_quat)

    return next_pos, next_quat


def _canonicalize_quaternion(quat, reference_quat=None):
    quat = np.asarray(quat, dtype=np.float64).copy()
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    quat /= norm
    if reference_quat is not None:
        reference_quat = np.asarray(reference_quat, dtype=np.float64)
        if np.dot(quat, reference_quat) < 0.0:
            quat *= -1.0
    elif quat[3] < 0.0:
        quat *= -1.0
    return quat


def _build_absolute_action(target_pos, target_quat):
    canonical_quat = _canonicalize_quaternion(target_quat)
    return np.concatenate([np.asarray(target_pos, dtype=np.float64), canonical_quat, [-1.0]])


def _poll_spacenav_state():
    global _reset_requested
    with _lock:
        sn_lin = _sn_lin.copy()
        sn_ang = _sn_ang.copy()
        last_twist_time = _last_twist_time
        do_reset = _reset_requested
        _reset_requested = False
    return sn_lin, sn_ang, do_reset, last_twist_time


def _capture_spacenav_bias():
    global _sn_lin_bias, _sn_ang_bias
    with _lock:
        _sn_lin_bias = _sn_lin.copy()
        _sn_ang_bias = _sn_ang.copy()
    return _sn_lin_bias.copy(), _sn_ang_bias.copy()


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface-cfg", default="local-host.yml")
    parser.add_argument(
        "--controller-cfg",
        default="",
        help="YAML under config/; default: built-in cartesian-velocity-controller.yml.",
    )
    parser.add_argument("--control-freq", type=float, default=200.0)
    parser.add_argument("--linear-scale", type=float, default=0.12)
    parser.add_argument("--angular-scale", type=float, default=0.3)
    parser.add_argument(
        "--traj-time-fraction",
        type=float,
        default=0.5,
        help="Trajectory minimum-duration multiplier for streaming absolute pose targets.",
    )
    parser.add_argument(
        "--linear-deadzone",
        type=float,
        default=0.01,
        help="Spacenav linear deadzone (deoxys spacenav_teleop_publisher default).",
    )
    parser.add_argument(
        "--angular-deadzone",
        type=float,
        default=0.01,
        help="Spacenav angular deadzone (deoxys spacenav_teleop_publisher default).",
    )
    parser.add_argument(
        "--debug-input",
        action="store_true",
        help="Print SpaceNav raw and filtered input norms periodically.",
    )
    parser.add_argument("--spacenav-timeout", type=float, default=5.0)
    parser.add_argument("--state-timeout", type=float, default=30.0)
    parser.add_argument("--init-timeout", type=float, default=120.0)
    parser.add_argument(
        "--reset-on-start",
        action="store_true",
        help="Reset to INIT_JOINT_ANGLES before entering Cartesian teleop.",
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--launch-controller",
        action="store_true",
        help="Launch local franka-interface with the same interface config before connecting.",
    )
    parser.add_argument(
        "--controller-log",
        default="/tmp/peirastic-franka-interface.log",
        help="Log path used when --launch-controller is enabled.",
    )
    parser.add_argument(
        "--terminate-at-exit",
        action="store_true",
        help="Send termination=True on exit instead of keeping the local controller alive.",
    )
    return parser.parse_args()


def _info(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    args = _parse_args()
    interface_cfg = _resolve_config_path(args.interface_cfg)
    controller_cfg_path = _resolve_config_path(args.controller_cfg) if args.controller_cfg else ""

    if controller_cfg_path:
        controller_cfg = verify_controller_config(YamlConfig(controller_cfg_path).as_easydict())
    else:
        controller_cfg = get_default_controller_config("CARTESIAN_VELOCITY")
    controller_cfg["is_delta"] = False
    controller_cfg["traj_interpolator_cfg"]["time_fraction"] = max(
        0.1, float(args.traj_time_fraction)
    )

    _info("Checking ROS master...")
    if not rosgraph.is_master_online():
        print("ROS master is not reachable. Start roscore first.", file=sys.stderr, flush=True)
        return 1

    _info("Connecting to ROS node...")
    rospy.init_node("spacenav_cartesian_min", anonymous=False)
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
        print(f"Launched local franka-interface (pid={pid}).", flush=True)
        time.sleep(1.0)

    _info("Connecting to franka-interface...")
    robot = FrankaInterface(
        interface_cfg,
        control_freq=args.control_freq,
        use_visualizer=False,
        automatic_gripper_reset=False,
    )
    zmq_cfg = YamlConfig(interface_cfg).as_easydict()
    _info(
        "ZMQ endpoints (FrankaInterface): "
        f"command_pub_bind=tcp://*:{zmq_cfg.NUC.SUB_PORT} "
        f"state_sub_connect=tcp://{zmq_cfg.NUC.IP}:{zmq_cfg.NUC.PUB_PORT} "
        f"pc_ip={zmq_cfg.PC.IP}"
    )
    _info(
        "If franka-interface logs only `Counter 0` while teleoping, the running "
        "franka-interface is not receiving these commands (wrong host/ports, or a "
        "second stale franka-interface instance)."
    )

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
            _info(
                "Skipping startup joint reset (pass --reset-on-start to move to INIT_JOINT_ANGLES first); "
                "using current robot pose as teleop reference."
            )

        bias_lin, bias_ang = _capture_spacenav_bias()
        current_pose = robot.last_eef_pose
        if current_pose is None:
            print("Robot pose is unavailable after startup initialization.", file=sys.stderr, flush=True)
            return 7
        target_pos = current_pose[:3, 3].astype(np.float64).copy()
        target_quat = _canonicalize_quaternion(
            transform_utils.mat2quat(current_pose[:3, :3]).astype(np.float64).copy()
        )
        _info(
            "Captured SpaceNav neutral bias: "
            f"lin={np.round(bias_lin, 6).tolist()} ang={np.round(bias_ang, 6).tolist()}"
        )
        _info(
            "Streaming config: "
            f"control_freq={args.control_freq:.1f}Hz "
            f"time_fraction={controller_cfg['traj_interpolator_cfg']['time_fraction']:.3f} "
            f"linear_scale={args.linear_scale:.3f} "
            f"angular_scale={args.angular_scale:.3f}"
        )
        _info("Ready: SpaceNav -> integrated pose; joy button 2 (index 1) triggers joint reset.")
        dt = 1.0 / args.control_freq
        steps = 0
        rate = rospy.Rate(args.control_freq)
        last_debug_time = 0.0

        while not rospy.is_shutdown():
            sn_lin, sn_ang, do_reset, last_twist_time = _poll_spacenav_state()
            raw_sn_lin = sn_lin - _sn_lin_bias
            raw_sn_ang = sn_ang - _sn_ang_bias
            sn_lin = raw_sn_lin.copy()
            sn_ang = raw_sn_ang.copy()

            if do_reset:
                if not robot.move_joints(INIT_JOINT_ANGLES, timeout=args.init_timeout):
                    print("Failed to reset to INIT_JOINT_ANGLES.", file=sys.stderr)
                    return 6
                current_pose = robot.last_eef_pose
                if current_pose is not None:
                    target_pos = current_pose[:3, 3].astype(np.float64).copy()
                    target_quat = _canonicalize_quaternion(
                        transform_utils.mat2quat(current_pose[:3, :3]).astype(np.float64).copy()
                    )
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
                    f"raw_lin={np.linalg.norm(raw_sn_lin):.6f} "
                    f"raw_ang={np.linalg.norm(raw_sn_ang):.6f} "
                    f"filtered_lin={np.linalg.norm(sn_lin):.6f} "
                    f"filtered_ang={np.linalg.norm(sn_ang):.6f} "
                    f"msg_age={input_age:.3f}s"
                )
                last_debug_time = now

            command_delta = np.concatenate(
                [
                    sn_lin * args.linear_scale * dt,
                    np.array([sn_ang[0], -sn_ang[1], -sn_ang[2]], dtype=np.float64)
                    * args.angular_scale
                    * dt,
                ]
            )
            command_delta_norm = float(np.linalg.norm(command_delta))

            if command_delta_norm <= 1e-12:
                rate.sleep()
                continue

            target_pos, target_quat = _integrate_absolute_target(
                target_pos,
                target_quat,
                sn_lin,
                sn_ang,
                dt,
                args.linear_scale,
                args.angular_scale,
            )
            action = _build_absolute_action(target_pos, target_quat)
            robot.control("CARTESIAN_VELOCITY", action, controller_cfg=controller_cfg)
            steps += 1
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
                controller_cfg=controller_cfg,
                termination=True,
            )
        robot.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
