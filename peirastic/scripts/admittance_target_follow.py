"""Standalone constant-force admittance with pose / twist command inputs.

This script does not depend on SpaceNav or SpaceMouse. It uses:

- `/netft_data` for force input
- a `PoseStamped` topic for absolute target updates
- a `Twist` topic for incremental motion commands

The force controller is always active. If the force target is non-zero in free
space, the robot may drift along the tool Z axis while trying to realize that
target. This script intentionally does not implement contact search logic.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Twist, WrenchStamped

from peirastic import config_root
from peirastic.franka_interface import FrankaInterface
from peirastic.utils.admittance_utils import (
    calibrate_tool_z_force,
    canonicalize_quaternion,
    ft_sensor_frame,
    integrate_body_rotation,
    load_netft_calib_param,
)
from peirastic.utils.config_utils import get_default_controller_config
from peirastic.utils.runtime_paths import get_default_netft_calib_yaml
from peirastic.utils.yaml_config import YamlConfig
import peirastic.utils.transform_utils as T


CONTROL_FREQ = 250.0
RESET_JOINTS_ON_START = False
JOINT_RESET_TIMEOUT = 30.0
INIT_JOINT_ANGLES = [0.0, 0.0, 0.0, -1.8, 0.0, 1.8, 0.0]
CONFIG_FILE = os.path.join(config_root, "local-host.yml")
NETFT_CALIB_YAML = os.environ.get(
    "PEIRASTIC_NETFT_CALIB_YAML", get_default_netft_calib_yaml()
)

EEF_OFFSET = np.array([0.0, 0.0, 0.24], dtype=np.float64)
EEF_OFFSET_RPY = [0.0, 0.0, 2.8797932657906435]

POSE_TOPIC = "/peirastic/admittance/target_pose"
TWIST_TOPIC = "/peirastic/admittance/target_twist"
TWIST_TIMEOUT = 0.25
LINEAR_SCALE = 1.0
ANGULAR_SCALE = 1.0

FORCE_TARGET_Z = -5.0
Md = 0.15
Dd_down = 40.0
Dd_up = 20.0
Kdf = 0.0
ACCEL_LIMIT = 2.0
FORCE_DEADBAND = 0.3
ALPHA_FORCE = 0.2
MAX_ADM_POS = 0.15
MAX_ADM_VEL = 0.5
BIAS_WAIT = 1.0

MAX_POS_ERR = 0.05
MAX_ROT_ERR = 0.8

_lock = threading.Lock()
_raw_ft = np.zeros(6, dtype=np.float64)
_force_ready = False
_target_pose_pos: np.ndarray | None = None
_target_pose_quat: np.ndarray | None = None
_twist_linear = np.zeros(3, dtype=np.float64)
_twist_angular = np.zeros(3, dtype=np.float64)
_last_twist_time = 0.0


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone constant-force admittance target follower."
    )
    parser.add_argument(
        "--controller-cfg",
        type=str,
        default="admittance-controller.yml",
        help="Admittance controller YAML under config/ or an absolute path.",
    )
    return parser.parse_args()


def _resolve_config_path(config_name: str) -> str:
    if config_name.startswith("/"):
        return config_name
    return os.path.join(config_root, config_name)


def _die(msg: str, code: int = 1) -> None:
    print(f"[admittance_target_follow] error: {msg}", file=sys.stderr)
    try:
        rospy.logerr(msg)
    except Exception:
        pass
    raise SystemExit(code)


def _apply_runtime_config(runtime_cfg) -> None:
    global CONTROL_FREQ, RESET_JOINTS_ON_START, JOINT_RESET_TIMEOUT
    global INIT_JOINT_ANGLES, CONFIG_FILE, NETFT_CALIB_YAML
    global EEF_OFFSET, EEF_OFFSET_RPY
    global POSE_TOPIC, TWIST_TOPIC, TWIST_TIMEOUT, LINEAR_SCALE, ANGULAR_SCALE
    global FORCE_TARGET_Z, Md, Dd_down, Dd_up, Kdf
    global ACCEL_LIMIT, FORCE_DEADBAND, ALPHA_FORCE, MAX_ADM_POS, MAX_ADM_VEL
    global BIAS_WAIT, MAX_POS_ERR, MAX_ROT_ERR

    CONTROL_FREQ = float(runtime_cfg.control_freq)
    RESET_JOINTS_ON_START = bool(getattr(runtime_cfg, "reset_joints_on_start", False))
    JOINT_RESET_TIMEOUT = float(getattr(runtime_cfg, "joint_reset_timeout", 30.0))
    INIT_JOINT_ANGLES = [float(v) for v in runtime_cfg.init_joint_angles]

    interface_cfg = runtime_cfg.interface_cfg
    CONFIG_FILE = (
        interface_cfg
        if interface_cfg.startswith("/")
        else os.path.join(config_root, interface_cfg)
    )

    env_key = runtime_cfg.netft_calibration.env_key
    NETFT_CALIB_YAML = os.path.abspath(
        os.path.expanduser(
            os.environ.get(env_key, runtime_cfg.netft_calibration.default_path)
        )
    )

    EEF_OFFSET = np.array(runtime_cfg.tool.eef_offset, dtype=np.float64)
    EEF_OFFSET_RPY = [float(v) for v in runtime_cfg.tool.eef_offset_rpy]

    POSE_TOPIC = str(runtime_cfg.command_input.pose_topic)
    TWIST_TOPIC = str(runtime_cfg.command_input.twist_topic)
    TWIST_TIMEOUT = float(runtime_cfg.command_input.twist_timeout)
    LINEAR_SCALE = float(runtime_cfg.command_input.linear_scale)
    ANGULAR_SCALE = float(runtime_cfg.command_input.angular_scale)

    FORCE_TARGET_Z = float(runtime_cfg.admittance.force_target_z)
    Md = float(runtime_cfg.admittance.mass)
    Dd_down = float(runtime_cfg.admittance.damping_down)
    Dd_up = float(runtime_cfg.admittance.damping_up)
    Kdf = float(runtime_cfg.admittance.force_derivative_gain)
    ACCEL_LIMIT = float(runtime_cfg.admittance.accel_limit)
    FORCE_DEADBAND = float(runtime_cfg.admittance.force_deadband)
    ALPHA_FORCE = float(runtime_cfg.admittance.force_alpha)
    MAX_ADM_POS = float(runtime_cfg.admittance.max_admittance_position)
    MAX_ADM_VEL = float(runtime_cfg.admittance.max_admittance_velocity)
    BIAS_WAIT = float(runtime_cfg.admittance.bias_wait)

    MAX_POS_ERR = float(runtime_cfg.safety.max_position_error)
    MAX_ROT_ERR = float(runtime_cfg.safety.max_rotation_error)


def _netft_cb(msg: WrenchStamped):
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


def _target_pose_cb(msg: PoseStamped):
    global _target_pose_pos, _target_pose_quat
    pos = np.array(
        [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
        dtype=np.float64,
    )
    quat = np.array(
        [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ],
        dtype=np.float64,
    )
    quat_norm = float(np.linalg.norm(quat))
    with _lock:
        _target_pose_pos = pos
        _target_pose_quat = (
            None if quat_norm <= 1e-12 else canonicalize_quaternion(quat)
        )


def _target_twist_cb(msg: Twist):
    global _twist_linear, _twist_angular, _last_twist_time
    with _lock:
        _twist_linear = np.array(
            [msg.linear.x, msg.linear.y, msg.linear.z], dtype=np.float64
        )
        _twist_angular = np.array(
            [msg.angular.x, msg.angular.y, msg.angular.z], dtype=np.float64
        )
        _last_twist_time = time.monotonic()


def _current_eef_pose(
    robot_interface: FrankaInterface, eef_offset_rot: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = robot_interface.last_state
    if state is None:
        raise RuntimeError("No robot state is available.")
    t_cur = np.array(state.O_T_EE, dtype=np.float64).reshape(4, 4).T
    r_link8 = t_cur[:3, :3]
    pos_link8 = t_cur[:3, 3]
    eef_pos = pos_link8 + r_link8 @ EEF_OFFSET
    eef_rot = r_link8 @ eef_offset_rot
    eef_quat = canonicalize_quaternion(T.mat2quat(eef_rot))
    return eef_pos, eef_quat, r_link8


def _wait_for_robot_state(robot_interface: FrankaInterface, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if robot_interface.last_state is not None:
            return
        time.sleep(0.1)
    _die(
        "No franka-interface state was received within 10 seconds. "
        "Start `franka-interface` and verify your network config."
    )


def _build_osc_cfg(runtime_cfg):
    cfg = get_default_controller_config("OSC_POSE")
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


def _rotation_error_clipped(
    nominal_quat: np.ndarray, current_quat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r_nom = T.quat2mat(nominal_quat)
    r_cur = T.quat2mat(current_quat)
    rot_error = T.quat_distance(nominal_quat, current_quat)
    quat_w = float(np.clip(np.abs(rot_error[3]), -1.0, 1.0))
    angle = float(2.0 * np.arccos(quat_w))
    if angle <= MAX_ROT_ERR or angle <= 1e-12:
        return nominal_quat, T.quat2axisangle(rot_error)

    rotvec = T.quat2axisangle(rot_error)
    clipped_rotvec = rotvec * (MAX_ROT_ERR / np.linalg.norm(rotvec))
    clipped_r = T.quat2mat(T.axisangle2quat(clipped_rotvec)) @ r_cur
    clipped_quat = canonicalize_quaternion(T.mat2quat(clipped_r), current_quat)
    return clipped_quat, clipped_rotvec


def main():
    args = _parse_args()
    runtime_cfg = YamlConfig(_resolve_config_path(args.controller_cfg)).as_easydict()
    _apply_runtime_config(runtime_cfg)

    rospy.init_node("admittance_target_follow", anonymous=False)

    if not os.path.isfile(NETFT_CALIB_YAML):
        _die(
            f"Calibration file not found: {NETFT_CALIB_YAML}\n"
            "Set PEIRASTIC_NETFT_CALIB_YAML or update the runtime config."
        )

    try:
        subprocess.run(["rosparam", "load", NETFT_CALIB_YAML], check=True)
    except subprocess.CalledProcessError as exc:
        _die(f"`rosparam load` failed: {exc}")
    except FileNotFoundError:
        _die("`rosparam` was not found. Source your ROS environment first.")

    try:
        netft_param = rospy.get_param("/netft_calib_param")
    except KeyError:
        _die("ROS parameter `/netft_calib_param` was not found after `rosparam load`.")

    try:
        calib_params, sensor_rot_z_rad, sensor_tz, ftf = load_netft_calib_param(
            netft_param
        )
    except KeyError as exc:
        _die(f"Calibration fields are missing: {exc}")

    rospy.Subscriber("/netft_data", WrenchStamped, _netft_cb, queue_size=1, tcp_nodelay=True)
    if POSE_TOPIC:
        rospy.Subscriber(POSE_TOPIC, PoseStamped, _target_pose_cb, queue_size=1)
    if TWIST_TOPIC:
        rospy.Subscriber(TWIST_TOPIC, Twist, _target_twist_cb, queue_size=1)

    robot_interface = FrankaInterface(
        CONFIG_FILE, control_freq=CONTROL_FREQ, use_visualizer=False
    )
    try:
        _wait_for_robot_state(robot_interface)
        if RESET_JOINTS_ON_START:
            ok = robot_interface.move_joints(
                INIT_JOINT_ANGLES,
                timeout=JOINT_RESET_TIMEOUT,
                state_timeout=10.0,
            )
            if not ok:
                _die("Failed to move to init_joint_angles before starting admittance.")

        eef_offset_rot = T.euler2mat(np.array(EEF_OFFSET_RPY, dtype=np.float64))
        cfg = _build_osc_cfg(runtime_cfg)
        hold_action = [0.0] * 6 + [-1.0]

        for _ in range(20):
            robot_interface.control(
                controller_type="OSC_POSE",
                action=hold_action,
                controller_cfg=cfg,
                shutdown_check=rospy.is_shutdown,
            )
            time.sleep(0.02)

        command_pos, command_quat, _ = _current_eef_pose(robot_interface, eef_offset_rot)

        force_samples = []
        filtered_fz = None
        bias_deadline = time.time() + BIAS_WAIT
        while time.time() < bias_deadline and not rospy.is_shutdown():
            robot_interface.control(
                controller_type="OSC_POSE",
                action=hold_action,
                controller_cfg=cfg,
                shutdown_check=rospy.is_shutdown,
            )
            try:
                _, _, r_link8 = _current_eef_pose(robot_interface, eef_offset_rot)
            except RuntimeError:
                time.sleep(1.0 / CONTROL_FREQ)
                continue
            with _lock:
                raw_ft = _raw_ft.copy()
                have_force = _force_ready
            if have_force:
                link8_rpy = T.mat2euler(r_link8, axes="sxyz")
                sensor_rpy = ft_sensor_frame(link8_rpy, sensor_rot_z_rad, sensor_tz)
                fz_cal = calibrate_tool_z_force(raw_ft, sensor_rpy, calib_params, ftf)
                if filtered_fz is None:
                    filtered_fz = fz_cal
                else:
                    filtered_fz = ALPHA_FORCE * filtered_fz + (1.0 - ALPHA_FORCE) * fz_cal
                force_samples.append(fz_cal)
            time.sleep(1.0 / CONTROL_FREQ)

        force_bias = float(np.mean(force_samples)) if force_samples else 0.0
        filtered_fz = force_bias if filtered_fz is None else filtered_fz

        adm_pos = 0.0
        adm_vel = 0.0
        fz_prev = 0.0
        dt = 1.0 / CONTROL_FREQ

        rospy.loginfo(
            "[admittance_target_follow] pose_topic=%s twist_topic=%s force_target_z=%.3f",
            POSE_TOPIC or "<disabled>",
            TWIST_TOPIC or "<disabled>",
            FORCE_TARGET_Z,
        )

        while not rospy.is_shutdown():
            loop_start = time.time()

            current_pos, current_quat, r_link8 = _current_eef_pose(
                robot_interface, eef_offset_rot
            )

            with _lock:
                pose_pos = None if _target_pose_pos is None else _target_pose_pos.copy()
                pose_quat = None if _target_pose_quat is None else _target_pose_quat.copy()
                twist_lin = _twist_linear.copy()
                twist_ang = _twist_angular.copy()
                twist_age = time.monotonic() - _last_twist_time
                raw_ft = _raw_ft.copy()
                have_force = _force_ready

            if pose_pos is not None:
                command_pos = pose_pos
            if pose_quat is not None:
                command_quat = canonicalize_quaternion(
                    pose_quat, reference_quat_xyzw=command_quat
                )

            if TWIST_TOPIC and twist_age <= TWIST_TIMEOUT:
                command_pos = command_pos + twist_lin * LINEAR_SCALE * dt
                command_quat = integrate_body_rotation(
                    command_quat, twist_ang * ANGULAR_SCALE, dt
                )

            if have_force:
                link8_rpy = T.mat2euler(r_link8, axes="sxyz")
                sensor_rpy = ft_sensor_frame(link8_rpy, sensor_rot_z_rad, sensor_tz)
                fz_cal = calibrate_tool_z_force(raw_ft, sensor_rpy, calib_params, ftf)
                filtered_fz = ALPHA_FORCE * filtered_fz + (1.0 - ALPHA_FORCE) * fz_cal

            fz_comp = float(filtered_fz - force_bias)
            fz_dot = (fz_comp - fz_prev) / dt
            fz_prev = fz_comp

            f_error = fz_comp - FORCE_TARGET_Z
            if abs(f_error) < FORCE_DEADBAND:
                f_error = 0.0

            dd_eff = Dd_up if adm_vel > 0.0 else Dd_down
            adm_acc = (f_error - Kdf * fz_dot - dd_eff * adm_vel) / Md
            adm_acc = float(np.clip(adm_acc, -ACCEL_LIMIT, ACCEL_LIMIT))

            adm_vel += adm_acc * dt
            adm_vel = float(np.clip(adm_vel, -MAX_ADM_VEL, MAX_ADM_VEL))
            adm_pos += adm_vel * dt
            adm_pos = float(np.clip(adm_pos, -MAX_ADM_POS, MAX_ADM_POS))

            tool_z_base = T.quat2mat(current_quat)[:, 2]
            nominal_pos = command_pos + adm_pos * tool_z_base
            pos_error = nominal_pos - current_pos
            pos_dist = float(np.linalg.norm(pos_error))
            if pos_dist > MAX_POS_ERR:
                nominal_pos = current_pos + pos_error * (MAX_POS_ERR / pos_dist)
                pos_error = nominal_pos - current_pos

            nominal_quat, rotvec_base = _rotation_error_clipped(command_quat, current_quat)

            max_pos_step = float(cfg["action_scale"]["translation"])
            max_rot_step = float(cfg["action_scale"]["rotation"])
            action_pos = np.clip(pos_error, -max_pos_step, max_pos_step)
            action_rot = np.clip(rotvec_base, -max_rot_step, max_rot_step)
            action = list(np.concatenate([action_pos, action_rot])) + [-1.0]

            robot_interface.control(
                controller_type="OSC_POSE",
                action=action,
                controller_cfg=cfg,
                shutdown_check=rospy.is_shutdown,
            )

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, dt - elapsed))
    finally:
        robot_interface.close()


if __name__ == "__main__":
    main()
