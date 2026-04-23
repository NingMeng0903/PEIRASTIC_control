"""
SpaceNav teleop + EEF-z admittance (constant force) using 2nd-order model.
Raw FT calibrated inline.
`NETFT_CALIB_YAML` is loaded through `rosparam load` at startup.

Dependencies: ROS, `/netft_data`, `franka-interface` (PEIRASTIC ZMQ), and a
force sensor calibration YAML file.
"""

import argparse
import os
import subprocess
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

import rospy
from geometry_msgs.msg import WrenchStamped, Twist

from peirastic import config_root
from peirastic.franka_interface import FrankaInterface
from peirastic.utils.config_utils import get_default_controller_config
from peirastic.utils.runtime_paths import get_default_netft_calib_yaml
from peirastic.utils.yaml_config import YamlConfig
import peirastic.utils.transform_utils as T

CONTROL_FREQ = 250.0
INIT_JOINT_ANGLES = [0.0, 0.0, 0.0, -1.8, 0.0, 1.8, 0.0]
CONFIG_FILE = os.path.join(config_root, "local-host.yml")
# Override with:
# export PEIRASTIC_NETFT_CALIB_YAML=/path/to/netft_calib_result.yaml
NETFT_CALIB_YAML = os.environ.get(
    "PEIRASTIC_NETFT_CALIB_YAML",
    get_default_netft_calib_yaml(),
)

EEF_OFFSET = np.array([0.0, 0.0, 0.24], dtype=np.float64)
EEF_OFFSET_RPY = [0.0, 0.0, 2.8797932657906435]

FORCE_TARGET_Z = -5.0

# 2nd-order admittance parameters
Md = 0.15
Dd_down = 40.0
Dd_up = 20.0
Kdf = 0.0
ACCEL_LIMIT = 2.0
FORCE_DEADBAND = 0.3
ALPHA_FORCE = 0.2
MAX_ADM_POS = 0.15
MAX_ADM_VEL = 0.50
BIAS_WAIT = 1.0

LIN_SCALE = 0.12
ROT_SCALE = 0.3
LIN_DEADZONE = 0.01
ROT_DEADZONE = 0.01

MAX_POS_ERR = 0.05
MAX_ROT_ERR = 0.8

_Ftf: np.ndarray = None

_lock = threading.Lock()
_raw_ft = np.zeros(6)
_force_ready = False
_sn_lin = np.zeros(3)
_sn_ang = np.zeros(3)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run SpaceNav teleoperation with a Z-axis admittance outer loop."
    )
    parser.add_argument(
        "--controller-cfg",
        type=str,
        default="spacenav-admittance-controller.yml",
        help="Admittance controller YAML under the config directory or an absolute path.",
    )
    return parser.parse_args()


def _resolve_config_path(config_name: str) -> str:
    if config_name.startswith("/"):
        return config_name
    return os.path.join(config_root, config_name)


def _apply_runtime_config(runtime_cfg) -> None:
    global CONTROL_FREQ, INIT_JOINT_ANGLES, CONFIG_FILE, NETFT_CALIB_YAML
    global EEF_OFFSET, EEF_OFFSET_RPY
    global FORCE_TARGET_Z, Md, Dd_down, Dd_up, Kdf
    global ACCEL_LIMIT, FORCE_DEADBAND, ALPHA_FORCE, MAX_ADM_POS, MAX_ADM_VEL
    global BIAS_WAIT, LIN_SCALE, ROT_SCALE, LIN_DEADZONE, ROT_DEADZONE
    global MAX_POS_ERR, MAX_ROT_ERR

    CONTROL_FREQ = float(runtime_cfg.control_freq)
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

    LIN_SCALE = float(runtime_cfg.teleop.linear_scale)
    ROT_SCALE = float(runtime_cfg.teleop.angular_scale)
    LIN_DEADZONE = float(runtime_cfg.teleop.linear_deadzone)
    ROT_DEADZONE = float(runtime_cfg.teleop.angular_deadzone)

    MAX_POS_ERR = float(runtime_cfg.safety.max_position_error)
    MAX_ROT_ERR = float(runtime_cfg.safety.max_rotation_error)


def _trans_mtx(x, y, z):
    M = np.eye(4)
    M[:3, 3] = [x, y, z]
    return M


def _euler_mtx_sxyz(rx, ry, rz):
    M = np.eye(4)
    M[:3, :3] = R_scipy.from_euler("xyz", [rx, ry, rz]).as_matrix()
    return M


def _rot_z_4x4(angle):
    M = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    M[0, 0], M[0, 1], M[1, 0], M[1, 1] = c, -s, s, c
    return M


def _euler_from_mtx_sxyz(M):
    return R_scipy.from_matrix(M[:3, :3]).as_euler("xyz")


def _ft_sensor_frame(rpy, sensor_rot_z_rad: float, sensor_tz: float):
    tf_tmp = _euler_mtx_sxyz(*rpy)
    Rz = _rot_z_4x4(sensor_rot_z_rad)
    Tz = _trans_mtx(0.0, 0.0, sensor_tz)
    tf_tmp = tf_tmp @ Tz @ Rz
    return _euler_from_mtx_sxyz(tf_tmp)


def _rotation_base_2_end(gamma, beta, alpha):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    return [[ca * cb, sa * cb, -sb],
            [ca * sb * sg - sa * cg, sa * sb * sg + ca * cg, cb * sg],
            [ca * sb * cg + sa * sg, sa * sb * cg - ca * sg, cb * cg]]


def _build_ftf(probe_length: float, rot_z_deg: float) -> np.ndarray:
    theta = np.deg2rad(rot_z_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0],
                  [s, c, 0.0],
                  [0.0, 0.0, 1.0]])
    p_s = np.array([0.0, 0.0, -probe_length])
    p_e = R @ p_s
    p_cross = np.array([[0.0, -p_e[2], p_e[1]],
                        [p_e[2], 0.0, -p_e[0]],
                        [-p_e[1], p_e[0], 0.0]])
    Ftf = np.zeros((6, 6))
    Ftf[:3, :3] = R
    Ftf[3:, :3] = p_cross @ R
    Ftf[3:, 3:] = R
    return Ftf


def _calibrate_fz(raw_ft, rpy_v, params):
    Fx0, Fy0, Fz0, Mx0, My0, Mz0, Lx, Ly, Lz, cmx, cmy, cmz = params
    Fx, Fy, Fz, Mx, My, Mz = raw_ft
    gamma, beta, alpha = rpy_v
    R_ee_b = _rotation_base_2_end(gamma, beta, alpha)
    G = np.dot(R_ee_b, [[Lx], [Ly], [Lz]]).flatten()
    Gx, Gy, Gz = G
    comp = np.array([
        Fx - Fx0 - Gx, Fy - Fy0 - Gy, Fz - Fz0 - Gz,
        Mx - Mx0 - (Gz * cmy - Gy * cmz),
        My - My0 - (Gx * cmz - Gz * cmx),
        Mz - Mz0 - (Gy * cmx - Gx * cmy),
    ]).reshape(6, 1)
    Fcali = (_Ftf @ comp).flatten()
    return float(Fcali[2])


def _netft_cb(msg: WrenchStamped):
    global _raw_ft, _force_ready
    with _lock:
        _raw_ft = np.array([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z,
        ])
        _force_ready = True


def _spacenav_cb(msg: Twist):
    global _sn_lin, _sn_ang
    lin = np.array([msg.linear.x, msg.linear.y, msg.linear.z])
    ang = np.array([msg.angular.x, msg.angular.y, msg.angular.z])
    if np.linalg.norm(lin) < LIN_DEADZONE:
        lin = np.zeros(3)
    if np.linalg.norm(ang) < ROT_DEADZONE:
        ang = np.zeros(3)
    with _lock:
        _sn_lin = lin
        _sn_ang = ang


def integrate_rotation(quat_xyzw: np.ndarray, omega_eef: np.ndarray, dt: float) -> np.ndarray:
    angle = np.linalg.norm(omega_eef) * dt
    if angle < 1e-9:
        return quat_xyzw
    axis = omega_eef / np.linalg.norm(omega_eef)
    delta = R_scipy.from_rotvec(axis * angle)
    R_cur = R_scipy.from_quat(quat_xyzw)
    R_new = R_cur * delta
    q = R_new.as_quat()
    return -q if q[3] < 0 else q


def go_to_init_joints(robot_interface: FrankaInterface):
    from peirastic.utils.config_utils import get_default_controller_config as gcc
    joint_cfg = gcc("JOINT_POSITION")
    action = list(INIT_JOINT_ANGLES) + [-1.0]
    target_q = np.array(INIT_JOINT_ANGLES)
    iterations = 0
    while not rospy.is_shutdown():
        state = robot_interface.last_state
        if state is not None:
            q = np.array(state.q)
            if np.max(np.abs(q - target_q)) < 1e-3:
                break
            if iterations % 200 == 0:
                pass
        robot_interface.control(
            controller_type="JOINT_POSITION",
            action=action,
            controller_cfg=joint_cfg,
        )
        iterations += 1
        time.sleep(0.01)
    time.sleep(0.5)


def _die(msg: str, code: int = 1) -> None:
    print(f"[test_spacenav_admittance] error: {msg}", file=sys.stderr)
    try:
        rospy.logerr(msg)
    except Exception:
        pass
    sys.exit(code)


def _calib_get(p: dict, key: str, default=None):
    """Return a calibration value and fall back to the provided default."""
    if key in p and p[key] is not None:
        return p[key]
    return default


def _load_netft_calib(p: dict):
    """Parse calibration values from `/netft_calib_param`."""
    global _Ftf
    required = [
        "Fx0", "Fy0", "Fz0", "Mx0", "My0", "Mz0",
        "Lx", "Ly", "Lz", "mcx", "mcy", "mcz",
    ]
    missing = [k for k in required if k not in p or p[k] is None]
    if missing:
        raise KeyError(f"Missing required fields: {missing}")

    calib_params = [p[k] for k in required]

    srz = float(_calib_get(p, "sensor_rot_z_deg", 0.0))
    stz = float(_calib_get(p, "sensor_tz", 0.0))
    plen = float(_calib_get(p, "probe_length", 0.0))
    sprz = float(_calib_get(p, "sensor_to_probe_rot_z_deg", 0.0))
    sensor_rot_z_rad = np.deg2rad(srz)
    _Ftf = _build_ftf(plen, sprz)
    rospy.loginfo(
        "[test_spacenav_admittance] geometry terms (default to 0 when omitted): "
        "sensor_rot_z_deg=%.4f, sensor_tz=%.4f, probe_length=%.4f, sensor_to_probe_rot_z_deg=%.4f",
        srz, stz, plen, sprz,
    )
    return calib_params, sensor_rot_z_rad, stz


def main():
    global _Ftf
    args = _parse_args()
    runtime_cfg = YamlConfig(_resolve_config_path(args.controller_cfg)).as_easydict()
    _apply_runtime_config(runtime_cfg)

    rospy.init_node("test_spacenav_admittance", anonymous=False)

    if not os.path.isfile(NETFT_CALIB_YAML):
        _die(
            f"Calibration file not found: {NETFT_CALIB_YAML}\n"
            "  Set a valid path with: export PEIRASTIC_NETFT_CALIB_YAML=/path/to/netft_calib_result.yaml"
        )

    try:
        subprocess.run(["rosparam", "load", NETFT_CALIB_YAML], check=True)
    except subprocess.CalledProcessError as e:
        _die(
            f"`rosparam load` failed. Is `roscore` running? {e}\n"
            f"  file: {NETFT_CALIB_YAML}"
        )
    except FileNotFoundError:
        _die("`rosparam` command not found. Source /opt/ros/<distro>/setup.bash first.")

    try:
        p = rospy.get_param("/netft_calib_param")
    except KeyError:
        _die(
            "ROS parameter `/netft_calib_param` was not found.\n"
            "  Make sure `rosparam load` created that key, usually from a top-level YAML key named `netft_calib_param`."
        )
    except Exception as e:
        _die(
            f"Failed to read ROS parameters: {e!r}\n"
            "  Make sure `roscore` is running and `rosparam load` succeeded."
        )

    try:
        calib_params, sensor_rot_z_rad, sensor_tz = _load_netft_calib(p)
    except KeyError as e:
        _die(
            f"Calibration fields are missing: {e}\n"
            "  Required fields: Fx0,Fy0,Fz0,Mx0,My0,Mz0,Lx,Ly,Lz,mcx,mcy,mcz.\n"
            "  Optional geometry fields: sensor_rot_z_deg, sensor_tz, probe_length, sensor_to_probe_rot_z_deg."
        )

    rospy.Subscriber("/netft_data", WrenchStamped, _netft_cb, queue_size=1, tcp_nodelay=True)
    rospy.Subscriber("/spacenav/twist", Twist, _spacenav_cb, queue_size=1)

    robot_interface = FrankaInterface(
        CONFIG_FILE, control_freq=CONTROL_FREQ, use_visualizer=False
    )
    time.sleep(2.0)

    deadline = time.time() + 10.0
    while time.time() < deadline:
        if robot_interface.state_buffer_size > 0:
            break
        time.sleep(0.1)
    if robot_interface.last_state is None:
        robot_interface.close()
        _die(
            "No franka-interface state was received within 10 seconds (`last_state` is empty).\n"
            "  Start `franka-interface` on this machine and verify that the NUC IP and ports in `config/local-host.yml` match the running node."
        )

    eef_offset_rot = R_scipy.from_euler("xyz", EEF_OFFSET_RPY).as_matrix()
    go_to_init_joints(robot_interface)

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

    hold_action = [0.0] * 6 + [-1.0]
    for _ in range(30):
        robot_interface.control(controller_type="OSC_POSE", action=hold_action, controller_cfg=cfg)
        time.sleep(0.02)

    state = robot_interface.last_state
    T_init = np.array(state.O_T_EE).reshape(4, 4).T
    R_link8_init = T_init[:3, :3]
    pos_link8_init = T_init[:3, 3]
    start_eef_pos = pos_link8_init + R_link8_init @ EEF_OFFSET
    start_eef_rot = R_link8_init @ eef_offset_rot
    start_eef_quat = T.mat2quat(start_eef_rot)

    filtered_fz = None
    force_bias = 0.0
    force_samples = []
    t_bias_start = time.time()

    while time.time() - t_bias_start < BIAS_WAIT and not rospy.is_shutdown():
        robot_interface.control(controller_type="OSC_POSE", action=hold_action, controller_cfg=cfg)
        state = robot_interface.last_state
        if state is not None:
            T_cur = np.array(state.O_T_EE).reshape(4, 4).T
            R_link8 = T_cur[:3, :3]
            link8_rpy = R_scipy.from_matrix(R_link8).as_euler("xyz")
            rpy_v = _ft_sensor_frame(link8_rpy, sensor_rot_z_rad, sensor_tz)
            with _lock:
                raw_ft = _raw_ft.copy()
                have_force = _force_ready
            if have_force:
                fz_cal = _calibrate_fz(raw_ft, rpy_v, calib_params)
                if filtered_fz is None:
                    filtered_fz = fz_cal
                else:
                    filtered_fz = ALPHA_FORCE * filtered_fz + (1.0 - ALPHA_FORCE) * fz_cal
                force_samples.append(fz_cal)
        time.sleep(1.0 / CONTROL_FREQ)

    if force_samples:
        force_bias = float(np.mean(force_samples))
        filtered_fz = force_bias
    else:
        filtered_fz = 0.0

    nominal_pos = start_eef_pos.copy()
    nominal_quat = start_eef_quat.copy()
    adm_vel = 0.0
    fz_prev = 0.0
    dt = 1.0 / CONTROL_FREQ

    is_in_contact = False
    CONTACT_MAKE = float(runtime_cfg.contact.make_threshold)
    CONTACT_BREAK_SLOW = float(runtime_cfg.contact.break_slow_threshold)
    CONTACT_BREAK_FAST = float(runtime_cfg.contact.break_fast_threshold)

    break_contact_timer = 0.0
    CONTACT_BREAK_DELAY = float(runtime_cfg.contact.break_delay)

    current_kp_z = float(runtime_cfg.osc.kp_z.free_space)

    try:
        while not rospy.is_shutdown():
            loop_start = time.time()

            state = robot_interface.last_state
            if state is None:
                time.sleep(dt)
                continue

            T_cur = np.array(state.O_T_EE).reshape(4, 4).T
            R_link8 = T_cur[:3, :3]
            pos_link8 = T_cur[:3, 3]
            curr_eef_pos = pos_link8 + R_link8 @ EEF_OFFSET
            curr_eef_rot = R_link8 @ eef_offset_rot
            curr_eef_quat = T.mat2quat(curr_eef_rot)
            if np.dot(nominal_quat, curr_eef_quat) < 0.0:
                curr_eef_quat = -curr_eef_quat

            link8_rpy = R_scipy.from_matrix(R_link8).as_euler("xyz")
            rpy_v = _ft_sensor_frame(link8_rpy, sensor_rot_z_rad, sensor_tz)
            with _lock:
                raw_ft = _raw_ft.copy()
                sn_lin = _sn_lin.copy()
                sn_ang = _sn_ang.copy()
            fz_cal = _calibrate_fz(raw_ft, rpy_v, calib_params)
            filtered_fz = ALPHA_FORCE * filtered_fz + (1.0 - ALPHA_FORCE) * fz_cal

            nominal_pos[0] += sn_lin[0] * LIN_SCALE * dt
            nominal_pos[1] += sn_lin[1] * LIN_SCALE * dt

            user_pull_up = sn_lin[2] > float(runtime_cfg.teleop.user_pull_up_threshold)
            user_override = sn_lin[2] > float(runtime_cfg.teleop.user_override_threshold)

            if not is_in_contact:
                nominal_pos[2] += sn_lin[2] * LIN_SCALE * dt
            elif sn_lin[2] > 0:
                nominal_pos[2] += sn_lin[2] * LIN_SCALE * dt

            omega_eef = np.array([sn_ang[0], -sn_ang[1], -sn_ang[2]]) * ROT_SCALE
            nominal_quat = integrate_rotation(nominal_quat, omega_eef, dt)

            fz_comp = filtered_fz - force_bias

            if not is_in_contact and fz_comp < CONTACT_MAKE and not user_override:
                is_in_contact = True
                break_contact_timer = 0.0
            elif is_in_contact:
                if user_override:
                    is_in_contact = False
                    break_contact_timer = 0.0
                elif fz_comp > CONTACT_BREAK_FAST:
                    is_in_contact = False
                    break_contact_timer = 0.0
                elif fz_comp > CONTACT_BREAK_SLOW:
                    break_contact_timer += dt
                    if break_contact_timer > CONTACT_BREAK_DELAY:
                        is_in_contact = False
                        break_contact_timer = 0.0
                else:
                    break_contact_timer = 0.0

            fz_dot = (fz_comp - fz_prev) / dt
            fz_prev = fz_comp

            if is_in_contact:
                target_kp_z = float(runtime_cfg.osc.kp_z.contact)
                f_error = fz_comp - FORCE_TARGET_Z

                if abs(f_error) < FORCE_DEADBAND:
                    f_error = 0.0

                if user_pull_up:
                    f_error = 0.0

                Dd_eff = Dd_up if adm_vel > 0 else Dd_down
                v_dot = (f_error - Kdf * fz_dot - Dd_eff * adm_vel) / Md
                v_dot = float(np.clip(v_dot, -ACCEL_LIMIT, ACCEL_LIMIT))
            else:
                target_kp_z = float(runtime_cfg.osc.kp_z.free_space)
                f_error = 0.0
                v_dot = (-200.0 * adm_vel - Dd_down * adm_vel) / Md

            adm_vel += v_dot * dt
            adm_vel = float(np.clip(adm_vel, -MAX_ADM_VEL, MAX_ADM_VEL))

            z_eef_base = T.quat2mat(curr_eef_quat)[:, 2]
            adm_vel_base = adm_vel * z_eef_base
            nominal_pos += adm_vel_base * dt
            z_displacement = nominal_pos[2] - start_eef_pos[2]
            z_displacement = float(np.clip(z_displacement, -MAX_ADM_POS, MAX_ADM_POS))
            nominal_pos[2] = start_eef_pos[2] + z_displacement

            kp_z_alpha = float(runtime_cfg.osc.kp_z.smoothing_alpha)
            current_kp_z = (1.0 - kp_z_alpha) * current_kp_z + kp_z_alpha * target_kp_z
            cfg["Kp"]["translation"] = [
                float(runtime_cfg.osc.kp_translation[0]),
                float(runtime_cfg.osc.kp_translation[1]),
                float(current_kp_z),
            ]

            if is_in_contact:
                diff_xy = nominal_pos[:2] - curr_eef_pos[:2]
                dist_xy = np.linalg.norm(diff_xy)
                if dist_xy > MAX_POS_ERR:
                    nominal_pos[:2] = curr_eef_pos[:2] + diff_xy * (MAX_POS_ERR / dist_xy)
            elif np.linalg.norm(sn_lin) > 0:
                pos_diff = nominal_pos - curr_eef_pos
                pos_dist = np.linalg.norm(pos_diff)
                if pos_dist > MAX_POS_ERR:
                    nominal_pos = curr_eef_pos + pos_diff * (MAX_POS_ERR / pos_dist)

            if np.linalg.norm(sn_ang) > 0:
                r_nom = R_scipy.from_quat(nominal_quat)
                r_cur = R_scipy.from_quat(curr_eef_quat)
                r_diff = r_nom * r_cur.inv()
                rotvec = r_diff.as_rotvec()
                angle = np.linalg.norm(rotvec)
                if angle > MAX_ROT_ERR:
                    r_nom_new = R_scipy.from_rotvec(rotvec * (MAX_ROT_ERR / angle)) * r_cur
                    nominal_quat = r_nom_new.as_quat()
                    if nominal_quat[3] < 0:
                        nominal_quat = -nominal_quat

            error_base = (nominal_pos - curr_eef_pos).flatten()
            max_pos_step = float(cfg["action_scale"]["translation"])
            action_pos = np.clip(error_base, -max_pos_step, max_pos_step)

            r_nom_obj = R_scipy.from_quat(nominal_quat)
            r_cur_obj = R_scipy.from_quat(curr_eef_quat)
            r_err_base = r_nom_obj * r_cur_obj.inv()
            rotvec_base = r_err_base.as_rotvec()

            max_rot_step = float(cfg["action_scale"]["rotation"])
            action_rot = np.clip(rotvec_base, -max_rot_step, max_rot_step)

            action = list(np.concatenate([action_pos, action_rot])) + [-1.0]

            robot_interface.control(
                controller_type="OSC_POSE",
                action=action,
                controller_cfg=cfg,
            )

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        pass
    finally:
        robot_interface.close()


if __name__ == "__main__":
    main()
