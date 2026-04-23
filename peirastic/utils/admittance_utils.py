"""Reusable helpers for NetFT-based admittance scripts."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy


def canonicalize_quaternion(
    quat_xyzw: np.ndarray, reference_quat_xyzw: np.ndarray | None = None
) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4).copy()
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    quat /= norm
    if reference_quat_xyzw is not None:
        ref = np.asarray(reference_quat_xyzw, dtype=np.float64).reshape(4)
        if float(np.dot(quat, ref)) < 0.0:
            quat *= -1.0
    elif quat[3] < 0.0:
        quat *= -1.0
    return quat


def integrate_body_rotation(
    quat_xyzw: np.ndarray, omega_body: np.ndarray, dt: float
) -> np.ndarray:
    omega = np.asarray(omega_body, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(omega) * dt)
    if angle < 1e-9:
        return canonicalize_quaternion(quat_xyzw)
    axis = omega / np.linalg.norm(omega)
    delta = R_scipy.from_rotvec(axis * angle)
    updated = (R_scipy.from_quat(quat_xyzw) * delta).as_quat()
    return canonicalize_quaternion(updated, reference_quat_xyzw=quat_xyzw)


def _trans_mtx(x: float, y: float, z: float) -> np.ndarray:
    mtx = np.eye(4, dtype=np.float64)
    mtx[:3, 3] = [x, y, z]
    return mtx


def _euler_mtx_sxyz(rx: float, ry: float, rz: float) -> np.ndarray:
    mtx = np.eye(4, dtype=np.float64)
    mtx[:3, :3] = R_scipy.from_euler("xyz", [rx, ry, rz]).as_matrix()
    return mtx


def _rot_z_4x4(angle: float) -> np.ndarray:
    mtx = np.eye(4, dtype=np.float64)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    mtx[0, 0], mtx[0, 1], mtx[1, 0], mtx[1, 1] = c, -s, s, c
    return mtx


def _euler_from_mtx_sxyz(mtx: np.ndarray) -> np.ndarray:
    return R_scipy.from_matrix(mtx[:3, :3]).as_euler("xyz")


def ft_sensor_frame(
    rpy_xyz: np.ndarray, sensor_rot_z_rad: float, sensor_tz: float
) -> np.ndarray:
    tf_tmp = _euler_mtx_sxyz(*np.asarray(rpy_xyz, dtype=np.float64).reshape(3))
    tf_tmp = tf_tmp @ _trans_mtx(0.0, 0.0, sensor_tz) @ _rot_z_4x4(sensor_rot_z_rad)
    return _euler_from_mtx_sxyz(tf_tmp)


def _rotation_base_to_end(gamma: float, beta: float, alpha: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    return np.array(
        [
            [ca * cb, sa * cb, -sb],
            [ca * sb * sg - sa * cg, sa * sb * sg + ca * cg, cb * sg],
            [ca * sb * cg + sa * sg, sa * sb * cg - ca * sg, cb * cg],
        ],
        dtype=np.float64,
    )


def build_ftf(probe_length: float, sensor_to_probe_rot_z_deg: float) -> np.ndarray:
    theta = np.deg2rad(sensor_to_probe_rot_z_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    p_sensor = np.array([0.0, 0.0, -probe_length], dtype=np.float64)
    p_probe = rot @ p_sensor
    p_cross = np.array(
        [
            [0.0, -p_probe[2], p_probe[1]],
            [p_probe[2], 0.0, -p_probe[0]],
            [-p_probe[1], p_probe[0], 0.0],
        ],
        dtype=np.float64,
    )
    ftf = np.zeros((6, 6), dtype=np.float64)
    ftf[:3, :3] = rot
    ftf[3:, :3] = p_cross @ rot
    ftf[3:, 3:] = rot
    return ftf


def load_netft_calib_param(param_dict: dict) -> tuple[list[float], float, float, np.ndarray]:
    required = [
        "Fx0",
        "Fy0",
        "Fz0",
        "Mx0",
        "My0",
        "Mz0",
        "Lx",
        "Ly",
        "Lz",
        "mcx",
        "mcy",
        "mcz",
    ]
    missing = [key for key in required if key not in param_dict or param_dict[key] is None]
    if missing:
        raise KeyError(f"Missing required fields: {missing}")

    calib_params = [float(param_dict[key]) for key in required]
    sensor_rot_z_deg = float(param_dict.get("sensor_rot_z_deg", 0.0) or 0.0)
    sensor_tz = float(param_dict.get("sensor_tz", 0.0) or 0.0)
    probe_length = float(param_dict.get("probe_length", 0.0) or 0.0)
    sensor_to_probe_rot_z_deg = float(
        param_dict.get("sensor_to_probe_rot_z_deg", 0.0) or 0.0
    )
    ftf = build_ftf(probe_length, sensor_to_probe_rot_z_deg)
    return calib_params, np.deg2rad(sensor_rot_z_deg), sensor_tz, ftf


def calibrate_tool_z_force(
    raw_ft: np.ndarray,
    rpy_xyz: np.ndarray,
    calib_params: list[float],
    ftf: np.ndarray,
) -> float:
    fx0, fy0, fz0, mx0, my0, mz0, lx, ly, lz, cmx, cmy, cmz = calib_params
    fx, fy, fz, mx, my, mz = np.asarray(raw_ft, dtype=np.float64).reshape(6)
    gamma, beta, alpha = np.asarray(rpy_xyz, dtype=np.float64).reshape(3)
    gravity = _rotation_base_to_end(gamma, beta, alpha) @ np.array(
        [lx, ly, lz], dtype=np.float64
    )
    gx, gy, gz = gravity
    compensated = np.array(
        [
            fx - fx0 - gx,
            fy - fy0 - gy,
            fz - fz0 - gz,
            mx - mx0 - (gz * cmy - gy * cmz),
            my - my0 - (gx * cmz - gz * cmx),
            mz - mz0 - (gy * cmx - gx * cmy),
        ],
        dtype=np.float64,
    ).reshape(6, 1)
    force_tool = ftf @ compensated
    return float(force_tool[2, 0])
