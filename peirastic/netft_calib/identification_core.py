#!/usr/bin/env python3
"""Identification math ported from netft_calib/scripts/identification.py (ROS tf removed)."""

from __future__ import annotations

import os
import sys

import numpy as np
import yaml

from peirastic.netft_calib import tf_transformations as t
from peirastic.utils.runtime_paths import get_netft_workspace


def txt_data_2_lists(filename):
    with open(filename, "r", encoding="utf-8") as f:
        txt_list = f.readlines()[1::]
        data_list = []
        for txt_line in txt_list:
            txt_line = txt_line[0:-1]
            data_line = [float(entry) for entry in txt_line.split("\t")]
            data_list.append(data_line)
    return data_list


def extract_lists(data_lists):
    r_lists = []
    f_lists = []
    t_lists = []
    for line in data_lists:
        [rx, ry, rz, fx, fy, fz, tx, ty, tz] = line
        r_lists.append([rx, ry, rz])
        f_lists.append([fx, fy, fz])
        t_lists.append([tx, ty, tz])
    return r_lists, f_lists, t_lists


def ft_mass_center_identify(f_lists, t_lists):
    F = []
    m = []
    for f_line in f_lists:
        fx, fy, fz = f_line
        f_mat = [[0, fz, -fy, 1, 0, 0], [-fz, 0, fx, 0, 1, 0], [fy, -fx, 0, 0, 0, 1]]
        F.extend(f_mat)
    for t_line in t_lists:
        tx, ty, tz = t_line
        m.extend([[tx], [ty], [tz]])
    p_array = np.dot(np.linalg.pinv(np.array(F)), np.array(m))
    return p_array.transpose()[0].tolist()


def ft_sensor_bias_identify_whole(r_lists, f_lists):
    R = []
    f = []
    for rpy in r_lists:
        rpy = ft_sensor_frame(rpy)
        gamma, beta, alpha = rpy
        R_2_1 = R_regression(gamma, beta, alpha)
        R.extend(R_2_1)
    for line in f_lists:
        f.extend([[line[0]], [line[1]], [line[2]]])
    res_array = np.dot(np.linalg.pinv(np.array(R)), np.array(f))
    return res_array.transpose()[0].tolist()


def trans_mtx(x, y, z):
    res = np.eye(4)
    res[:3, 3] = np.array([x, y, z])
    return res


def R_regression(gamma, beta, alpha):
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)
    sg = np.sin(gamma)
    cg = np.cos(gamma)
    R_2_1 = [
        [ca * cb, sa * cb, -sb, 1, 0, 0],
        [ca * sb * sg - sa * cg, sa * sb * sg + ca * cg, cb * sg, 0, 1, 0],
        [ca * sb * cg + sa * sg, sa * sb * cg - ca * sg, cb * cg, 0, 0, 1],
    ]
    return R_2_1


def rotation_base_2_end(gamma, beta, alpha):
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)
    sg = np.sin(gamma)
    cg = np.cos(gamma)
    R_ee_b = [
        [ca * cb, sa * cb, -sb],
        [ca * sb * sg - sa * cg, sa * sb * sg + ca * cg, cb * sg],
        [ca * sb * cg + sa * sg, sa * sb * cg - ca * sg, cb * cg],
    ]
    return R_ee_b


def ft_sensor_bias_identify_simple(r_lists, f_lists):
    R = []
    f = []
    for rpy in r_lists:
        gamma, beta, alpha = rpy
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cb = np.cos(beta)
        sb = np.sin(beta)
        sg = np.sin(gamma)
        cg = np.cos(gamma)
        R_2_1 = [
            [sa * cb, -sb, 1, 0, 0],
            [sa * sb * sg + ca * cg, cb * sg, 0, 1, 0],
            [sa * sb * cg - ca * sg, cb * cg, 0, 0, 1],
        ]
        R.extend(R_2_1)
    for line in f_lists:
        f.extend([[line[0]], [line[1]], [line[2]]])
    res_array = np.dot(np.linalg.pinv(np.array(R)), np.array(f))
    return res_array.transpose()[0].tolist()


def identification_result(r_list1, r_list2, simple_version=True):
    [mcx, mcy, mcz, k1, k2, k3] = r_list1
    if simple_version:
        [g0, g1, Fx0, Fy0, Fz0] = r_list2
        [Lx, Ly, Lz] = [0, g0, g1]
        G = np.sqrt(Lx * Lx + Ly * Ly + Lz * Lz)
    else:
        [Lx, Ly, Lz, Fx0, Fy0, Fz0] = r_list2
        G = np.sqrt(Lx * Lx + Ly * Ly + Lz * Lz)

    U = np.arcsin(-Ly / G) / np.pi * 180.0
    V = np.arctan(-Lx / Lz) / np.pi * 180.0

    Mx0 = k1 - Fy0 * mcz + Fz0 * mcy
    My0 = k2 - Fz0 * mcx + Fx0 * mcz
    Mz0 = k3 - Fx0 * mcy + Fy0 * mcx

    if not simple_version:
        print(
            "-------------------------- identification results (whole version)------------------------------"
        )
    else:
        print(
            "------------------------- identification results (simple version)------------------------------"
        )
    print("identify results: Lx:" + str(Lx) + ", Ly:" + str(Ly) + ", Lz:" + str(Lz))
    print("- Gravity of prob  : " + str(G) + " N\t" + str(G / 9.81 * 1000) + " grams (g=9.81 m/s^2)")
    print("- Installation tilt: [U: " + str(U) + " deg \tV: " + str(V) + " deg]")
    print("- Mass of center   : [x: " + str(mcx) + "\ty: " + str(mcy) + "\tz: " + str(mcz) + "]")
    print(
        "- F/T zero values  : [Fx0: "
        + str(Fx0)
        + "\tFy0: "
        + str(Fy0)
        + "\tFz0: "
        + str(Fz0)
        + "\n                      Mx0: "
        + str(Mx0)
        + "\tMy0: "
        + str(My0)
        + "\tMz0: "
        + str(Mz0)
        + "]"
    )
    print("-----------------------------------------------------------------------------------------------")
    return [Fx0, Fy0, Fz0, Mx0, My0, Mz0, Lx, Ly, Lz]


def calibration_simple(measured_ft, curr_rpy, identify_params):
    [Fx, Fy, Fz, Mx, My, Mz] = measured_ft
    [Fx0, Fy0, Fz0, Mx0, My0, Mz0, Lx, Ly, Lz, cmx, cmy, cmz] = identify_params

    gamma, beta, alpha = curr_rpy
    R_ee_b = rotation_base_2_end(gamma, beta, alpha)
    G_arr = np.dot(np.array(R_ee_b), np.array([[Lx], [Ly], [Lz]]))
    [Gx, Gy, Gz] = G_arr.transpose()[0].tolist()

    Mgx = Gz * cmy - Gy * cmz
    Mgy = Gx * cmz - Gz * cmx
    Mgz = Gy * cmx - Gx * cmy

    Fex = Fx - Fx0 - Gx
    Fey = Fy - Fy0 - Gy
    Fez = Fz - Fz0 - Gz

    Mex = Mx - Mx0 - Mgx
    Mey = My - My0 - Mgy
    Mez = Mz - Mz0 - Mgz
    return [Fex, Fey, Fez, Mex, Mey, Mez]


def ft_sensor_frame(rpy):
    tf_tmp = t.euler_matrix(*rpy)
    rz = t.rotation_matrix(-105 * np.pi / 180.0, [0, 0, 1])
    tz = trans_mtx(0.0, 0.0, 0.055)
    tf_tmp = np.dot(tf_tmp, np.dot(tz, rz))
    rpy = t.euler_from_matrix(tf_tmp)
    return rpy


def set_calb_mtx_into_yaml(file_path, identify_params):
    d = dict()
    d["netft_calib_param"] = dict()
    [Fx0, Fy0, Fz0, Mx0, My0, Mz0, Lx, Ly, Lz, mcx, mcy, mcz] = identify_params
    with open(file_path, "w", encoding="utf-8") as f:
        d["netft_calib_param"]["Fx0"] = Fx0
        d["netft_calib_param"]["Fy0"] = Fy0
        d["netft_calib_param"]["Fz0"] = Fz0
        d["netft_calib_param"]["Mx0"] = Mx0
        d["netft_calib_param"]["My0"] = My0
        d["netft_calib_param"]["Mz0"] = Mz0
        d["netft_calib_param"]["Lx"] = Lx
        d["netft_calib_param"]["Ly"] = Ly
        d["netft_calib_param"]["Lz"] = Lz
        d["netft_calib_param"]["mcx"] = mcx
        d["netft_calib_param"]["mcy"] = mcy
        d["netft_calib_param"]["mcz"] = mcz
        yaml.dump(d, f)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv
    simple_version = False
    path_prefix = get_netft_workspace(argv[1] if len(argv) > 1 else None)
    filepath = os.path.join(path_prefix, "log") + "/"
    filename = "write_data_identify.txt"
    data_lists = txt_data_2_lists(filepath + filename)
    rpy_lists, force_lists, torque_lists = extract_lists(data_lists)
    [mcx, mcy, mcz, k1, k2, k3] = ft_mass_center_identify(force_lists, torque_lists)
    res1 = [mcx, mcy, mcz, k1, k2, k3]
    res_simple = ft_sensor_bias_identify_simple(rpy_lists, force_lists)
    res_whole = ft_sensor_bias_identify_whole(rpy_lists, force_lists)
    if simple_version:
        [Fx0, Fy0, Fz0, Mx0, My0, Mz0, Lx, Ly, Lz] = identification_result(res1, res_simple, True)
    else:
        [Fx0, Fy0, Fz0, Mx0, My0, Mz0, Lx, Ly, Lz] = identification_result(res1, res_whole, False)

    identify_params = [Fx0, Fy0, Fz0, Mx0, My0, Mz0, Lx, Ly, Lz, mcx, mcy, mcz]
    yaml_file_path = os.path.join(path_prefix, "config", "netft_calib_result.yaml")
    os.makedirs(os.path.dirname(yaml_file_path), exist_ok=True)
    set_calb_mtx_into_yaml(yaml_file_path, identify_params)
    print("Data processed done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
