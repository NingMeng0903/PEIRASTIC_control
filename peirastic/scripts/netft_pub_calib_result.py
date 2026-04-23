#!/usr/bin/env python3
"""Publish calibrated wrench (ported from netft_calib/scripts/pub_calib_result.py)."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import rospy
from geometry_msgs.msg import WrenchStamped

from peirastic import config_root
from peirastic.franka_interface import FrankaInterface
from peirastic.netft_calib.identification_core import calibration_simple, ft_sensor_frame
from peirastic.utils import transform_utils

Fext = [0, 0, 0, 0, 0, 0]


def sub_f_ext_cb(msg):
    global Fext
    Fext = [
        msg.wrench.force.x,
        msg.wrench.force.y,
        msg.wrench.force.z,
        msg.wrench.torque.x,
        msg.wrench.torque.y,
        msg.wrench.torque.z,
    ]


def _rpy_panda_link8(robot_interface: FrankaInterface):
    pose = robot_interface.last_eef_pose
    if pose is None:
        return None
    rot = pose[:3, :3]
    return transform_utils.mat2euler(rot, axes="sxyz")


def _parse_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface-cfg",
        type=str,
        default=os.environ.get("PEIRASTIC_INTERFACE_CFG", "local-host.yml"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = _parse_args(argv)

    rospy.init_node("FTCalibration", anonymous=True)

    robot_interface = FrankaInterface(config_root + f"/{args.interface_cfg}", use_visualizer=False)
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if robot_interface.state_buffer_size > 0:
            break
        time.sleep(0.1)
    if robot_interface.last_state is None:
        robot_interface.close()
        rospy.logfatal("No robot state received from franka-interface.")
        return 2

    try:
        netft_calib_param = rospy.get_param("/netft_calib_param")
        fx0 = netft_calib_param["Fx0"]
        fy0 = netft_calib_param["Fy0"]
        fz0 = netft_calib_param["Fz0"]
        lx = netft_calib_param["Lx"]
        ly = netft_calib_param["Ly"]
        lz = netft_calib_param["Lz"]
        mx0 = netft_calib_param["Mx0"]
        my0 = netft_calib_param["My0"]
        mz0 = netft_calib_param["Mz0"]
        mcx = netft_calib_param["mcx"]
        mcy = netft_calib_param["mcy"]
        mcz = netft_calib_param["mcz"]
        identify_params = [fx0, fy0, fz0, mx0, my0, mz0, lx, ly, lz, mcx, mcy, mcz]
        rospy.loginfo("Successfully loaded FT calibration parameters.")
        rospy.loginfo(identify_params)

        rate = rospy.Rate(400)
        _sub_fext = rospy.Subscriber("/netft_data", WrenchStamped, sub_f_ext_cb, queue_size=1, tcp_nodelay=True)
        _pub_cali_fext = rospy.Publisher("/franka_state_controller/Cali_F_ext", WrenchStamped, queue_size=1)

        rospy.loginfo("FT Calibration node is running...")

        while not rospy.is_shutdown():
            try:
                rpy_v = _rpy_panda_link8(robot_interface)
                if rpy_v is None:
                    rate.sleep()
                    continue
                rpy_v = ft_sensor_frame(list(rpy_v))

                measured_ft = list(Fext)
                [fex, fey, fez, mex, mey, mez] = calibration_simple(measured_ft, rpy_v, identify_params)

                ftf = np.array(
                    [
                        [7.10137241e-17, -1.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                        [1.00000000e00, 9.07576257e-17, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                        [0.00000000e00, 0.00000000e00, 1.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00],
                        [1.85000000e-01, 1.67901608e-17, 0.00000000e00, 7.10137241e-17, -1.00000000e00, 0.00000000e00],
                        [-1.31375390e-17, 1.85000000e-01, 0.00000000e00, 1.00000000e00, 9.07576257e-17, 0.00000000e00],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                )

                fmeas = np.array([[fex], [fey], [fez], [mex], [mey], [mez]])
                fcali = np.dot(ftf, fmeas)

                cali_fext = WrenchStamped()
                cali_fext.header.stamp = rospy.Time.now()
                cali_fext.header.frame_id = "panda_probe_ee"
                cali_fext.wrench.force.x = float(fcali[0, 0])
                cali_fext.wrench.force.y = float(fcali[1, 0])
                cali_fext.wrench.force.z = float(fcali[2, 0])
                cali_fext.wrench.torque.x = float(fcali[3, 0])
                cali_fext.wrench.torque.y = float(fcali[4, 0])
                cali_fext.wrench.torque.z = float(fcali[5, 0])
                _pub_cali_fext.publish(cali_fext)
            except Exception as exc:
                rospy.logwarn("Could not publish calibrated wrench. Error: %s", exc)

            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        rospy.logfatal("FT Calibration node failed to initialize. Error: %s", exc)
        return 1
    finally:
        try:
            robot_interface.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
