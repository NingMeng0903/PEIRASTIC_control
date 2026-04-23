#!/usr/bin/env python3
"""NetFT calibration data acquisition via FrankaInterface (ZMQ); ROS used for /netft_data only."""

from __future__ import annotations

import argparse
import os
import pkgutil
import sys
import time
from typing import List, Sequence

import numpy as np
import rospy
from geometry_msgs.msg import WrenchStamped

from peirastic import config_root
from peirastic.franka_interface import FrankaInterface
from peirastic.netft_calib import named_joint_states as named_js
from peirastic.utils.runtime_paths import get_netft_workspace
from peirastic.utils import YamlConfig
from peirastic.utils import transform_utils
from peirastic.utils.ros_netft_launch import (
    reset_netft_workspace_artifacts,
    start_netft_roslaunch,
    stop_netft_roslaunch,
    wait_for_netft_topic,
)


def _read_bundled_netft_joint_cmds_txt() -> str:
    data = pkgutil.get_data("peirastic.netft_calib", "config/joint_cmds.txt")
    if data is None:
        raise RuntimeError("Bundled joint_cmds.txt not found in peirastic.netft_calib package")
    return data.decode("utf-8")


def _rpy_panda_link8_from_state(robot_interface: FrankaInterface):
    pose = robot_interface.last_eef_pose
    if pose is None:
        return None
    rot = pose[:3, :3]
    return transform_utils.mat2euler(rot, axes="sxyz")


def _rospy_sleep_up_to(period_s: float, step_s: float = 0.02) -> bool:
    """Sleep up to period_s in small rospy.sleep steps; False if shutdown requested."""
    if period_s <= 0.0:
        return not rospy.is_shutdown()
    end = time.time() + period_s
    while time.time() < end:
        if rospy.is_shutdown():
            return False
        rospy.sleep(min(step_s, end - time.time()))
    return True


class ForceCaliDataAcq:
    def __init__(
        self,
        path_prefix: str,
        end_effector_link: str,
        robot_interface: FrankaInterface,
        joint_controller_cfg,
        named_states: dict,
        *,
        joint_cmd_move_timeout_s: float = 10.0,
        netft_samples_timeout_s: float = 5.0,
    ):
        rospy.loginfo("Force calibration data acquisition started")

        self.path_prefix = path_prefix
        self.ee_link = end_effector_link
        self.robot_interface = robot_interface
        self.joint_controller_cfg = joint_controller_cfg
        self.named_states = named_states
        self.joint_cmd_move_timeout_s = float(joint_cmd_move_timeout_s)
        self.netft_samples_timeout_s = float(netft_samples_timeout_s)

        self.fext: List[float] = []
        self.fext_avg = [0.0] * 6
        self.fext_meas_cnt = 0
        self.fext_meas_done = False
        self.fext_meas_avg_num = 20
        self.rpy_avg = [0.0, 0.0, 0.0]

        self._sub_fext = rospy.Subscriber(
            "/netft_data", WrenchStamped, self.sub_f_ext_cb, queue_size=1, tcp_nodelay=True
        )

        self.log_states_file = f"{path_prefix}/log/write_data_identify.txt"
        self.log_joints_file = f"{path_prefix}/log/position_data.txt"
        self.init_log_files()

    def _send_joint_hold(self, joints7: Sequence[float]) -> bool:
        """Keep the ZMQ control stream alive while holding the current waypoint."""
        self.robot_interface.control(
            controller_type="JOINT_POSITION",
            action=list(joints7) + [-1.0],
            controller_cfg=self.joint_controller_cfg,
            shutdown_check=rospy.is_shutdown,
        )
        return not rospy.is_shutdown()

    def _hold_joint_target_for(
        self, joints7: Sequence[float], duration_s: float, step_s: float = 0.02
    ) -> bool:
        end = time.time() + duration_s
        while time.time() < end:
            if not self._send_joint_hold(joints7):
                return False
            if not _rospy_sleep_up_to(min(step_s, end - time.time()), step_s=step_s):
                return False
        return not rospy.is_shutdown()

    def _move_to_joint_target(
        self,
        joints7: Sequence[float],
        timeout_s: float = 5.0,
        position_tol: float = 5e-3,
        reopen_control_after_idle: bool = False,
    ) -> bool:
        if reopen_control_after_idle:
            rospy.loginfo("Re-arming control after sample pause.")
            self.robot_interface.force_next_control_preprocess()
        target_q = np.array(joints7, dtype=np.float64)
        action = list(joints7) + [-1.0]
        start = time.time()
        while not rospy.is_shutdown() and (time.time() - start < timeout_s):
            state = self.robot_interface.last_state
            if state is not None:
                q = np.array(state.q, dtype=np.float64)
                if float(np.max(np.abs(q - target_q))) < position_tol:
                    return True
            self.robot_interface.control(
                controller_type="JOINT_POSITION",
                action=action,
                controller_cfg=self.joint_controller_cfg,
                shutdown_check=rospy.is_shutdown,
            )
            if rospy.is_shutdown():
                return False
            if not _rospy_sleep_up_to(0.01, step_s=0.01):
                return False
        state = self.robot_interface.last_state
        if state is None:
            rospy.logerr(
                "Joint move timeout: no robot state (is franka-interface running and ZMQ "
                "reachable per your --interface-cfg?)."
            )
        else:
            q = np.array(state.q, dtype=np.float64)
            err = float(np.max(np.abs(q - target_q)))
            rospy.logerr(
                "Joint move timeout (%.1fs): max_abs_err=%.5f rad (tol=%.5f). "
                "target_q=%s current_q=%s",
                timeout_s,
                err,
                position_tol,
                np.array2string(target_q, precision=4, separator=", "),
                np.array2string(q, precision=4, separator=", "),
            )
        return False

    def joint_pos_init(self, group_name: str, *, after_idle: bool = False) -> bool:
        if group_name not in ["ready", "ftcalib_jgroup"]:
            rospy.loginfo("No moving! Valid names: ready / ftcalib_jgroup")
            return False
        joints = named_js.get_named_joints(self.named_states, group_name)
        ok = self._move_to_joint_target(
            joints,
            timeout_s=120.0,
            position_tol=5e-3,
            reopen_control_after_idle=after_idle,
        )
        if not ok:
            rospy.logerr(
                "Failed to reach named joint target %r. Script exits (single-shot). "
                "Unlock brakes / clear errors on teach pendant, confirm ROBOT.IP in YAML.",
                group_name,
            )
            return False
        return True

    def run_data_acquisition(self, init_group_name: str):
        rospy.loginfo(
            "Running joint_cmds from %s/config/joint_cmds.txt, then returning to named pose %r.",
            self.path_prefix,
            init_group_name,
        )
        self.acquisition_with_joint_commands(init_group_name)

    def acquisition_with_joint_commands(self, init_group_name: str):
        joint_cmds_file = f"{self.path_prefix}/config/joint_cmds.txt"
        joint_cmd_list = self.txt_to_list(joint_cmds_file)
        n = len(joint_cmd_list)
        rospy.loginfo("Loaded %d joint waypoints from %s", n, joint_cmds_file)
        if n == 0:
            rospy.logerr(
                "No valid joint rows in joint_cmds.txt (need 7 numbers per line). "
                "Delete the file and re-run with --seed-joint-cmds-from-package, or fix the file."
            )
            return
        move_timeout_s = self.joint_cmd_move_timeout_s
        rospy.loginfo("Per-waypoint move timeout: %.1fs (override with --joint-cmd-move-timeout)", move_timeout_s)
        for i, cmd in enumerate(joint_cmd_list):
            if rospy.is_shutdown():
                rospy.loginfo("Shutdown requested; stopping joint_cmds before waypoint %d/%d.", i + 1, n)
                return
            rospy.loginfo("joint_cmds Step %d/%d: moving, then recording sample", i + 1, n)
            ok = self._move_to_joint_target(
                cmd,
                timeout_s=move_timeout_s,
                position_tol=5e-3,
                reopen_control_after_idle=(i >= 1),
            )
            if not ok:
                rospy.logwarn(
                    "Waypoint %d/%d: pose not reached in %.1fs; skipping record.",
                    i + 1,
                    n,
                    move_timeout_s,
                )
                continue
            self.record_data(cmd)
            rospy.loginfo("joint_cmds Step %d/%d: sample written", i + 1, n)
        if not rospy.is_shutdown():
            self.joint_pos_init(init_group_name, after_idle=True)

    def txt_to_list(self, filename: str):
        """Parse joint_cmds.txt: any whitespace or comma-separated; skip header / bad lines."""
        data_list: List[List[float]] = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) != 7:
                    continue
                try:
                    data_list.append([float(x) for x in parts])
                except ValueError:
                    continue
        return data_list

    def sub_f_ext_cb(self, msg: WrenchStamped):
        self.fext = [
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z,
        ]
        if not self.fext_meas_done and self.fext_meas_cnt < self.fext_meas_avg_num:
            for i, force in enumerate(self.fext):
                self.fext_avg[i] = (self.fext_meas_cnt * self.fext_avg[i] + force) / (self.fext_meas_cnt + 1)
            self.fext_meas_cnt += 1

    def state_str(self, hold_joints7: Sequence[float]):
        self.fext_avg = [0.0] * 6
        self.fext_meas_cnt = 0
        self.fext_meas_done = False

        start_time = time.time()
        while (
            not rospy.is_shutdown()
            and not self.fext_meas_done
            and (time.time() - start_time < self.netft_samples_timeout_s)
        ):
            if self.fext_meas_cnt >= self.fext_meas_avg_num:
                self.fext_meas_done = True
                break
            if not self._send_joint_hold(hold_joints7):
                return ""
            if not _rospy_sleep_up_to(0.01, step_s=0.01):
                return ""
        if rospy.is_shutdown():
            return ""
        if not self.fext_meas_done:
            rospy.logwarn(
                "Force averaging timeout (%d /netft_data samples in %.1fs); check netft_node.",
                self.fext_meas_cnt,
                self.netft_samples_timeout_s,
            )
            self.fext_meas_done = True

        self.rpy_avg = [0.0, 0.0, 0.0]
        for _ in range(self.fext_meas_avg_num):
            if rospy.is_shutdown():
                return ""
            if not self._send_joint_hold(hold_joints7):
                return ""
            if not _rospy_sleep_up_to(1.0 / 50.0, step_s=0.01):
                return ""
            rpy_tmp = _rpy_panda_link8_from_state(self.robot_interface)
            if rpy_tmp is None:
                continue
            for i in range(3):
                self.rpy_avg[i] += float(rpy_tmp[i])
        self.rpy_avg = [x / self.fext_meas_avg_num for x in self.rpy_avg]

        rpy_str = "\t".join(f"{x:.6f}" for x in self.rpy_avg)
        fext_str = "\t".join(f"{x:.6f}" for x in self.fext_avg)
        return f"{rpy_str}\t{fext_str}\n"

    def joint_str(self):
        joint_vals = self.robot_interface.last_q
        if joint_vals is None:
            return "\t".join(["0.000000"] * 7) + "\n"
        return "\t".join(f"{float(x):.6f}" for x in joint_vals) + "\n"

    def init_log_files(self):
        os.makedirs(os.path.dirname(self.log_states_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_joints_file), exist_ok=True)
        with open(self.log_states_file, "w", encoding="utf-8") as f:
            f.write("rpy.x\trpy.y\trpy.z\tFext.x\tFext.y\tFext.z\tText.x\tText.y\tText.z\n")
        with open(self.log_joints_file, "w", encoding="utf-8") as f:
            f.write("q1\tq2\tq3\tq4\tq5\tq6\tq7\n")

    def record_data(self, hold_joints7: Sequence[float]):
        if not self._hold_joint_target_for(hold_joints7, duration_s=3.0, step_s=0.02):
            return
        if rospy.is_shutdown():
            return
        line = self.state_str(hold_joints7)
        if rospy.is_shutdown() or not line:
            return
        with open(self.log_states_file, "a", encoding="utf-8") as f:
            f.write(line)
        with open(self.log_joints_file, "a", encoding="utf-8") as f:
            f.write(self.joint_str())


def _parse_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "joint_group_name",
        help='Named joint group from named_joint_states: "ready" or "ftcalib_jgroup"',
    )
    parser.add_argument(
        "path_prefix",
        nargs="?",
        default=None,
        help="Optional workspace folder containing config/ and log/.",
    )
    parser.add_argument(
        "--interface-cfg",
        type=str,
        default=os.environ.get("PEIRASTIC_INTERFACE_CFG", "local-host.yml"),
    )
    parser.add_argument(
        "--joint-controller-cfg",
        type=str,
        default=os.environ.get("PEIRASTIC_NETFT_JOINT_CONTROLLER_CFG", "joint-position-controller.yml"),
    )
    parser.add_argument(
        "--named-states-yaml",
        type=str,
        default=os.environ.get("PEIRASTIC_NETFT_NAMED_JOINT_STATES_YAML", ""),
        help="Optional YAML mapping {name: [7 joint radians]} overriding built-in defaults.",
    )
    parser.add_argument(
        "--seed-joint-cmds-from-package",
        action="store_true",
        help="If config/joint_cmds.txt is missing under path_prefix, copy the bundled default file.",
    )
    parser.add_argument(
        "--continue-without-named-init",
        action="store_true",
        help=(
            "Skip the initial move to the named joint group and go straight to joint_cmds.txt. "
            "For debugging only; breaks calibration if the arm is not already in the expected pose."
        ),
    )
    parser.add_argument(
        "--joint-cmd-move-timeout",
        type=float,
        default=10.0,
        metavar="SEC",
        help="Max seconds per joint_cmds waypoint (default: 10).",
    )
    parser.add_argument(
        "--netft-samples-timeout",
        type=float,
        default=5.0,
        metavar="SEC",
        help="Max seconds to collect /netft_data averages per sample row (default: 5).",
    )
    parser.add_argument(
        "--with-roslaunch",
        action="store_true",
        help="Start netft_utils netft_node via roslaunch before acquisition; tear down on exit.",
    )
    parser.add_argument(
        "--netft-ip",
        type=str,
        default=os.environ.get("PEIRASTIC_NETFT_IP", ""),
        metavar="ADDR",
        help="NetFT box IP for --with-roslaunch (or set PEIRASTIC_NETFT_IP).",
    )
    parser.add_argument(
        "--ros-setup",
        type=str,
        default=os.environ.get("PEIRASTIC_ROS_SETUP", ""),
        metavar="SETUP_BASH",
        help="Path to catkin devel/setup.bash overlay containing netft_utils (or PEIRASTIC_ROS_SETUP).",
    )
    parser.add_argument(
        "--reset-netft-workspace",
        action="store_true",
        help="Delete prior netft_calib_result.yaml and acquisition logs under the workspace before run.",
    )
    parser.add_argument(
        "--no-reset-netft-workspace",
        action="store_true",
        help="With --with-roslaunch: keep existing workspace YAML/logs (default is to reset when using --with-roslaunch).",
    )
    parser.add_argument(
        "--roslaunch-wait",
        type=float,
        default=60.0,
        metavar="SEC",
        help="Seconds to wait for /netft_data after starting roslaunch (default: 60).",
    )
    return parser.parse_args(argv)


def _ensure_joint_cmds(path_prefix: str, seed_from_package: bool) -> None:
    dst = os.path.join(path_prefix, "config", "joint_cmds.txt")
    if os.path.isfile(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    src_text = _read_bundled_netft_joint_cmds_txt()
    with open(dst, "w", encoding="utf-8") as f:
        f.write(src_text)
    if seed_from_package:
        rospy.loginfo("Seeded joint commands from bundled defaults: %s", dst)
    else:
        rospy.logwarn("Missing joint command file, seeded bundled defaults: %s", dst)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = _parse_args(argv)
    args.path_prefix = get_netft_workspace(args.path_prefix)

    should_reset = bool(args.reset_netft_workspace) or (
        bool(args.with_roslaunch) and not bool(args.no_reset_netft_workspace)
    )
    if should_reset:
        reset_netft_workspace_artifacts(args.path_prefix)

    launch_proc = None
    launch_path = None
    robot_interface = None
    exit_code = 0
    try:
        if args.with_roslaunch:
            if not str(args.netft_ip).strip():
                print(
                    "ERROR: --with-roslaunch requires --netft-ip or PEIRASTIC_NETFT_IP.",
                    file=sys.stderr,
                )
                return 2
            if wait_for_netft_topic(timeout_s=2.0):
                print(
                    "[peirastic] /netft_data already present; not starting another netft_node.",
                )
            else:
                try:
                    launch_proc, launch_path = start_netft_roslaunch(
                        str(args.netft_ip).strip(),
                        str(args.ros_setup).strip(),
                        wait_for_topic_s=float(args.roslaunch_wait),
                    )
                except (FileNotFoundError, ValueError, RuntimeError) as e:
                    print(f"ERROR: failed to start netft roslaunch: {e}", file=sys.stderr)
                    return 2

        rospy.init_node("move_to_position", anonymous=True)

        os.makedirs(os.path.join(args.path_prefix, "log"), exist_ok=True)
        os.makedirs(os.path.join(args.path_prefix, "config"), exist_ok=True)
        _ensure_joint_cmds(args.path_prefix, seed_from_package=args.seed_joint_cmds_from_package)

        named_states = named_js.load_named_joint_states(args.named_states_yaml or None)

        robot_interface = FrankaInterface(
            config_root + f"/{args.interface_cfg}",
            use_visualizer=False,
            automatic_gripper_reset=False,
        )
        joint_controller_cfg = YamlConfig(config_root + f"/{args.joint_controller_cfg}").as_easydict()

        deadline = time.time() + 10.0
        while time.time() < deadline and not rospy.is_shutdown():
            if robot_interface.state_buffer_size > 0:
                break
            rospy.sleep(0.1)
        if robot_interface.last_state is None:
            robot_interface.close()
            robot_interface = None
            rospy.logfatal("No robot state received from franka-interface.")
            exit_code = 2
        else:
            try:
                fcda = ForceCaliDataAcq(
                    path_prefix=args.path_prefix,
                    end_effector_link="panda_link8",
                    robot_interface=robot_interface,
                    joint_controller_cfg=joint_controller_cfg,
                    named_states=named_states,
                    joint_cmd_move_timeout_s=args.joint_cmd_move_timeout,
                    netft_samples_timeout_s=args.netft_samples_timeout,
                )
                init_ok = True
                if args.continue_without_named_init:
                    rospy.logwarn(
                        "Skipping initial named pose (--continue-without-named-init). "
                        "joint_cmds will start from the current configuration."
                    )
                else:
                    rospy.loginfo(
                        "Step 1/2: move to named pose %r. The joint_cmds sequence runs only after this succeeds.",
                        args.joint_group_name,
                    )
                    init_ok = fcda.joint_pos_init(args.joint_group_name)
                if init_ok:
                    fcda.run_data_acquisition(args.joint_group_name)
                else:
                    rospy.logerr(
                        "Stopped before joint_cmds: the arm never reached the named pose, so no later waypoints run. "
                        "Fix franka-interface, robot unlock, and ROBOT.IP; or use --continue-without-named-init only to debug."
                    )
            except rospy.ROSInterruptException:
                rospy.loginfo("ROS Interrupt, exiting")
            except KeyboardInterrupt:
                rospy.loginfo("KeyboardInterrupt, exiting")
                rospy.signal_shutdown("KeyboardInterrupt")
    finally:
        try:
            if rospy.core.is_initialized():
                rospy.loginfo("Program done!")
        except Exception:
            pass
        if robot_interface is not None:
            try:
                robot_interface.close()
            except Exception:
                pass
        stop_netft_roslaunch(launch_proc, launch_path)
        try:
            if rospy.core.is_initialized():
                rospy.signal_shutdown("Finished")
        except Exception:
            pass
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
