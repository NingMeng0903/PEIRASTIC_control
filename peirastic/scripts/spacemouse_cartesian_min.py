#!/usr/bin/env python3
"""Minimal SpaceMouse -> CARTESIAN_VELOCITY pose tracking teleop."""

import argparse
import sys
import time

import numpy as np

from peirastic import config_root
from peirastic.franka_interface import FrankaInterface
from peirastic.utils.config_utils import get_default_controller_config, verify_controller_config
from peirastic.utils import transform_utils
from peirastic.utils.io_devices import SpaceMouse
from peirastic.utils.yaml_config import YamlConfig

INIT_JOINT_ANGLES = [0.0, 0.0, 0.0, -1.8, 0.0, 1.8, 0.0]


def _print_spacemouse_open_failed(err: OSError) -> None:
    print(
        "SpaceMouse HID open failed (hid.device.open). On Linux this is usually permissions or an "
        "exclusive lock.\n"
        "  - Unplug/replug the USB receiver; close 3DxWare, Blender, CAD, or other teleop scripts.\n"
        "  - Quick test: sudo chmod a+rw /dev/hidraw*  (then fix with a udev rule for vendor 0x256f)\n"
        "  - Add your user to the group used in the udev rule (often plugdev), then log out/in.\n"
        f"Original error: {err!r}",
        file=sys.stderr,
    )


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


def _get_spacemouse_motion(device: SpaceMouse):
    state = device.get_controller_state()
    dpos = np.asarray(state["dpos"], dtype=np.float64)
    drot = np.asarray(state["raw_drotation"], dtype=np.float64)[[1, 0, 2]]
    drot[2] *= -1.0
    grasp = 1.0 if state["grasp"] else -1.0
    return dpos, drot, bool(state["reset"]), grasp


def _capture_spacemouse_bias(device: SpaceMouse, sample_count: int, sleep_s: float):
    pos_samples = []
    rot_samples = []
    for _ in range(max(1, sample_count)):
        dpos, drot, _, _ = _get_spacemouse_motion(device)
        pos_samples.append(dpos)
        rot_samples.append(drot)
        time.sleep(sleep_s)

    return (
        np.mean(np.asarray(pos_samples), axis=0),
        np.mean(np.asarray(rot_samples), axis=0),
    )


def _build_absolute_action(target_pos, target_quat, grasp):
    canonical_quat = _canonicalize_quaternion(target_quat)
    return np.concatenate(
        [np.asarray(target_pos, dtype=np.float64), canonical_quat, [float(grasp)]]
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interface-cfg", default="local-host.yml")
    p.add_argument(
        "--controller-cfg",
        default="",
        help="YAML under config/; default: built-in cartesian-velocity-controller.yml.",
    )
    p.add_argument(
        "--control-freq",
        type=float,
        default=200.0,
        help="Command send frequency enforced by FrankaInterface.",
    )
    p.add_argument(
        "--period",
        type=float,
        default=0.0,
        help="Optional extra sleep after each loop; 0 means rely on control-freq only.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Max control sends with non-None action; 0 = unlimited (Ctrl+C to exit).",
    )
    p.add_argument("--vendor-id", type=int, default=9583)
    p.add_argument("--product-id", type=int, default=None)
    p.add_argument(
        "--terminate-at-exit",
        action="store_true",
        help="Send termination=True on exit (otherwise franka-interface keeps running).",
    )
    p.add_argument(
        "--init-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for INIT_JOINT_ANGLES via JOINT_POSITION before teleop.",
    )
    p.add_argument(
        "--robot-first",
        action="store_true",
        help="Connect to franka-interface and move to INIT_JOINT_ANGLES before opening SpaceMouse HID.",
    )
    p.add_argument("--linear-scale", type=float, default=20.0)
    p.add_argument("--angular-scale", type=float, default=8.0)
    p.add_argument("--linear-deadzone", type=float, default=5e-4)
    p.add_argument("--angular-deadzone", type=float, default=2.5e-3)
    p.add_argument("--bias-samples", type=int, default=40)
    args = p.parse_args()

    cfg = args.interface_cfg if args.interface_cfg.startswith("/") else f"{config_root}/{args.interface_cfg}"
    if args.controller_cfg:
        path = (
            args.controller_cfg
            if args.controller_cfg.startswith("/")
            else f"{config_root}/{args.controller_cfg}"
        )
        controller_cfg = verify_controller_config(YamlConfig(path).as_easydict())
    else:
        controller_cfg = get_default_controller_config("CARTESIAN_VELOCITY")
    controller_cfg["is_delta"] = False

    ri = FrankaInterface(
        cfg,
        control_freq=args.control_freq,
        use_visualizer=False,
        automatic_gripper_reset=False,
    )
    sm = None
    try:
        if args.robot_first:
            if not ri.wait_for_state(timeout=30.0):
                print("Timeout: no robot state from franka-interface.", file=sys.stderr)
                return 2
            if not ri.move_joints(INIT_JOINT_ANGLES, timeout=args.init_timeout):
                print("Failed to reach INIT_JOINT_ANGLES.", file=sys.stderr)
                return 3
            try:
                sm = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id)
            except OSError as e:
                _print_spacemouse_open_failed(e)
                return 4
            sm.start_control()
        else:
            try:
                sm = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id)
            except OSError as e:
                _print_spacemouse_open_failed(e)
                return 4
            sm.start_control()
            if not ri.wait_for_state(timeout=30.0):
                print("Timeout: no robot state from franka-interface.", file=sys.stderr)
                return 2
            if not ri.move_joints(INIT_JOINT_ANGLES, timeout=args.init_timeout):
                print("Failed to reach INIT_JOINT_ANGLES.", file=sys.stderr)
                return 3

        ctype = "CARTESIAN_VELOCITY"
        n = 0
        bias_pos, bias_rot = _capture_spacemouse_bias(sm, args.bias_samples, 0.01)
        current_pose = ri.last_eef_pose
        if current_pose is None:
            print("Robot pose is unavailable after joint reset.", file=sys.stderr)
            return 5

        target_pos = current_pose[:3, 3].astype(np.float64).copy()
        target_quat = _canonicalize_quaternion(
            transform_utils.mat2quat(current_pose[:3, :3]).astype(np.float64).copy()
        )
        dt = 1.0 / max(args.control_freq, 1.0)
        sleep_interval = args.period if args.period > 0.0 else 0.001
        try:
            while args.max_steps == 0 or n < args.max_steps:
                dpos, drot, do_reset, grasp = _get_spacemouse_motion(sm)
                dpos -= bias_pos
                drot -= bias_rot

                if do_reset:
                    if not ri.move_joints(INIT_JOINT_ANGLES, timeout=args.init_timeout):
                        print("Failed to reach INIT_JOINT_ANGLES.", file=sys.stderr)
                        return 6
                    current_pose = ri.last_eef_pose
                    if current_pose is not None:
                        target_pos = current_pose[:3, 3].astype(np.float64).copy()
                        target_quat = _canonicalize_quaternion(
                            transform_utils.mat2quat(current_pose[:3, :3]).astype(np.float64).copy()
                        )
                    bias_pos, bias_rot = _capture_spacemouse_bias(sm, args.bias_samples, 0.01)
                    time.sleep(sleep_interval)
                    continue

                if np.linalg.norm(dpos) < args.linear_deadzone:
                    dpos[:] = 0.0
                if np.linalg.norm(drot) < args.angular_deadzone:
                    drot[:] = 0.0

                if np.linalg.norm(dpos) <= 1e-12 and np.linalg.norm(drot) <= 1e-12:
                    time.sleep(sleep_interval)
                    continue

                target_pos += dpos * args.linear_scale * dt
                delta_axis_angle = drot * args.angular_scale * dt
                if np.linalg.norm(delta_axis_angle) > 0.0:
                    previous_target_quat = target_quat.copy()
                    delta_quat = transform_utils.axisangle2quat(delta_axis_angle)
                    target_quat = transform_utils.quat_multiply(
                        delta_quat, target_quat
                    ).astype(np.float64)
                    target_quat = _canonicalize_quaternion(
                        target_quat, reference_quat=previous_target_quat
                    )

                action = _build_absolute_action(target_pos, target_quat, grasp)
                ri.control(ctype, action, controller_cfg=controller_cfg)
                n += 1
                if args.period > 0.0:
                    time.sleep(args.period)
        except KeyboardInterrupt:
            pass
        finally:
            if args.terminate_at_exit:
                ri.control(
                    ctype,
                    np.concatenate([np.zeros(6), [1.0]]),
                    controller_cfg=controller_cfg,
                    termination=True,
                )
    finally:
        ri.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
