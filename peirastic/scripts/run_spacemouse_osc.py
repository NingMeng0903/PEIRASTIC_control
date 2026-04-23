"""Run SpaceMouse teleoperation through the peirastic interface."""

import argparse
import time

import numpy as np

from peirastic import config_root
from peirastic.franka_interface import FrankaInterface
from peirastic.utils.config_utils import get_default_controller_config
from peirastic.utils.input_utils import input2action
from peirastic.utils.io_devices import SpaceMouse
from peirastic.utils.log_utils import get_peirastic_example_logger

logger = get_peirastic_example_logger()


def main():
    parser = argparse.ArgumentParser(
        description="SpaceMouse USB teleoperation for PEIRASTIC_control."
    )
    parser.add_argument(
        "--interface-cfg",
        type=str,
        default="local-host.yml",
        help="YAML file under the config directory.",
    )
    parser.add_argument(
        "--controller-type",
        type=str,
        default="OSC_POSE",
        choices=[
            "OSC_POSE",
            "OSC_POSITION",
            "OSC_YAW",
            "CARTESIAN_VELOCITY",
        ],
    )
    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument(
        "--product-id",
        type=int,
        default=None,
        help="Optional HID product id. Defaults to auto-detecting the connected SpaceMouse.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3000,
        help="Number of control steps before optional termination.",
    )
    parser.add_argument(
        "--terminate-at-exit",
        action="store_true",
        help="Send termination=True on exit (default: keep franka-interface controller running).",
    )
    args = parser.parse_args()

    cfg_path = args.interface_cfg
    if not cfg_path.startswith("/"):
        cfg_path = f"{config_root}/{cfg_path}"

    device = SpaceMouse(vendor_id=args.vendor_id, product_id=args.product_id)
    device.start_control()

    robot_interface = FrankaInterface(cfg_path, use_visualizer=False)

    controller_type = args.controller_type
    controller_cfg = get_default_controller_config(controller_type=controller_type)
    if controller_type == "CARTESIAN_VELOCITY":
        controller_cfg["is_delta"] = True

    robot_interface.clear_state_buffers()

    for _ in range(args.steps):
        start_time = time.time_ns()

        action, _grasp = input2action(
            device=device,
            controller_type=controller_type,
        )
        if controller_cfg.get("is_delta", False):
            action = np.asarray(action, dtype=float)
            action[:3] *= float(controller_cfg["action_scale"]["translation"])
            action[3:6] *= float(controller_cfg["action_scale"]["rotation"])

        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )
        end_time = time.time_ns()
        logger.debug("cycle: %.4fs", (end_time - start_time) / 1e9)

    if args.terminate_at_exit:
        robot_interface.control(
            controller_type=controller_type,
            action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [1.0],
            controller_cfg=controller_cfg,
            termination=True,
        )

    state_history = robot_interface.get_state_buffer_snapshot()
    robot_interface.close()

    for state, next_state in zip(state_history[:-1], state_history[1:]):
        if (next_state.frame - state.frame) > 1:
            print(state.frame, next_state.frame)


if __name__ == "__main__":
    main()
