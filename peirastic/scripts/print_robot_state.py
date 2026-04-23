"""Moving robot joint positions to initial pose for starting new experiments."""
import argparse
import pickle
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from peirastic import config_root
from peirastic.franka_interface import FrankaInterface
from peirastic.utils import YamlConfig
from peirastic.utils.input_utils import input2action
from peirastic.utils.io_devices import SpaceMouse
from peirastic.utils.log_utils import get_peirastic_example_logger

import pprint

logger = get_peirastic_example_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="local-host.yml")
    parser.add_argument(
        "--controller-cfg", type=str, default="joint-position-controller.yml"
    )
    parser.add_argument(
        "--folder", type=Path, default="data_collection_example/example_data"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    robot_interface = FrankaInterface(
        config_root + f"/{args.interface_cfg}", use_visualizer=False
    )
    controller_cfg = YamlConfig(config_root + f"/{args.controller_cfg}").as_easydict()

    controller_type = "JOINT_POSITION"

    pp = pprint.PrettyPrinter(indent=2)
    while True:
        if len(robot_interface._state_buffer) > 0:
            logger.info(f"Current Robot joint: {np.round(robot_interface.last_q, 3)}")
            last_eef_rot, last_eef_pos = robot_interface.last_eef_rot_and_pos
            logger.info(f"Current eef rotation: ")
            pp.pprint([list(i) for i in last_eef_rot])

            logger.info(f"Current eef position: ")
            pp.pprint([list(i)[0] for i in last_eef_pos])
            
            break
    robot_interface.close()


if __name__ == "__main__":
    main()
