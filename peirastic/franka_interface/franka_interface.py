import copy
import logging
import threading
import time
from collections import deque
from typing import Callable, Mapping, Optional, Tuple, Union

import numpy as np
import zmq

import peirastic.proto.franka_interface.franka_controller_pb2 as franka_controller_pb2
import peirastic.proto.franka_interface.franka_robot_state_pb2 as franka_robot_state_pb2
from peirastic.utils import transform_utils
from peirastic.utils.config_utils import get_default_controller_config, verify_controller_config
from peirastic.utils.yaml_config import YamlConfig

logger = logging.getLogger(__name__)

DEFAULT_STATE_BUFFER_SIZE = 2048
DEFAULT_GRIPPER_STATE_BUFFER_SIZE = 2048


def action_to_osc_pose_goal(action, is_delta=True) -> franka_controller_pb2.Goal:
    goal = franka_controller_pb2.Goal()
    goal.is_delta = is_delta
    goal.x = action[0]
    goal.y = action[1]
    goal.z = action[2]
    goal.ax = action[3]
    goal.ay = action[4]
    goal.az = action[5]
    return goal


def action_to_cartesian_velocity(action, is_delta=True) -> franka_controller_pb2.Goal:
    goal = franka_controller_pb2.Goal()
    goal.is_delta = is_delta
    goal.x = action[0]
    goal.y = action[1]
    goal.z = action[2]
    if len(action) == 6:
        goal.ax = action[3]
        goal.ay = action[4]
        goal.az = action[5]
    return goal

def action_to_joint_pos_goal(action, is_delta=False) -> franka_controller_pb2.JointGoal:
    goal = franka_controller_pb2.JointGoal()
    goal.is_delta = is_delta
    goal.q1 = action[0]
    goal.q2 = action[1]
    goal.q3 = action[2]
    goal.q4 = action[3]
    goal.q5 = action[4]
    goal.q6 = action[5]
    goal.q7 = action[6]
    return goal


TRAJ_INTERPOLATOR_MAPPING = {
    "SMOOTH_JOINT_POSITION": franka_controller_pb2.FrankaControlMessage.TrajInterpolatorType.SMOOTH_JOINT_POSITION,
    "LINEAR_POSE": franka_controller_pb2.FrankaControlMessage.TrajInterpolatorType.LINEAR_POSE,
    "LINEAR_POSITION": franka_controller_pb2.FrankaControlMessage.TrajInterpolatorType.LINEAR_POSITION,
    "LINEAR_JOINT_POSITION": franka_controller_pb2.FrankaControlMessage.TrajInterpolatorType.LINEAR_JOINT_POSITION,
    "MIN_JERK_POSE": franka_controller_pb2.FrankaControlMessage.TrajInterpolatorType.MIN_JERK_POSE,
    "MIN_JERK_JOINT_POSITION": franka_controller_pb2.FrankaControlMessage.TrajInterpolatorType.MIN_JERK_JOINT_POSITION,
    "COSINE_CARTESIAN_VELOCITY": franka_controller_pb2.FrankaControlMessage.TrajInterpolatorType.COSINE_CARTESIAN_VELOCITY,
    "LINEAR_CARTESIAN_VELOCITY": franka_controller_pb2.FrankaControlMessage.TrajInterpolatorType.LINEAR_CARTESIAN_VELOCITY,
    "RUCKIG_POSE": franka_controller_pb2.FrankaControlMessage.TrajInterpolatorType.RUCKIG_POSE,
}


class FrankaInterface:
    """
    This is the Python Interface for communicating with franka interface on NUC.
    Args:
        general_cfg_file (str, optional): _description_. Defaults to "config/local-host.yml".
        control_freq (float, optional): _description_. Defaults to 20.0.
        state_freq (float, optional): _description_. Defaults to 100.0.
        control_timeout (float, optional): _description_. Defaults to 1.0.
        has_gripper (bool, optional): _description_. Defaults to True.
        use_visualizer (bool, optional): _description_. Defaults to False.
    """

    def __init__(
        self,
        general_cfg_file: str = "config/local-host.yml",
        control_freq: float = 20.0,
        state_freq: float = 100.0,
        control_timeout: float = 1.0,
        has_gripper: bool = True,
        use_visualizer: bool = False,
        automatic_gripper_reset: bool = True,
        state_buffer_size: int = DEFAULT_STATE_BUFFER_SIZE,
        gripper_state_buffer_size: int = DEFAULT_GRIPPER_STATE_BUFFER_SIZE,
    ):
        general_cfg = YamlConfig(general_cfg_file).as_easydict()
        self._name = general_cfg.PC.NAME
        self._ip = general_cfg.NUC.IP
        self._pub_port = general_cfg.NUC.SUB_PORT
        self._sub_port = general_cfg.NUC.PUB_PORT

        self._gripper_pub_port = general_cfg.NUC.GRIPPER_SUB_PORT
        self._gripper_sub_port = general_cfg.NUC.GRIPPER_PUB_PORT

        self._context = zmq.Context()
        self._publisher = self._context.socket(zmq.PUB)
        self._subscriber = self._context.socket(zmq.SUB)

        self._gripper_publisher = self._context.socket(zmq.PUB)
        self._gripper_subscriber = self._context.socket(zmq.SUB)
        for socket in (
            self._publisher,
            self._subscriber,
            self._gripper_publisher,
            self._gripper_subscriber,
        ):
            socket.setsockopt(zmq.LINGER, 0)
        self._subscriber.setsockopt(zmq.RCVTIMEO, 100)
        self._gripper_subscriber.setsockopt(zmq.RCVTIMEO, 100)

        # publisher
        self._publisher.bind(f"tcp://*:{self._pub_port}")
        self._gripper_publisher.bind(f"tcp://*:{self._gripper_pub_port}")

        # subscriber
        self._subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        self._subscriber.connect(f"tcp://{self._ip}:{self._sub_port}")

        self._gripper_subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        self._gripper_subscriber.connect(f"tcp://{self._ip}:{self._gripper_sub_port}")

        self._state_lock = threading.Lock()
        self._gripper_state_lock = threading.Lock()
        self._state_buffer = deque(maxlen=max(1, int(state_buffer_size)))
        self._state_buffer_idx = 0

        self._gripper_state_buffer = deque(
            maxlen=max(1, int(gripper_state_buffer_size))
        )
        self._gripper_buffer_idx = 0

        # control frequency
        self._control_freq = control_freq
        self._control_interval = 1.0 / self._control_freq

        # state frequency
        self._state_freq = state_freq

        # control timeout (s)
        self._control_timeout = control_timeout

        self.counter = 0
        self.termination = False
        self._stop_event = threading.Event()
        self._closed = False

        self._state_sub_thread = threading.Thread(target=self.get_state)
        self._state_sub_thread.daemon = True
        self._state_sub_thread.start()

        self._gripper_sub_thread = threading.Thread(target=self.get_gripper_state)
        self._gripper_sub_thread.daemon = True
        self._gripper_sub_thread.start()

        self.last_time = None

        # PC -> NUC control session (protobuf uint64). 0 = legacy (no session gating on NUC).
        self._control_session = 0
        self._session_hard_reset_once = False

        self.has_gripper = has_gripper

        self.use_visualizer = use_visualizer
        self.visualizer = None
        if self.use_visualizer:
            from peirastic.franka_interface.visualizer import visualizer_factory

            self.visualizer = visualizer_factory(backend="pybullet")

        self._last_controller_type = "Dummy"

        self.last_gripper_dim = 1 if has_gripper else 0
        self.last_gripper_action = 0

        self.last_gripper_command_counter = 0
        self._history_actions = []

        # automatically reset gripper by default
        self.automatic_gripper_reset = automatic_gripper_reset

    @staticmethod
    def _sleep_interruptible(
        duration: float,
        shutdown_check: Optional[Callable[[], bool]] = None,
        step: float = 0.02,
    ) -> bool:
        """Sleep up to duration in short steps; return False if shutdown_check is true."""
        if duration <= 0.0:
            return True
        end = time.time() + duration
        while time.time() < end:
            if shutdown_check is not None and shutdown_check():
                return False
            time.sleep(min(step, end - time.time()))
        return True

    def get_state(self, no_block: bool = False):
        """_summary_

        Args:
            no_block (bool, optional): Decide if zmq receives messages synchronously or asynchronously. Defaults to False.
        """
        if no_block:
            recv_kwargs = {"flags": zmq.NOBLOCK}
        else:
            recv_kwargs = {}
        while not self._stop_event.is_set():
            try:
                franka_robot_state = franka_robot_state_pb2.FrankaRobotStateMessage()
                message = self._subscriber.recv(**recv_kwargs)
                franka_robot_state.ParseFromString(message)
                with self._state_lock:
                    self._state_buffer.append(franka_robot_state)
            except zmq.Again:
                continue
            except zmq.ZMQError:
                if self._stop_event.is_set():
                    break
                logger.exception("State subscriber failed.")
                time.sleep(0.05)
            except Exception:
                logger.exception("Failed to parse robot state message.")
                time.sleep(0.05)

    def get_gripper_state(self):
        while not self._stop_event.is_set():
            try:
                franka_gripper_state = (
                    franka_robot_state_pb2.FrankaGripperStateMessage()
                )
                message = self._gripper_subscriber.recv()
                franka_gripper_state.ParseFromString(message)
                with self._gripper_state_lock:
                    self._gripper_state_buffer.append(franka_gripper_state)
            except zmq.Again:
                continue
            except zmq.ZMQError:
                if self._stop_event.is_set():
                    break
                logger.exception("Gripper state subscriber failed.")
                time.sleep(0.05)
            except Exception:
                logger.exception("Failed to parse gripper state message.")
                time.sleep(0.05)

    def clear_state_buffers(self) -> None:
        with self._state_lock:
            self._state_buffer.clear()
        with self._gripper_state_lock:
            self._gripper_state_buffer.clear()
        self._state_buffer_idx = 0
        self._gripper_buffer_idx = 0

    def _split_action(
        self,
        action: Union[np.ndarray, list],
        *,
        valid_control_dims: Tuple[int, ...],
        controller_type: str,
    ) -> Tuple[np.ndarray, Optional[float]]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        gripper_dims = 1 if self.has_gripper else 0
        self.last_gripper_dim = gripper_dims
        for control_dims in valid_control_dims:
            if action.size == control_dims + gripper_dims:
                if gripper_dims == 0:
                    return action.copy(), None
                return action[:-gripper_dims].copy(), float(action[-1])

        valid_totals = [control_dims + gripper_dims for control_dims in valid_control_dims]
        raise ValueError(
            f"{controller_type} action has {action.size} dims; expected one of {valid_totals}."
        )

    def preprocess(self, shutdown_check: Optional[Callable[[], bool]] = None) -> bool:

        if self.automatic_gripper_reset and self.has_gripper:
            gripper_control_msg = franka_controller_pb2.FrankaGripperControlMessage()
            move_msg = franka_controller_pb2.FrankaGripperMoveMessage()
            move_msg.width = 0.08
            move_msg.speed = 0.1
            gripper_control_msg.control_msg.Pack(move_msg)

            logger.debug("Moving Command")
            self._gripper_publisher.send(gripper_control_msg.SerializeToString())

        # Send only 2 NO_CONTROL (was 20). Long NO_CONTROL makes C++ set running=false
        # so no torques are sent and the robot can drop; 2 msgs ~0.1s minimizes drop.
        for _ in range(2):
            dummy_msg = franka_controller_pb2.FrankaDummyControllerMessage()
            control_msg = franka_controller_pb2.FrankaControlMessage()
            control_msg.controller_type = (
                franka_controller_pb2.FrankaControlMessage.ControllerType.NO_CONTROL
            )

            control_msg.control_msg.Pack(dummy_msg)
            control_msg.timeout = 0.2
            control_msg.termination = False

            self._send_control_msg(control_msg)
            if not self._sleep_interruptible(0.05, shutdown_check):
                return False

        logger.debug("Preprocess fnished")
        # Resync policy-rate throttling. Switches are bursty; without this, the next
        # control() right after NO_CONTROL can be skipped by the remaining_time gate
        # in control(), and franka-interface may not apply the new controller/goal in time.
        self.last_time = None
        return True

    def force_next_control_preprocess(self) -> None:
        """After long gaps without ZMQ control, next control() runs preprocess (Dummy -> active)."""
        self._last_controller_type = "Dummy"

    def reset_control_rate_limiter(self) -> None:
        """Reset policy frequency throttling state for the next control() call."""
        self.last_time = None

    def set_control_session(self, session: int) -> None:
        self._control_session = int(session) & ((1 << 64) - 1)

    def get_control_session(self) -> int:
        return int(self._control_session)

    def request_session_hard_reset(self) -> None:
        self._session_hard_reset_once = True

    def bump_control_session(self) -> None:
        if self._control_session == 0:
            self._control_session = 1
        else:
            self._control_session = (int(self._control_session) + 1) & ((1 << 64) - 1)

    @staticmethod
    def _canonicalize_quaternion(quat: np.ndarray, reference_quat: Optional[np.ndarray] = None) -> np.ndarray:
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

    @staticmethod
    def _set_traj_interpolator_config(control_msg, controller_cfg):
        traj_cfg = controller_cfg.traj_interpolator_cfg
        control_msg.traj_interpolator_type = TRAJ_INTERPOLATOR_MAPPING[
            traj_cfg.traj_interpolator_type
        ]
        control_msg.traj_interpolator_time_fraction = traj_cfg["time_fraction"]

        if hasattr(controller_cfg, "traj_interpolator_config"):
            interpolator_cfg = controller_cfg.traj_interpolator_config
            control_msg.traj_interpolator_config.max_velocity = (
                interpolator_cfg.max_velocity
            )
            control_msg.traj_interpolator_config.max_acceleration = (
                interpolator_cfg.max_acceleration
            )
            control_msg.traj_interpolator_config.max_jerk = (
                interpolator_cfg.max_jerk
            )

    @staticmethod
    def _build_state_estimator_msg(controller_cfg):
        state_estimator_msg = franka_controller_pb2.FrankaStateEstimatorMessage()
        state_estimator_msg.is_estimation = (
            controller_cfg.state_estimator_cfg.is_estimation
        )
        state_estimator_msg.estimator_type = (
            franka_controller_pb2.FrankaStateEstimatorMessage.EstimatorType.EXPONENTIAL_SMOOTHING_ESTIMATOR
        )
        exponential_estimator = franka_controller_pb2.ExponentialSmoothingConfig()
        exponential_estimator.alpha_q = controller_cfg.state_estimator_cfg.alpha_q
        exponential_estimator.alpha_dq = controller_cfg.state_estimator_cfg.alpha_dq
        exponential_estimator.alpha_eef = controller_cfg.state_estimator_cfg.alpha_eef
        exponential_estimator.alpha_eef_vel = (
            controller_cfg.state_estimator_cfg.alpha_eef_vel
        )
        state_estimator_msg.config.Pack(exponential_estimator)
        return state_estimator_msg

    def _send_control_msg(self, control_msg):
        self._require_open()
        control_msg.control_session = int(self._control_session)
        control_msg.session_hard_reset = bool(self._session_hard_reset_once)
        self._session_hard_reset_once = False
        self._publisher.send(control_msg.SerializeToString())

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("FrankaInterface is closed.")

    def _scale_cartesian_action(self, action, controller_cfg):
        """Legacy helper for scripts that explicitly want Python-side scaling."""
        scaled_action = np.array(action, copy=True)
        scaled_action[0:3] *= controller_cfg.action_scale.translation
        scaled_action[3:6] *= controller_cfg.action_scale.rotation
        return scaled_action

    @staticmethod
    def _uses_pose_tracking(controller_cfg) -> bool:
        return (
            controller_cfg.traj_interpolator_cfg.traj_interpolator_type
            == "RUCKIG_POSE"
        )

    def control(
        self,
        controller_type: str,
        action: Union[np.ndarray, list],
        controller_cfg: dict = None,
        termination: bool = False,
        shutdown_check: Optional[Callable[[], bool]] = None,
    ):
        """A function that controls every step on the policy level.

        Args:
            controller_type (str): The type of controller used in this step.
            action (Union[np.ndarray, list]): The action command for the controller.
            controller_cfg (dict, optional): Controller configuration that corresponds to the first argument`controller_type`. Defaults to None.
            termination (bool, optional): If set True, the control will be terminated. Defaults to False.
            shutdown_check (optional): When set, sleeps split so this can abort early (e.g. rospy.is_shutdown).
        """
        self._require_open()
        if self.last_time == None:
            self.last_time = time.time_ns()
        elif not termination:
            # Control the policy frequency if not terminated.
            current_time = time.time_ns()
            remaining_time = self._control_interval - (
                current_time - self.last_time
            ) / (10**9)
            if 0.0001 < remaining_time < self._control_timeout:
                if not self._sleep_interruptible(remaining_time, shutdown_check):
                    self.last_time = time.time_ns()
                    return
            self.last_time = time.time_ns()

        if self._last_controller_type != controller_type:
            if not self.preprocess(shutdown_check):
                return
            self._last_controller_type = controller_type

        controller_cfg = verify_controller_config(controller_cfg, use_default=True)

        state_estimator_msg = self._build_state_estimator_msg(controller_cfg)

        if controller_type == "OSC_POSE":
            assert controller_cfg is not None
            control_action, gripper_action = self._split_action(
                action, valid_control_dims=(6,), controller_type=controller_type
            )

            osc_msg = franka_controller_pb2.FrankaOSCPoseControllerMessage()
            osc_msg.translational_stiffness[:] = controller_cfg.Kp.translation
            osc_msg.rotational_stiffness[:] = controller_cfg.Kp.rotation

            osc_config = franka_controller_pb2.FrankaOSCControllerConfig()

            osc_config.residual_mass_vec[:] = controller_cfg.residual_mass_vec
            osc_msg.config.CopyFrom(osc_config)

            logger.debug(f"OSC action: {np.round(control_action, 3)}")

            self._history_actions.append(control_action.copy())
            goal = action_to_osc_pose_goal(
                control_action, is_delta=controller_cfg.is_delta
            )
            osc_msg.goal.CopyFrom(goal)

            control_msg = franka_controller_pb2.FrankaControlMessage()
            control_msg.controller_type = (
                franka_controller_pb2.FrankaControlMessage.ControllerType.OSC_POSE
            )
            self._set_traj_interpolator_config(control_msg, controller_cfg)
            control_msg.control_msg.Pack(osc_msg)
            control_msg.timeout = 0.2
            control_msg.termination = termination

            control_msg.state_estimator_msg.CopyFrom(state_estimator_msg)
            self._send_control_msg(control_msg)

        elif controller_type == "OSC_POSITION":

            assert controller_cfg is not None
            control_action, gripper_action = self._split_action(
                action, valid_control_dims=(6,), controller_type=controller_type
            )

            osc_msg = franka_controller_pb2.FrankaOSCPoseControllerMessage()
            osc_msg.translational_stiffness[:] = controller_cfg.Kp.translation
            osc_msg.rotational_stiffness[:] = controller_cfg.Kp.rotation

            osc_config = franka_controller_pb2.FrankaOSCControllerConfig()
            osc_config.residual_mass_vec[:] = controller_cfg.residual_mass_vec
            osc_msg.config.CopyFrom(osc_config)

            self._history_actions.append(control_action.copy())

            goal = action_to_osc_pose_goal(
                control_action, is_delta=controller_cfg.is_delta
            )
            osc_msg.goal.CopyFrom(goal)
            control_msg = franka_controller_pb2.FrankaControlMessage()
            control_msg.controller_type = (
                franka_controller_pb2.FrankaControlMessage.ControllerType.OSC_POSITION
            )
            self._set_traj_interpolator_config(control_msg, controller_cfg)
            control_msg.control_msg.Pack(osc_msg)
            control_msg.timeout = 0.2
            control_msg.termination = termination

            control_msg.state_estimator_msg.CopyFrom(state_estimator_msg)
            self._send_control_msg(control_msg)

        elif controller_type == "OSC_YAW":
            assert controller_cfg is not None
            control_action, gripper_action = self._split_action(
                action, valid_control_dims=(6,), controller_type=controller_type
            )

            osc_msg = franka_controller_pb2.FrankaOSCPoseControllerMessage()
            osc_msg.translational_stiffness[:] = controller_cfg.Kp.translation
            osc_msg.rotational_stiffness[:] = controller_cfg.Kp.rotation

            osc_config = franka_controller_pb2.FrankaOSCControllerConfig()
            osc_config.residual_mass_vec[:] = controller_cfg.residual_mass_vec
            osc_msg.config.CopyFrom(osc_config)

            self._history_actions.append(control_action.copy())

            goal = action_to_osc_pose_goal(
                control_action, is_delta=controller_cfg.is_delta
            )
            osc_msg.goal.CopyFrom(goal)
            control_msg = franka_controller_pb2.FrankaControlMessage()
            control_msg.controller_type = (
                franka_controller_pb2.FrankaControlMessage.ControllerType.OSC_YAW
            )
            self._set_traj_interpolator_config(control_msg, controller_cfg)
            control_msg.control_msg.Pack(osc_msg)
            control_msg.timeout = 0.2
            control_msg.termination = termination

            control_msg.state_estimator_msg.CopyFrom(state_estimator_msg)
            self._send_control_msg(control_msg)

        elif controller_type == "JOINT_POSITION":

            assert controller_cfg is not None
            control_action, gripper_action = self._split_action(
                action, valid_control_dims=(7,), controller_type=controller_type
            )

            joint_pos_msg = franka_controller_pb2.FrankaJointPositionControllerMessage()
            joint_pos_msg.speed_factor = 0.1
            goal = action_to_joint_pos_goal(
                control_action, is_delta=controller_cfg.is_delta
            )

            joint_pos_msg.goal.CopyFrom(goal)

            control_msg = franka_controller_pb2.FrankaControlMessage()
            control_msg.controller_type = (
                franka_controller_pb2.FrankaControlMessage.ControllerType.JOINT_POSITION
            )
            self._set_traj_interpolator_config(control_msg, controller_cfg)
            control_msg.control_msg.Pack(joint_pos_msg)
            control_msg.timeout = 0.2
            control_msg.termination = termination

            control_msg.state_estimator_msg.CopyFrom(state_estimator_msg)
            self._send_control_msg(control_msg)

        elif controller_type == "JOINT_IMPEDANCE":

            assert controller_cfg is not None
            control_action, gripper_action = self._split_action(
                action, valid_control_dims=(7,), controller_type=controller_type
            )

            joint_impedance_msg = (
                franka_controller_pb2.FrankaJointImpedanceControllerMessage()
            )
            goal = action_to_joint_pos_goal(
                control_action, is_delta=controller_cfg.is_delta
            )

            joint_impedance_msg.goal.CopyFrom(goal)

            joint_impedance_msg.kp[:] = controller_cfg.joint_kp
            joint_impedance_msg.kd[:] = controller_cfg.joint_kd

            control_msg = franka_controller_pb2.FrankaControlMessage()
            control_msg.controller_type = (
                franka_controller_pb2.FrankaControlMessage.ControllerType.JOINT_IMPEDANCE
            )
            self._set_traj_interpolator_config(control_msg, controller_cfg)
            control_msg.control_msg.Pack(joint_impedance_msg)
            control_msg.timeout = 0.2
            control_msg.termination = termination

            control_msg.state_estimator_msg.CopyFrom(state_estimator_msg)
            self._send_control_msg(control_msg)

        elif controller_type == "CARTESIAN_VELOCITY":
            assert controller_cfg is not None
            goal_dims, gripper_action = self._split_action(
                action, valid_control_dims=(6, 7), controller_type=controller_type
            )

            cartesian_velocity_msg = franka_controller_pb2.FrankaCartesianVelocityControllerMessage()
            logger.debug(f"Cartesian velocity action: {np.round(goal_dims, 3)}")

            self._history_actions.append(goal_dims.copy())
            goal = action_to_cartesian_velocity(goal_dims, is_delta=controller_cfg.is_delta)
            cartesian_velocity_msg.goal.CopyFrom(goal)
            if (
                goal_dims.shape[0] == 7
                and not controller_cfg.is_delta
                and self._uses_pose_tracking(controller_cfg)
            ):
                cartesian_velocity_msg.absolute_quaternion[:] = goal_dims[3:7]
            cartesian_velocity_msg.speed_factor = 1.0
            if hasattr(controller_cfg, "Kp") and hasattr(controller_cfg.Kp, "translation"):
                cartesian_velocity_msg.translation_kp[:] = controller_cfg.Kp.translation
            if hasattr(controller_cfg, "Kp") and hasattr(controller_cfg.Kp, "rotation"):
                cartesian_velocity_msg.rotation_kp[:] = controller_cfg.Kp.rotation
            if hasattr(controller_cfg, "Kd"):
                cartesian_velocity_msg.kd_gains = float(controller_cfg.Kd)

            control_msg = franka_controller_pb2.FrankaControlMessage()
            control_msg.controller_type = (
                franka_controller_pb2.FrankaControlMessage.ControllerType.CARTESIAN_VELOCITY
            )
            self._set_traj_interpolator_config(control_msg, controller_cfg)
            control_msg.control_msg.Pack(cartesian_velocity_msg)
            control_msg.timeout = 0.2
            control_msg.termination = termination

            control_msg.state_estimator_msg.CopyFrom(state_estimator_msg)
            self._send_control_msg(control_msg)
        else:
            raise ValueError(f"Unsupported controller_type: {controller_type}")

        if gripper_action is not None:
            self.gripper_control(gripper_action)

        if self.use_visualizer and self.last_state is not None:
            self.visualizer.update(joint_positions=np.array(self.last_state.q))

    def gripper_control(self, action: float):
        """Control the gripper

        Args:
            action (float): The control command for Franka gripper. Currently assuming scalar control commands.
        """

        gripper_control_msg = franka_controller_pb2.FrankaGripperControlMessage()

        # action 0-> 1 : Grasp
        # action 1-> 0 : Release

        # TODO (Yifeng): Test if sending grasping or gripper directly
        # will stop executing the previous command
        if action < 0.0:  #  and self.last_gripper_action == 1):
            move_msg = franka_controller_pb2.FrankaGripperMoveMessage()
            move_msg.width = 0.08 * np.abs(action)
            move_msg.speed = 0.1
            gripper_control_msg.control_msg.Pack(move_msg)

            logger.debug("Gripper opening")

            self._gripper_publisher.send(gripper_control_msg.SerializeToString())
        elif action >= 0.0:  #  and self.last_gripper_action == 0:
            grasp_msg = franka_controller_pb2.FrankaGripperGraspMessage()
            grasp_msg.width = -0.01
            grasp_msg.speed = 0.5
            grasp_msg.force = 30.0
            grasp_msg.epsilon_inner = 0.08
            grasp_msg.epsilon_outer = 0.08

            gripper_control_msg.control_msg.Pack(grasp_msg)

            logger.debug("Gripper closing")

            self._gripper_publisher.send(gripper_control_msg.SerializeToString())
        self.last_gripper_action = action

    def wait_for_state(self, timeout: float = 5.0, poll_interval: float = 0.01) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.last_state is not None:
                return True
            time.sleep(poll_interval)
        return self.last_state is not None

    @staticmethod
    def _as_absolute_controller_cfg(controller_cfg, controller_type: str):
        cfg = copy.deepcopy(
            controller_cfg
            if controller_cfg is not None
            else get_default_controller_config(controller_type)
        )
        cfg.is_delta = False
        return cfg

    @staticmethod
    def _normalize_joint_target(joint_positions) -> np.ndarray:
        q = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
        if q.size != 7:
            raise ValueError("joint_positions must contain exactly 7 joint values.")
        return q

    @staticmethod
    def _normalize_pose_target(target_pose) -> np.ndarray:
        pose = np.asarray(target_pose, dtype=np.float64)
        if pose.shape == (4, 4):
            rot = pose[:3, :3]
            pos = pose[:3, 3]
            quat = FrankaInterface._canonicalize_quaternion(
                transform_utils.mat2quat(rot)
            )
            return np.concatenate([pos, quat]).astype(np.float64)

        pose = pose.reshape(-1)
        if pose.size == 7:
            pose = pose.copy()
            pose[3:7] = FrankaInterface._canonicalize_quaternion(pose[3:7])
            return pose
        if pose.size != 6:
            raise ValueError(
                "target_pose must be a 4x4 transform, a 6D [x, y, z, ax, ay, az] vector, "
                "or a 7D [x, y, z, qx, qy, qz, qw] vector."
            )
        return pose

    @staticmethod
    def _resolve_named_joint_target(
        target, named_targets: Union[None, str, Mapping[str, Union[list, np.ndarray]]] = None
    ) -> np.ndarray:
        if not isinstance(target, str):
            return FrankaInterface._normalize_joint_target(target)

        if named_targets is None:
            from peirastic.netft_calib.named_joint_states import load_named_joint_states

            target_map = load_named_joint_states(None)
        elif isinstance(named_targets, str):
            from peirastic.netft_calib.named_joint_states import load_named_joint_states

            target_map = load_named_joint_states(named_targets)
        else:
            target_map = named_targets

        if target not in target_map:
            known = ", ".join(sorted(target_map.keys()))
            raise KeyError(f"Unknown named joint target '{target}'. Known targets: {known}")
        return FrankaInterface._normalize_joint_target(target_map[target])

    @staticmethod
    def _pose_error(current_pose: np.ndarray, target_pose: np.ndarray) -> Tuple[float, float]:
        pos_error = float(np.linalg.norm(current_pose[:3, 3] - target_pose[:3, 3]))
        current_quat = transform_utils.mat2quat(current_pose[:3, :3])
        target_quat = transform_utils.mat2quat(target_pose[:3, :3])
        quat_error = transform_utils.quat_distance(target_quat, current_quat)
        quat_w = float(np.clip(np.abs(quat_error[3]), -1.0, 1.0))
        rot_error = float(2.0 * np.arccos(quat_w))
        return pos_error, rot_error

    def move_joints(
        self,
        joint_positions,
        *,
        gripper_action: float = -1.0,
        controller_cfg=None,
        blocking: bool = True,
        timeout: float = 10.0,
        position_tolerance: float = 1e-3,
        state_timeout: float = 5.0,
        shutdown_check: Optional[Callable[[], bool]] = None,
        progress_interval: float = 0.0,
        progress_callback: Optional[Callable[[float, float], None]] = None,
    ) -> bool:
        q_target = self._normalize_joint_target(joint_positions)
        cfg = self._as_absolute_controller_cfg(controller_cfg, "JOINT_POSITION")
        action = q_target.tolist() + [float(gripper_action)]

        if not blocking:
            self.control("JOINT_POSITION", action, controller_cfg=cfg, shutdown_check=shutdown_check)
            return True

        if not self.wait_for_state(timeout=state_timeout):
            return False

        deadline = time.time() + timeout
        last_progress_time = 0.0
        while time.time() < deadline:
            if shutdown_check is not None and shutdown_check():
                return False
            current_q = self.last_q
            if current_q is not None:
                max_position_error = float(np.max(np.abs(current_q - q_target)))
                if progress_interval > 0.0 and time.time() - last_progress_time >= progress_interval:
                    current_dq = self.last_dq
                    max_joint_speed = (
                        float(np.max(np.abs(current_dq))) if current_dq is not None else float("nan")
                    )
                    if progress_callback is not None:
                        progress_callback(max_position_error, max_joint_speed)
                    else:
                        logger.info(
                            "move_joints progress: max_position_error=%.6f max_joint_speed=%.6f",
                            max_position_error,
                            max_joint_speed,
                        )
                    last_progress_time = time.time()
                if max_position_error <= position_tolerance:
                    return True
            self.control("JOINT_POSITION", action, controller_cfg=cfg, shutdown_check=shutdown_check)

        current_q = self.last_q
        return current_q is not None and np.max(np.abs(current_q - q_target)) <= position_tolerance

    def reset_joints(
        self,
        name_or_q,
        *,
        named_targets: Union[None, str, Mapping[str, Union[list, np.ndarray]]] = None,
        gripper_action: float = -1.0,
        controller_cfg=None,
        blocking: bool = True,
        timeout: float = 10.0,
        position_tolerance: float = 1e-3,
        state_timeout: float = 5.0,
        shutdown_check: Optional[Callable[[], bool]] = None,
        progress_interval: float = 0.0,
        progress_callback: Optional[Callable[[float, float], None]] = None,
    ) -> bool:
        q_target = self._resolve_named_joint_target(name_or_q, named_targets=named_targets)
        return self.move_joints(
            q_target,
            gripper_action=gripper_action,
            controller_cfg=controller_cfg,
            blocking=blocking,
            timeout=timeout,
            position_tolerance=position_tolerance,
            state_timeout=state_timeout,
            shutdown_check=shutdown_check,
            progress_interval=progress_interval,
            progress_callback=progress_callback,
        )

    def move_pose(
        self,
        target_pose,
        *,
        gripper_action: float = -1.0,
        controller_cfg=None,
        blocking: bool = True,
        timeout: float = 10.0,
        position_tolerance: float = 2e-3,
        rotation_tolerance: float = 5e-2,
        state_timeout: float = 5.0,
        shutdown_check: Optional[Callable[[], bool]] = None,
    ) -> bool:
        pose_vec = self._normalize_pose_target(target_pose)
        cfg = self._as_absolute_controller_cfg(controller_cfg, "CARTESIAN_VELOCITY")
        action = pose_vec.tolist() + [float(gripper_action)]

        if not blocking:
            self.control("CARTESIAN_VELOCITY", action, controller_cfg=cfg, shutdown_check=shutdown_check)
            return True

        if not self.wait_for_state(timeout=state_timeout):
            return False

        target_pose_mat = np.eye(4, dtype=np.float64)
        target_pose_mat[:3, 3] = pose_vec[:3]
        target_quat = (
            pose_vec[3:7]
            if pose_vec.size == 7
            else transform_utils.axisangle2quat(pose_vec[3:6])
        )
        target_pose_mat[:3, :3] = transform_utils.quat2mat(target_quat)

        deadline = time.time() + timeout
        while time.time() < deadline:
            if shutdown_check is not None and shutdown_check():
                return False
            current_pose = self.last_eef_pose
            if current_pose is not None:
                pos_error, rot_error = self._pose_error(current_pose, target_pose_mat)
                if pos_error <= position_tolerance and rot_error <= rotation_tolerance:
                    return True
            self.control("CARTESIAN_VELOCITY", action, controller_cfg=cfg, shutdown_check=shutdown_check)

        current_pose = self.last_eef_pose
        if current_pose is None:
            return False
        pos_error, rot_error = self._pose_error(current_pose, target_pose_mat)
        return pos_error <= position_tolerance and rot_error <= rotation_tolerance

    def reset_pose(
        self,
        target_pose,
        *,
        gripper_action: float = -1.0,
        controller_cfg=None,
        blocking: bool = True,
        timeout: float = 10.0,
        position_tolerance: float = 2e-3,
        rotation_tolerance: float = 5e-2,
        state_timeout: float = 5.0,
        shutdown_check: Optional[Callable[[], bool]] = None,
    ) -> bool:
        return self.move_pose(
            target_pose,
            gripper_action=gripper_action,
            controller_cfg=controller_cfg,
            blocking=blocking,
            timeout=timeout,
            position_tolerance=position_tolerance,
            rotation_tolerance=rotation_tolerance,
            state_timeout=state_timeout,
            shutdown_check=shutdown_check,
        )

    def close(self):
        if self._closed:
            return

        self._closed = True
        self.termination = True
        self._stop_event.set()
        for socket in (
            self._subscriber,
            self._gripper_subscriber,
            self._publisher,
            self._gripper_publisher,
        ):
            try:
                socket.close(0)
            except zmq.ZMQError:
                pass

        self._state_sub_thread.join(1.0)
        self._gripper_sub_thread.join(1.0)
        try:
            self._context.term()
        except zmq.ZMQError:
            pass

    @property
    def last_eef_pose(self) -> np.ndarray:
        """_summary_

        Returns:
            np.ndarray: The 4x4 homogeneous matrix of end effector pose.
        """
        state = self.last_state
        if state is None:
            return None
        return np.array(state.O_T_EE).reshape(4, 4).transpose()

    @property
    def last_eef_rot_and_pos(self) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Returns:
            Tuple[np.ndarray, np.ndarray]: (eef_rot, eef_pos), eef_rot in rotation matrix, eef_pos in 3d vector.
        """
        state = self.last_state
        if state is None:
            return None, None
        O_T_EE = np.array(state.O_T_EE).reshape(4, 4).transpose()
        return O_T_EE[:3, :3], O_T_EE[:3, 3:]

    @property
    def last_eef_quat_and_pos(self) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Returns:
            Tuple[np.ndarray, np.ndarray]: (eef_quat, eef_pos), eef_quat in quaternion (xyzw), eef_pos in 3d vector.
        """
        state = self.last_state
        if state is None:
            return None, None
        O_T_EE = np.array(state.O_T_EE).reshape(4, 4).transpose()
        return transform_utils.mat2quat(O_T_EE[:3, :3]), O_T_EE[:3, 3:]

    def check_nonzero_configuration(self) -> bool:
        """Check nonzero configuration.

        Returns:
            bool: The boolean variable that indicates if the reading of robot joint configuration is non-zero.
        """
        state = self.last_state
        if state is None:
            return False
        if np.max(np.abs(np.array(state.O_T_EE))) < 1e-3:
            return False
        return True

    def reset(self):
        """Reset internal states of FrankaInterface and clear buffers. Useful when you run multiple episodes in a single python interpretor process."""
        self.clear_state_buffers()

        self.counter = 0
        self.termination = False
        self._stop_event.clear()

        self.last_time = None
        self._last_controller_type = "Dummy"
        self._control_session = 0
        self._session_hard_reset_once = False
        self.last_gripper_dim = 1 if self.has_gripper else 0
        self.last_gripper_action = 0
        self.last_gripper_command_counter = 0
        self._history_actions = []

    @property
    def received_states(self):
        return self.state_buffer_size > 0

    @property
    def last_q(self) -> np.ndarray:
        state = self.last_state
        if state is None:
            return None
        return np.array(state.q)

    @property
    def last_q_d(self) -> np.ndarray:
        state = self.last_state
        if state is None:
            return None
        return np.array(state.q_d)

    @property
    def last_gripper_q(self) -> np.ndarray:
        with self._gripper_state_lock:
            if len(self._gripper_state_buffer) == 0:
                return None
            return np.array(self._gripper_state_buffer[-1].width)

    @property
    def last_dq(self) -> np.ndarray:
        state = self.last_state
        if state is None:
            return None
        return np.array(state.dq)

    @property
    def last_state(self):
        """Default state"""
        with self._state_lock:
            if len(self._state_buffer) == 0:
                return None
            return self._state_buffer[-1]

    @property
    def last_pose(self):
        state = self.last_state
        if state is None:
            return None
        return np.array(state.O_T_EE).reshape(4, 4).transpose()

    @property
    def state_buffer_size(self) -> int:
        with self._state_lock:
            return len(self._state_buffer)

    @property
    def gripper_state_buffer_size(self) -> int:
        with self._gripper_state_lock:
            return len(self._gripper_state_buffer)

    def get_state_buffer_snapshot(self):
        with self._state_lock:
            return list(self._state_buffer)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def ip(self) -> str:
        return self._ip

    @property
    def pub_port(self) -> int:
        return self._pub_port

    @property
    def sub_port(self) -> int:
        return self._sub_port

    @property
    def gripper_pub_port(self) -> int:
        return self._gripper_pub_port

    @property
    def gripper_sub_port(self) -> int:
        return self._gripper_sub_port
