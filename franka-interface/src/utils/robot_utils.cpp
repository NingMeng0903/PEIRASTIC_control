// Copyright 2022 Yifeng Zhu

#include "utils/robot_utils.h"

namespace {
template <typename RepeatedField, typename Container>
void AppendRepeatedField(RepeatedField *field, const Container &container) {
  for (const auto &value : container) {
    field->Add(value);
  }
}
} // namespace

namespace robot_utils {

void FrankaRobotStateUtils::LoadErrorStateToMsg(
    const franka::Errors &robot_state_error,
    FrankaRobotStateMessage::Errors &error_msg) {
  error_msg.set_joint_position_limits_violation(
      robot_state_error.joint_position_limits_violation);
  error_msg.set_cartesian_position_limits_violation(
      robot_state_error.cartesian_position_limits_violation);
  error_msg.set_self_collision_avoidance_violation(
      robot_state_error.self_collision_avoidance_violation);
  error_msg.set_joint_velocity_violation(
      robot_state_error.joint_velocity_violation);
  error_msg.set_cartesian_velocity_violation(
      robot_state_error.cartesian_velocity_violation);
  error_msg.set_force_control_safety_violation(
      robot_state_error.force_control_safety_violation);
  error_msg.set_joint_reflex(robot_state_error.joint_reflex);
  error_msg.set_cartesian_reflex(robot_state_error.cartesian_reflex);
  error_msg.set_max_goal_pose_deviation_violation(
      robot_state_error.max_goal_pose_deviation_violation);
  error_msg.set_max_path_pose_deviation_violation(
      robot_state_error.max_path_pose_deviation_violation);
  error_msg.set_cartesian_velocity_profile_safety_violation(
      robot_state_error.cartesian_velocity_profile_safety_violation);
  error_msg.set_joint_position_motion_generator_start_pose_invalid(
      robot_state_error.joint_position_motion_generator_start_pose_invalid);
  error_msg.set_joint_motion_generator_position_limits_violation(
      robot_state_error.joint_motion_generator_position_limits_violation);
  error_msg.set_joint_motion_generator_velocity_limits_violation(
      robot_state_error.joint_motion_generator_velocity_limits_violation);
  error_msg.set_joint_motion_generator_velocity_discontinuity(
      robot_state_error.joint_motion_generator_velocity_discontinuity);
  error_msg.set_joint_motion_generator_acceleration_discontinuity(
      robot_state_error.joint_motion_generator_acceleration_discontinuity);
  error_msg.set_cartesian_position_motion_generator_start_pose_invalid(
      robot_state_error.cartesian_position_motion_generator_start_pose_invalid);
  error_msg.set_cartesian_motion_generator_elbow_limit_violation(
      robot_state_error.cartesian_motion_generator_elbow_limit_violation);
  error_msg.set_cartesian_motion_generator_velocity_limits_violation(
      robot_state_error.cartesian_motion_generator_velocity_limits_violation);
  error_msg.set_cartesian_motion_generator_velocity_discontinuity(
      robot_state_error.cartesian_motion_generator_velocity_discontinuity);
  error_msg.set_cartesian_motion_generator_acceleration_discontinuity(
      robot_state_error.cartesian_motion_generator_acceleration_discontinuity);
  error_msg.set_cartesian_motion_generator_elbow_sign_inconsistent(
      robot_state_error.cartesian_motion_generator_elbow_sign_inconsistent);
  error_msg.set_cartesian_motion_generator_start_elbow_invalid(
      robot_state_error.cartesian_motion_generator_start_elbow_invalid);
  error_msg.set_cartesian_motion_generator_joint_position_limits_violation(
      robot_state_error
          .cartesian_motion_generator_joint_position_limits_violation);
  error_msg.set_cartesian_motion_generator_joint_velocity_limits_violation(
      robot_state_error
          .cartesian_motion_generator_joint_velocity_limits_violation);
  error_msg.set_cartesian_motion_generator_joint_velocity_discontinuity(
      robot_state_error
          .cartesian_motion_generator_joint_velocity_discontinuity);
  error_msg.set_cartesian_motion_generator_joint_acceleration_discontinuity(
      robot_state_error
          .cartesian_motion_generator_joint_acceleration_discontinuity);
  error_msg.set_cartesian_position_motion_generator_invalid_frame(
      robot_state_error.cartesian_position_motion_generator_invalid_frame);
  error_msg.set_force_controller_desired_force_tolerance_violation(
      robot_state_error.force_controller_desired_force_tolerance_violation);
  error_msg.set_controller_torque_discontinuity(
      robot_state_error.controller_torque_discontinuity);
  error_msg.set_start_elbow_sign_inconsistent(
      robot_state_error.start_elbow_sign_inconsistent);
  error_msg.set_communication_constraints_violation(
      robot_state_error.communication_constraints_violation);
  error_msg.set_power_limit_violation(robot_state_error.power_limit_violation);
  error_msg.set_joint_p2p_insufficient_torque_for_planning(
      robot_state_error.joint_p2p_insufficient_torque_for_planning);
  error_msg.set_tau_j_range_violation(robot_state_error.tau_j_range_violation);
  error_msg.set_instability_detected(robot_state_error.instability_detected);
  error_msg.set_joint_move_in_wrong_direction(
      robot_state_error.joint_move_in_wrong_direction);
}

void FrankaRobotStateUtils::LoadRobotStateToMsg(
    const franka::RobotState &robot_state,
    FrankaRobotStateMessage &robot_state_msg) {

  AppendRepeatedField(robot_state_msg.mutable_o_t_ee(), robot_state.O_T_EE);
  AppendRepeatedField(robot_state_msg.mutable_o_t_ee_d(), robot_state.O_T_EE_d);

  AppendRepeatedField(robot_state_msg.mutable_f_t_ee(), robot_state.F_T_EE);
  AppendRepeatedField(robot_state_msg.mutable_f_t_ne(), robot_state.F_T_NE);
  AppendRepeatedField(robot_state_msg.mutable_ne_t_ee(), robot_state.NE_T_EE);
  AppendRepeatedField(robot_state_msg.mutable_ee_t_k(), robot_state.EE_T_K);
  robot_state_msg.set_m_ee(robot_state.m_ee);
  AppendRepeatedField(robot_state_msg.mutable_i_ee(), robot_state.I_ee);
  AppendRepeatedField(robot_state_msg.mutable_f_x_cee(), robot_state.F_x_Cee);
  robot_state_msg.set_m_load(robot_state.m_load);
  AppendRepeatedField(robot_state_msg.mutable_i_load(), robot_state.I_load);
  AppendRepeatedField(robot_state_msg.mutable_f_x_cload(), robot_state.F_x_Cload);
  robot_state_msg.set_m_total(robot_state.m_total);
  AppendRepeatedField(robot_state_msg.mutable_i_total(), robot_state.I_total);
  AppendRepeatedField(robot_state_msg.mutable_f_x_ctotal(), robot_state.F_x_Ctotal);
  AppendRepeatedField(robot_state_msg.mutable_elbow(), robot_state.elbow);
  AppendRepeatedField(robot_state_msg.mutable_elbow_d(), robot_state.elbow_d);
  AppendRepeatedField(robot_state_msg.mutable_elbow_c(), robot_state.elbow_c);
  AppendRepeatedField(robot_state_msg.mutable_delbow_c(), robot_state.delbow_c);
  AppendRepeatedField(robot_state_msg.mutable_ddelbow_c(), robot_state.ddelbow_c);
  AppendRepeatedField(robot_state_msg.mutable_tau_j(), robot_state.tau_J);
  AppendRepeatedField(robot_state_msg.mutable_tau_j_d(), robot_state.tau_J_d);
  AppendRepeatedField(robot_state_msg.mutable_dtau_j(), robot_state.dtau_J);
  AppendRepeatedField(robot_state_msg.mutable_q(), robot_state.q);
  AppendRepeatedField(robot_state_msg.mutable_q_d(), robot_state.q_d);
  AppendRepeatedField(robot_state_msg.mutable_dq(), robot_state.dq);
  AppendRepeatedField(robot_state_msg.mutable_dq_d(), robot_state.dq_d);
  AppendRepeatedField(robot_state_msg.mutable_ddq_d(), robot_state.ddq_d);
  AppendRepeatedField(robot_state_msg.mutable_joint_contact(), robot_state.joint_contact);
  AppendRepeatedField(robot_state_msg.mutable_cartesian_contact(), robot_state.cartesian_contact);
  AppendRepeatedField(robot_state_msg.mutable_joint_collision(), robot_state.joint_collision);
  AppendRepeatedField(robot_state_msg.mutable_cartesian_collision(), robot_state.cartesian_collision);
  AppendRepeatedField(robot_state_msg.mutable_tau_ext_hat_filtered(),
                      robot_state.tau_ext_hat_filtered);
  AppendRepeatedField(robot_state_msg.mutable_o_f_ext_hat_k(),
                      robot_state.O_F_ext_hat_K);
  AppendRepeatedField(robot_state_msg.mutable_k_f_ext_hat_k(),
                      robot_state.K_F_ext_hat_K);
  AppendRepeatedField(robot_state_msg.mutable_o_dp_ee_d(), robot_state.O_dP_EE_d);
  AppendRepeatedField(robot_state_msg.mutable_o_t_ee_c(), robot_state.O_T_EE_c);
  AppendRepeatedField(robot_state_msg.mutable_o_dp_ee_c(), robot_state.O_dP_EE_c);
  AppendRepeatedField(robot_state_msg.mutable_o_ddp_ee_c(),
                      robot_state.O_ddP_EE_c);
  AppendRepeatedField(robot_state_msg.mutable_theta(), robot_state.theta);
  AppendRepeatedField(robot_state_msg.mutable_dtheta(), robot_state.dtheta);

  // Error
  FrankaRobotStateMessage::Errors *current_errors =
      new FrankaRobotStateMessage::Errors();
  this->LoadErrorStateToMsg(robot_state.current_errors, *current_errors);

  FrankaRobotStateMessage::Errors *last_motion_errors =
      new FrankaRobotStateMessage::Errors();
  this->LoadErrorStateToMsg(robot_state.last_motion_errors,
                            *last_motion_errors);

  robot_state_msg.set_allocated_current_errors(current_errors);
  robot_state_msg.set_allocated_last_motion_errors(last_motion_errors);

  robot_state_msg.set_control_command_success_rate(
      robot_state.control_command_success_rate);

  FrankaRobotStateMessage::Duration *time =
      new FrankaRobotStateMessage::Duration();
  time->set_tosec(robot_state.time.toSec());
  time->set_tomsec(robot_state.time.toMSec());

  robot_state_msg.set_allocated_time(time);

  // set robot mode

  switch (robot_state.robot_mode) {
  case franka::RobotMode::kOther:
    robot_state_msg.set_robot_mode(FrankaRobotStateMessage_RobotMode_Other);
    break;
  case franka::RobotMode::kIdle:
    robot_state_msg.set_robot_mode(FrankaRobotStateMessage_RobotMode_Idle);
    break;
  case franka::RobotMode::kMove:
    robot_state_msg.set_robot_mode(FrankaRobotStateMessage_RobotMode_Move);
    break;
  case franka::RobotMode::kGuiding:
    robot_state_msg.set_robot_mode(FrankaRobotStateMessage_RobotMode_Guiding);
    break;
  case franka::RobotMode::kReflex:
    robot_state_msg.set_robot_mode(FrankaRobotStateMessage_RobotMode_Reflex);
    break;
  case franka::RobotMode::kUserStopped:
    robot_state_msg.set_robot_mode(
        FrankaRobotStateMessage_RobotMode_UserStopped);
    break;
  case franka::RobotMode::kAutomaticErrorRecovery:
    robot_state_msg.set_robot_mode(
        FrankaRobotStateMessage_RobotMode_AutomaticErrorRecovery);
    break;
  }
}

void FrankaGripperStateUtils::LoadGripperStateToMsg(
    const franka::GripperState &gripper_state,
    FrankaGripperStateMessage &gripper_state_msg) {
  FrankaGripperStateMessage::Duration *time =
      new FrankaGripperStateMessage::Duration();
  time->set_tosec(gripper_state.time.toSec());
  time->set_tomsec(gripper_state.time.toMSec());

  gripper_state_msg.set_width(gripper_state.width);
  gripper_state_msg.set_max_width(gripper_state.max_width);
  gripper_state_msg.set_is_grasped(gripper_state.is_grasped);
  gripper_state_msg.set_temperature(gripper_state.temperature);
  gripper_state_msg.set_allocated_time(time);
}

StatePublisher::StatePublisher(std::string pub_port, int state_pub_rate)
    : zmq_publisher_(pub_port), state_pub_rate_(state_pub_rate) {}

StatePublisher::~StatePublisher() {}

void StatePublisher::StartPublishing() {
  running_ = true;
  state_pub_thread_ = std::thread([&]() {
    int count = 0;
    while (running_) {
      franka::RobotState current_robot_state;

      if (state_.mutex.try_lock()) {
        current_robot_state = state_.robot_state;
        state_.mutex.unlock();
      } else {
        continue;
      }
      FrankaRobotStateMessage robot_state_msg;
      robot_state_utils_.LoadRobotStateToMsg(current_robot_state,
                                             robot_state_msg);
      robot_state_msg.set_frame(count);
      count++;
      std::string msg_str;
      robot_state_msg.SerializeToString(&msg_str);
      zmq_publisher_.send(msg_str);

      std::this_thread::sleep_for(std::chrono::milliseconds(
          static_cast<int>(1. / state_pub_rate_ * 1000)));
    }
  });
}

void StatePublisher::StopPublishing() {
  running_ = false;
  state_pub_thread_.join();
}

void StatePublisher::UpdateNewState(const franka::RobotState &robot_state,
                                    const franka::Model *robot_model) {
  if (state_.mutex.try_lock()) {
    state_.robot_state = robot_state;
    int n_frame = 0;
    for (franka::Frame frame = franka::Frame::kJoint1;
         frame <= franka::Frame::kEndEffector; frame++) {
      auto pose = robot_model->pose(frame, robot_state);
      for (int i = 0; i < 16; i++) {
        state_.current_robot_frames[n_frame * 16 + i] = pose[i];
      }
      n_frame++;
    }
    state_.mutex.unlock();
  }
}

} // namespace robot_utils
