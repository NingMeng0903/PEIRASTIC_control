// Copyright 2023 Yifeng Zhu

#include "franka_controller.pb.h"
#include "franka_robot_state.pb.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>

#include "utils/common_utils.h"
#include "utils/control_utils.h"
#include "utils/robot_utils.h"

#include "controllers/cartesian_velocity.h"

#include <memory>

namespace {
Eigen::Quaterniond CanonicalizeQuaternion(
    const Eigen::Quaterniond &quaternion,
    const Eigen::Quaterniond *reference = nullptr) {
  Eigen::Quaterniond normalized = quaternion.normalized();
  if (reference != nullptr &&
      normalized.coeffs().dot(reference->normalized().coeffs()) < 0.0) {
    normalized.coeffs() *= -1.0;
  } else if (reference == nullptr && normalized.w() < 0.0) {
    normalized.coeffs() *= -1.0;
  }
  return normalized;
}

Eigen::Quaterniond AxisAngleVectorToQuaternion(const Eigen::Vector3d &axis_angle) {
  const double angle = axis_angle.norm();
  if (angle < 1e-12) {
    return Eigen::Quaterniond::Identity();
  }

  return CanonicalizeQuaternion(
      Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis_angle / angle)));
}

Eigen::Quaterniond QuaternionFromXyzw(const FrankaCartesianVelocityControllerMessage &msg) {
  return CanonicalizeQuaternion(
      Eigen::Quaterniond(msg.absolute_quaternion(3), msg.absolute_quaternion(0),
                         msg.absolute_quaternion(1), msg.absolute_quaternion(2)));
}
} // namespace

namespace controller {
CartesianVelocityController::CartesianVelocityController() {}
CartesianVelocityController::~CartesianVelocityController() {}

CartesianVelocityController::CartesianVelocityController(franka::Model &model) {
  model_ = &model;
}

bool CartesianVelocityController::ParseMessage(const FrankaControlMessage &msg) {

  if (!msg.control_msg().UnpackTo(&control_msg_)) {
    return false;
  }

  speed_factor_ = control_msg_.speed_factor();
  use_pose_tracking_ =
      msg.traj_interpolator_type() ==
      FrankaControlMessage_TrajInterpolatorType_RUCKIG_POSE;
  command_acceleration_limit_ = msg.traj_interpolator_config().max_acceleration() > 0.0
                                    ? msg.traj_interpolator_config().max_acceleration()
                                    : 1.0;
  damping_gain_ = control_msg_.kd_gains() > 0.0 ? control_msg_.kd_gains() : 0.0;
  translation_kp_.setConstant(2.5);
  rotation_kp_.setConstant(1.0);
  if (control_msg_.translation_kp_size() == 3) {
    translation_kp_ << control_msg_.translation_kp(0),
        control_msg_.translation_kp(1), control_msg_.translation_kp(2);
  }
  if (control_msg_.rotation_kp_size() == 3) {
    rotation_kp_ << control_msg_.rotation_kp(0), control_msg_.rotation_kp(1),
        control_msg_.rotation_kp(2);
  }

  this->state_estimator_ptr_->ParseMessage(msg.state_estimator_msg());

  return true;

}

void CartesianVelocityController::ComputeGoal(const std::shared_ptr<StateInfo> &state_info,
                std::shared_ptr<StateInfo> &goal_state_info) {
  if (!use_pose_tracking_) {
    goal_state_info->twist_trans_EE_in_base_frame =
        Eigen::Vector3d(control_msg_.goal().x(), control_msg_.goal().y(),
                        control_msg_.goal().z());
    goal_state_info->twist_rot_EE_in_base_frame =
        Eigen::Vector3d(control_msg_.goal().ax(), control_msg_.goal().ay(),
                        control_msg_.goal().az());
    return;
  }

  const Eigen::Vector3d goal_position(control_msg_.goal().x(),
                                      control_msg_.goal().y(),
                                      control_msg_.goal().z());
  const Eigen::Vector3d goal_axis_angle(control_msg_.goal().ax(),
                                        control_msg_.goal().ay(),
                                        control_msg_.goal().az());
  const Eigen::Quaterniond delta_quaternion =
      AxisAngleVectorToQuaternion(goal_axis_angle);

  if (control_msg_.goal().is_delta()) {
    if (!has_reference_goal_) {
      reference_goal_position_ = state_info->pos_EE_in_base_frame;
      reference_goal_quaternion_ =
          CanonicalizeQuaternion(state_info->quat_EE_in_base_frame);
      has_reference_goal_ = true;
    }

    reference_goal_position_ += goal_position;
    reference_goal_quaternion_ =
        CanonicalizeQuaternion(delta_quaternion * reference_goal_quaternion_,
                               &reference_goal_quaternion_);

    goal_state_info->pos_EE_in_base_frame = reference_goal_position_;
    goal_state_info->quat_EE_in_base_frame = reference_goal_quaternion_;
    return;
  }

  goal_state_info->pos_EE_in_base_frame = goal_position;
  if (control_msg_.absolute_quaternion_size() == 4) {
    const Eigen::Quaterniond absolute_quaternion =
        QuaternionFromXyzw(control_msg_);
    goal_state_info->quat_EE_in_base_frame =
        has_reference_goal_
            ? CanonicalizeQuaternion(absolute_quaternion,
                                     &reference_goal_quaternion_)
            : CanonicalizeQuaternion(absolute_quaternion);
  } else {
    goal_state_info->quat_EE_in_base_frame =
        has_reference_goal_
            ? CanonicalizeQuaternion(delta_quaternion, &reference_goal_quaternion_)
            : CanonicalizeQuaternion(delta_quaternion);
  }
  reference_goal_position_ = goal_state_info->pos_EE_in_base_frame;
  reference_goal_quaternion_ = goal_state_info->quat_EE_in_base_frame;
  has_reference_goal_ = true;
}

void CartesianVelocityController::EstimateVelocities(const franka::RobotState &robot_state, std::shared_ptr<StateInfo>& current_state_info) {
    Eigen::Matrix<double, 7, 1> current_dq;
    // Get state from a specified state estimator
    current_dq = this->state_estimator_ptr_->GetCurrentJointVel();

    // Estimate the current end effector velocities
    std::array<double, 42> jacobian_array =
        model_->zeroJacobian(franka::Frame::kEndEffector, robot_state);
    Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());

    Eigen::MatrixXd jacobian_pos(3, 7);
    Eigen::MatrixXd jacobian_ori(3, 7);
    jacobian_pos << jacobian.block(0, 0, 3, 7);
    jacobian_ori << jacobian.block(3, 0, 3, 7);

    current_state_info->twist_trans_EE_in_base_frame << jacobian_pos * current_dq;
    current_state_info->twist_rot_EE_in_base_frame << jacobian_ori * current_dq;
  
}

std::array<double, 6> CartesianVelocityController::Step(const franka::RobotState &,
                            const Eigen::Vector3d & desired_twist_trans_EE_in_base_frame,
                            const Eigen::Vector3d & desired_twist_rot_EE_in_base_frame) {
  Eigen::Matrix<double, 6, 1> target_v;

  target_v << desired_twist_trans_EE_in_base_frame, desired_twist_rot_EE_in_base_frame;
  
  // TODO: Scale the target command with speed_factor

  std::array<double, 6> vel_d_array{};
  Eigen::VectorXd::Map(&vel_d_array[0], 6) = target_v;
  return vel_d_array;
}

bool CartesianVelocityController::UsesPoseTracking() const {
  return use_pose_tracking_;
}

const Eigen::Vector3d &CartesianVelocityController::GetTranslationKp() const {
  return translation_kp_;
}

const Eigen::Vector3d &CartesianVelocityController::GetRotationKp() const {
  return rotation_kp_;
}

double CartesianVelocityController::GetDampingGain() const {
  return damping_gain_;
}

double CartesianVelocityController::GetCommandAccelerationLimit() const {
  return command_acceleration_limit_;
}

} // namespace controller