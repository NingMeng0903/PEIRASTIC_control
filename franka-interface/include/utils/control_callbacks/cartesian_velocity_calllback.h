// Copyright 2023 Yifeng Zhu

#include <algorithm>
#include <chrono>
#include <cmath>
#include <franka/model.h>
#include <franka/robot.h>

#include "controllers/cartesian_velocity.h"
#include "utils/control_utils.h"
#include "utils/robot_utils.h"
#include "utils/shared_memory.h"
#include "utils/shared_state.h"


#ifndef UTILS_CONTROL_CALLBACKS_CARTESIAN_VELOCITY_CALLBACK_H_
#define UTILS_CONTROL_CALLBACKS_CARTESIAN_VELOCITY_CALLBACK_H_


namespace control_callbacks {

namespace {
inline Eigen::Quaterniond CanonicalizeQuaternion(
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

inline Eigen::Vector3d QuaternionErrorToAxisAngle(
    const Eigen::Quaterniond &desired_quaternion,
    const Eigen::Quaterniond &actual_quaternion) {
  const Eigen::Quaterniond normalized_actual =
      CanonicalizeQuaternion(actual_quaternion);
  const Eigen::Quaterniond normalized_desired =
      CanonicalizeQuaternion(desired_quaternion, &normalized_actual);
  Eigen::Quaterniond error_quaternion =
      normalized_desired * normalized_actual.conjugate();
  error_quaternion = CanonicalizeQuaternion(error_quaternion);

  const Eigen::AngleAxisd axis_angle(error_quaternion);
  if (!std::isfinite(axis_angle.angle()) || axis_angle.angle() < 1e-12) {
    return Eigen::Vector3d::Zero();
  }

  return axis_angle.angle() * axis_angle.axis();
}
} // namespace

std::function<franka::CartesianVelocities(const franka::RobotState &,
                                          franka::Duration)>
CreateCartesianVelocitiesCallback(
    const std::shared_ptr<SharedMemory> &global_handler,
    const std::shared_ptr<robot_utils::StatePublisher> state_publisher,
    const franka::Model &model, std::shared_ptr<StateInfo> &current_state_info,
    std::shared_ptr<StateInfo> &goal_state_info, const int &policy_rate,
    const int &traj_rate) {
  return [&global_handler, &state_publisher, &model, &current_state_info,
          &goal_state_info, &policy_rate,
          &traj_rate](const franka::RobotState &robot_state,
                      franka::Duration period) -> franka::CartesianVelocities {
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();

    Eigen::Affine3d current_T_EE_in_base_frame(
        Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));

    if (!global_handler->running) {
      franka::CartesianVelocities zero_velocities{
          {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
      return franka::MotionFinished(zero_velocities);
    }

    std::lock_guard<std::mutex> control_lock(global_handler->control_mutex);
    auto velocity_controller =
        std::dynamic_pointer_cast<controller::CartesianVelocityController>(
            global_handler->controller_ptr);
    auto traj_interpolator = global_handler->traj_interpolator_ptr;
    if (!velocity_controller || !traj_interpolator) {
      return franka::CartesianVelocities{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    }

    const bool use_pose_tracking = velocity_controller->UsesPoseTracking();
    std::array<double, 6> vel_d_array{};
    Eigen::Vector3d desired_twist_trans_EE_in_base_frame =
        Eigen::Vector3d::Zero();
    Eigen::Vector3d desired_twist_rot_EE_in_base_frame =
        Eigen::Vector3d::Zero();
    Eigen::Vector3d desired_pos_EE_in_base_frame =
        current_T_EE_in_base_frame.translation();
    Eigen::Quaterniond desired_quat_EE_in_base_frame(
        current_T_EE_in_base_frame.linear());
    StateInfo current_state_snapshot;
    StateInfo goal_state_snapshot;

    {
      std::lock_guard<std::mutex> lock(global_handler->state_info_mutex);
      current_state_info->joint_positions =
          Eigen::VectorXd::Map(robot_state.q.data(), 7);
      current_state_info->pos_EE_in_base_frame
          << current_T_EE_in_base_frame.translation();
      current_state_info->quat_EE_in_base_frame =
          Eigen::Quaterniond(current_T_EE_in_base_frame.linear());
      velocity_controller->EstimateVelocities(robot_state, current_state_info);
      current_state_snapshot = *current_state_info;
      goal_state_snapshot = *goal_state_info;
    }

    if (!use_pose_tracking && global_handler->time == 0.0) {
      traj_interpolator->Reset(
          0.0, current_state_snapshot.twist_trans_EE_in_base_frame,
          current_state_snapshot.twist_rot_EE_in_base_frame,
          goal_state_snapshot.twist_trans_EE_in_base_frame,
          goal_state_snapshot.twist_rot_EE_in_base_frame, policy_rate,
          traj_rate, global_handler->traj_interpolator_time_fraction);
    }

    global_handler->time += period.toSec();

    if (use_pose_tracking) {
      Eigen::Affine3d last_cmd_T_EE_in_base_frame(
          Eigen::Matrix4d::Map(robot_state.O_T_EE_c.data()));
      traj_interpolator->SyncCartesianCommandState(
          last_cmd_T_EE_in_base_frame.translation(),
          Eigen::Quaterniond(last_cmd_T_EE_in_base_frame.linear()),
          robot_state.O_dP_EE_c, robot_state.O_ddP_EE_c);
      traj_interpolator->GetNextStep(
          global_handler->time, desired_pos_EE_in_base_frame,
          desired_quat_EE_in_base_frame, desired_twist_trans_EE_in_base_frame,
          desired_twist_rot_EE_in_base_frame);
      // Keep Ruckig smoothing on translation only. For orientation, track the
      // streamed absolute quaternion directly to avoid rotvec/AngleAxis branch
      // changes near upside-down poses.
      if (goal_state_snapshot.quat_EE_in_base_frame.norm() > 1e-12) {
        desired_quat_EE_in_base_frame = CanonicalizeQuaternion(
            goal_state_snapshot.quat_EE_in_base_frame,
            &desired_quat_EE_in_base_frame);
      }
      desired_twist_rot_EE_in_base_frame.setZero();
      const Eigen::Quaterniond current_quat_EE_in_base_frame(
          current_T_EE_in_base_frame.linear());
      const Eigen::Vector3d position_error =
          desired_pos_EE_in_base_frame -
          current_T_EE_in_base_frame.translation();
      const Eigen::Vector3d rotation_error =
          QuaternionErrorToAxisAngle(desired_quat_EE_in_base_frame,
                                     current_quat_EE_in_base_frame);
      const double damping_gain = velocity_controller->GetDampingGain();
      const Eigen::Vector3d commanded_translation =
          desired_twist_trans_EE_in_base_frame +
          velocity_controller->GetTranslationKp().cwiseProduct(position_error) -
          damping_gain * current_state_snapshot.twist_trans_EE_in_base_frame;
      const Eigen::Vector3d commanded_rotation =
          desired_twist_rot_EE_in_base_frame +
          velocity_controller->GetRotationKp().cwiseProduct(rotation_error) -
          damping_gain * current_state_snapshot.twist_rot_EE_in_base_frame;

      vel_d_array[0] = commanded_translation.x();
      vel_d_array[1] = commanded_translation.y();
      vel_d_array[2] = commanded_translation.z();
      vel_d_array[3] = commanded_rotation.x();
      vel_d_array[4] = commanded_rotation.y();
      vel_d_array[5] = commanded_rotation.z();

      const double dt = std::max(period.toSec(), 1e-6);
      const double max_velocity_step =
          velocity_controller->GetCommandAccelerationLimit() * dt;
      if (!global_handler->has_previous_cartesian_velocity_command) {
        global_handler->previous_cartesian_velocity_command = robot_state.O_dP_EE_c;
        global_handler->has_previous_cartesian_velocity_command = true;
      }
      for (size_t i = 0; i < vel_d_array.size(); ++i) {
        const double previous_velocity =
            global_handler->previous_cartesian_velocity_command[i];
        const double velocity_difference = vel_d_array[i] - previous_velocity;
        if (velocity_difference > max_velocity_step) {
          vel_d_array[i] = previous_velocity + max_velocity_step;
        } else if (velocity_difference < -max_velocity_step) {
          vel_d_array[i] = previous_velocity - max_velocity_step;
        }
      }
    } else {
      traj_interpolator->GetNextStep(
          global_handler->time, desired_twist_trans_EE_in_base_frame,
          desired_twist_rot_EE_in_base_frame);
      vel_d_array = velocity_controller->Step(
          robot_state, desired_twist_trans_EE_in_base_frame,
          desired_twist_rot_EE_in_base_frame);
    }

    state_publisher->UpdateNewState(robot_state, &model);
    control_utils::CartesianVelocitySafetyGuardFn(
        vel_d_array, global_handler->min_trans_speed,
        global_handler->max_trans_speed, global_handler->min_rot_speed,
        global_handler->max_rot_speed);
    global_handler->previous_cartesian_velocity_command = vel_d_array;
    global_handler->has_previous_cartesian_velocity_command = true;

    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    global_handler->logger->debug("{0} microseconds", time.count());
    return vel_d_array;
  };
}
} // namespace control_callbacks

#endif // UTILS_CONTROL_CALLBACKS_CARTESIAN_VELOCITY_CALLBACK_H_

