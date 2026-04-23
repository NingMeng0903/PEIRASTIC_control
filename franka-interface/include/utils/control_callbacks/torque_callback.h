// Copyright 2022 Yifeng Zhu

#include <chrono>
#include <franka/model.h>
#include <franka/robot.h>

#include "utils/control_utils.h"
#include "utils/robot_utils.h"
#include "utils/shared_memory.h"
#include "utils/shared_state.h"

#ifndef UTILS_CONTROL_CALLBACKS_TORQUE_CALLBACK_H_
#define UTILS_CONTROL_CALLBACKS_TORQUE_CALLBACK_H_

namespace control_callbacks {

std::function<franka::Torques(const franka::RobotState &, franka::Duration)>
CreateTorqueFromCartesianSpaceCallback(
    const std::shared_ptr<SharedMemory> &global_handler,
    const std::shared_ptr<robot_utils::StatePublisher> state_publisher,
    const franka::Model &model, std::shared_ptr<StateInfo> &current_state_info,
    std::shared_ptr<StateInfo> &goal_state_info, const int &policy_rate,
    const int &traj_rate) {
  return [&global_handler, &state_publisher, &model, &current_state_info,
          &goal_state_info, &policy_rate,
          &traj_rate](const franka::RobotState &robot_state,
                      franka::Duration period) -> franka::Torques {
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();

    Eigen::Affine3d current_T_EE_in_base_frame(
        Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));

    if (!global_handler->running) {
      franka::Torques zero_torques{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
      return franka::MotionFinished(zero_torques);
    }

    std::lock_guard<std::mutex> control_lock(global_handler->control_mutex);
    auto controller = global_handler->controller_ptr;
    auto traj_interpolator = global_handler->traj_interpolator_ptr;
    if (!controller || !traj_interpolator) {
      return franka::Torques{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    }

    std::array<double, 7> tau_d_array{};
    Eigen::Vector3d desired_pos_EE_in_base_frame;
    Eigen::Quaterniond desired_quat_EE_in_base_frame;
    StateInfo current_state_snapshot;
    StateInfo goal_state_snapshot;

    {
      std::lock_guard<std::mutex> state_lock(global_handler->state_info_mutex);
      current_state_info->joint_positions =
          Eigen::VectorXd::Map(robot_state.q.data(), 7);
      current_state_info->pos_EE_in_base_frame
          << current_T_EE_in_base_frame.translation();
      current_state_info->quat_EE_in_base_frame =
          Eigen::Quaterniond(current_T_EE_in_base_frame.linear());
      current_state_snapshot = *current_state_info;
      goal_state_snapshot = *goal_state_info;
    }

    if (global_handler->time == 0.) {
      traj_interpolator->Reset(
          0., current_state_snapshot.pos_EE_in_base_frame,
          current_state_snapshot.quat_EE_in_base_frame,
          goal_state_snapshot.pos_EE_in_base_frame,
          goal_state_snapshot.quat_EE_in_base_frame, policy_rate, traj_rate,
          global_handler->traj_interpolator_time_fraction);
    }
    global_handler->time += period.toSec();
    traj_interpolator->GetNextStep(
        global_handler->time, desired_pos_EE_in_base_frame,
        desired_quat_EE_in_base_frame);
    tau_d_array = controller->Step(
        robot_state, desired_pos_EE_in_base_frame,
        desired_quat_EE_in_base_frame);

    state_publisher->UpdateNewState(robot_state, &model);

    std::array<double, 7> tau_d_rate_limited = franka::limitRate(
        franka::kMaxTorqueRate, tau_d_array, robot_state.tau_J_d);
    control_utils::TorqueSafetyGuardFn(tau_d_rate_limited,
                                       global_handler->min_torque,
                                       global_handler->max_torque);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    // global_handler->logger->debug("{0} microseconds" , time.count());

    return tau_d_rate_limited;
  };
}

std::function<franka::Torques(const franka::RobotState &, franka::Duration)>
CreateTorqueFromJointSpaceCallback(
    const std::shared_ptr<SharedMemory> &global_handler,
    const std::shared_ptr<robot_utils::StatePublisher> state_publisher,
    const franka::Model &model, std::shared_ptr<StateInfo> &current_state_info,
    std::shared_ptr<StateInfo> &goal_state_info, const int &policy_rate,
    const int &traj_rate) {
  return [&global_handler, &state_publisher, &model, &current_state_info,
          &goal_state_info, &policy_rate,
          &traj_rate](const franka::RobotState &robot_state,
                      franka::Duration period) -> franka::Torques {
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();

    Eigen::Affine3d current_T_EE_in_base_frame(
        Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));

    if (!global_handler->running) {
      franka::Torques zero_torques{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
      return franka::MotionFinished(zero_torques);
    }

    std::lock_guard<std::mutex> control_lock(global_handler->control_mutex);
    auto controller = global_handler->controller_ptr;
    auto traj_interpolator = global_handler->traj_interpolator_ptr;
    if (!controller || !traj_interpolator) {
      return franka::Torques{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    }

    std::array<double, 7> tau_d_array{};
    Eigen::Matrix<double, 7, 1> desired_q;
    StateInfo current_state_snapshot;
    StateInfo goal_state_snapshot;

    {
      std::lock_guard<std::mutex> state_lock(global_handler->state_info_mutex);
      current_state_info->joint_positions =
          Eigen::VectorXd::Map(robot_state.q.data(), 7);
      current_state_info->pos_EE_in_base_frame
          << current_T_EE_in_base_frame.translation();
      current_state_info->quat_EE_in_base_frame =
          Eigen::Quaterniond(current_T_EE_in_base_frame.linear());
      current_state_snapshot = *current_state_info;
      goal_state_snapshot = *goal_state_info;
    }

    if (global_handler->time == 0.) {
      traj_interpolator->Reset(
          0., current_state_snapshot.joint_positions,
          goal_state_snapshot.joint_positions, policy_rate, traj_rate,
          global_handler->traj_interpolator_time_fraction);
    }
    global_handler->time += period.toSec();
    traj_interpolator->GetNextStep(global_handler->time, desired_q);
    tau_d_array = controller->Step(robot_state, desired_q);

    state_publisher->UpdateNewState(robot_state, &model);

    std::array<double, 7> tau_d_rate_limited = franka::limitRate(
        franka::kMaxTorqueRate, tau_d_array, robot_state.tau_J_d);
    control_utils::TorqueSafetyGuardFn(tau_d_rate_limited,
                                       global_handler->min_torque,
                                       global_handler->max_torque);
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    // global_handler->logger->debug("{0} microseconds" , time.count());

    return tau_d_rate_limited;
  };
}

} // namespace control_callbacks

#endif // UTILS_CONTROL_CALLBACKS_TORQUE_CALLBACK_H_
