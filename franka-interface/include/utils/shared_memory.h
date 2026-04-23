// Copyright 2022 Yifeng Zhu

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <spdlog/spdlog.h>

#include "controllers/base_controller.h"
#include "utils/control_utils.h"
#include "utils/traj_interpolators/base_traj_interpolator.h"

#ifndef DEOXYS_FRANKA_INTERFACE_INCLUDE_UTILS_SHARED_MEMORY_H_
#define DEOXYS_FRANKA_INTERFACE_INCLUDE_UTILS_SHARED_MEMORY_H_

struct SharedMemory {
  std::atomic_bool running{true};      // controlling control callback
  std::atomic_bool termination{false}; // controlling main loop
  std::mutex control_mutex;
  double time = 0.0;

  std::shared_ptr<controller::BaseController> controller_ptr;
  std::shared_ptr<traj_utils::BaseTrajInterpolator> traj_interpolator_ptr;
  std::shared_ptr<spdlog::logger> logger;
  std::mutex state_info_mutex;

  // for torque control
  std::atomic<double> max_torque;
  std::atomic<double> min_torque;

  // for velocity control
  std::atomic<double> max_trans_speed;
  std::atomic<double> min_trans_speed;
  std::atomic<double> max_rot_speed;
  std::atomic<double> min_rot_speed;
  std::array<double, 6> previous_cartesian_velocity_command{
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  bool has_previous_cartesian_velocity_command{false};

  // for trajectory interpolation
  double traj_interpolator_time_fraction = 1.0;

  std::atomic_int no_msg_counter;
  std::atomic_bool start{
      false}; // indicate if a message is received and start controlling

  inline void ResetCartesianVelocityCommandLocked() {
    previous_cartesian_velocity_command = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    has_previous_cartesian_velocity_command = false;
  }

  inline void ResetRuntimeStateLocked() {
    time = 0.0;
    ResetCartesianVelocityCommandLocked();
  }
};

#endif // DEOXYS_FRANKA_INTERFACE_INCLUDE_UTILS_SHARED_MEMORY_H_
