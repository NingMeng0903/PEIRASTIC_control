// Copyright 2023 Yifeng Zhu

#include "controllers/base_controller.h"

#ifndef DEOXYS_FRANKA_INTERFACE_INCLUDE_CONTROLLERS_CARTESIAN_VELOCITY_H_
#define DEOXYS_FRANKA_INTERFACE_INCLUDE_CONTROLLERS_CARTESIAN_VELOCITY_H_

namespace controller {
class CartesianVelocityController : public BaseController {
protected:
  FrankaCartesianVelocityControllerMessage control_msg_;

  double speed_factor_{1.0};
  bool use_pose_tracking_{false};
  double command_acceleration_limit_{1.0};
  double damping_gain_{0.0};
  Eigen::Vector3d translation_kp_{Eigen::Vector3d::Constant(2.5)};
  Eigen::Vector3d rotation_kp_{Eigen::Vector3d::Constant(1.0)};
  bool has_reference_goal_{false};
  Eigen::Vector3d reference_goal_position_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond reference_goal_quaternion_{Eigen::Quaterniond::Identity()};

public:
  CartesianVelocityController();
  CartesianVelocityController(franka::Model &model);

  ~CartesianVelocityController();

  bool ParseMessage(const FrankaControlMessage &msg);

  void ComputeGoal(const std::shared_ptr<StateInfo> &state_info,
                   std::shared_ptr<StateInfo> &goal_info);

  std::array<double, 6> Step(const franka::RobotState &,
                             const Eigen::Vector3d &,
                             const Eigen::Vector3d &);
  void EstimateVelocities(const franka::RobotState &robot_state,
                          std::shared_ptr<StateInfo> &);
  bool UsesPoseTracking() const;
  const Eigen::Vector3d &GetTranslationKp() const;
  const Eigen::Vector3d &GetRotationKp() const;
  double GetDampingGain() const;
  double GetCommandAccelerationLimit() const;
};

} // namespace controller

#endif // DEOXYS_FRANKA_INTERFACE_INCLUDE_CONTROLLERS_CARTESIAN_VELOCITY_H_



