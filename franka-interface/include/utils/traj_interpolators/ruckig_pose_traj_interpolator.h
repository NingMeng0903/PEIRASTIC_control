// Copyright 2026 PEIRASTIC

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>

#include <Eigen/Dense>

#include <ruckig/ruckig.hpp>

#include "base_traj_interpolator.h"

#ifndef DEOXYS_FRANKA_INTERFACE_INCLUDE_UTILS_TRAJ_INTERPOLATORS_RUCKIG_POSE_TRAJ_INTERPOLATOR_H_
#define DEOXYS_FRANKA_INTERFACE_INCLUDE_UTILS_TRAJ_INTERPOLATORS_RUCKIG_POSE_TRAJ_INTERPOLATOR_H_

namespace traj_utils {
class RuckigPoseTrajInterpolator : public BaseTrajInterpolator {
private:
  static constexpr size_t kDofs = 6;

  std::unique_ptr<ruckig::Ruckig<kDofs>> otg_;
  ruckig::InputParameter<kDofs> input_;
  ruckig::OutputParameter<kDofs> output_;
  std::mutex state_mutex_;

  Eigen::Vector3d goal_position_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond goal_quaternion_{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d last_position_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond last_quaternion_{Eigen::Quaterniond::Identity()};
  std::array<double, kDofs> last_velocity_{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  std::array<double, kDofs> last_acceleration_{
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  bool has_motion_state_{false};

  double dt_{0.002};
  double last_time_{0.0};
  double max_velocity_{1.5};
  double max_acceleration_{6.0};
  double max_jerk_{20.0};
  bool initialized_{false};

  static Eigen::Vector3d QuaternionToRotationVector(
      const Eigen::Quaterniond &quaternion) {
    const Eigen::Quaterniond normalized_quaternion = quaternion.normalized();
    const Eigen::AngleAxisd axis_angle(normalized_quaternion);
    if (!std::isfinite(axis_angle.angle()) || axis_angle.angle() < 1e-12) {
      return Eigen::Vector3d::Zero();
    }

    return axis_angle.angle() * axis_angle.axis();
  }

  static Eigen::Quaterniond RotationVectorToQuaternion(
      const Eigen::Vector3d &rotation_vector) {
    const double angle = rotation_vector.norm();
    if (angle < 1e-12) {
      return Eigen::Quaterniond::Identity();
    }

    return Eigen::Quaterniond(
        Eigen::AngleAxisd(angle, rotation_vector / angle));
  }

  static Eigen::Quaterniond CanonicalizeQuaternion(
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

  static bool SameGoalQuaternion(const Eigen::Quaterniond &lhs,
                                 const Eigen::Quaterniond &rhs) {
    return std::abs(lhs.normalized().coeffs().dot(rhs.normalized().coeffs())) >
           1.0 - 1e-9;
  }

public:
  inline RuckigPoseTrajInterpolator() = default;
  inline ~RuckigPoseTrajInterpolator() override = default;

  inline void SyncCartesianCommandState(
      const Eigen::Vector3d &p_cmd, const Eigen::Quaterniond &q_cmd,
      const std::array<double, 6> &velocity,
      const std::array<double, 6> &acceleration) override {
    std::lock_guard<std::mutex> lock(state_mutex_);

    last_position_ = p_cmd;
    if (has_motion_state_ || initialized_) {
      last_quaternion_ = CanonicalizeQuaternion(q_cmd, &last_quaternion_);
    } else {
      last_quaternion_ = CanonicalizeQuaternion(q_cmd);
    }
    if (!has_motion_state_) {
      last_velocity_ = velocity;
      last_acceleration_ = acceleration;
      has_motion_state_ = true;
    }

    if (!initialized_ || !otg_) {
      return;
    }

    // Keep the commanded pose aligned with libfranka while preserving
    // Ruckig's internal velocity / acceleration continuity.
    const Eigen::Vector3d current_rotation =
        QuaternionToRotationVector(last_quaternion_);
    input_.current_position = {last_position_[0], last_position_[1],
                               last_position_[2], current_rotation[0],
                               current_rotation[1], current_rotation[2]};
    output_.new_position = input_.current_position;
  }

  inline void Reset(const double &time_sec, const Eigen::Vector3d &p_start,
                    const Eigen::Quaterniond &q_start,
                    const Eigen::Vector3d &p_goal,
                    const Eigen::Quaterniond &q_goal, const int &policy_rate,
                    const int &rate,
                    const double &traj_interpolator_time_fraction) override {
    std::lock_guard<std::mutex> lock(state_mutex_);
    const bool initialized_before = initialized_;
    const Eigen::Vector3d prev_goal_position = goal_position_;
    const Eigen::Quaterniond prev_goal_quaternion = goal_quaternion_;

    Eigen::Quaterniond start_quaternion =
        initialized_before ? CanonicalizeQuaternion(q_start, &last_quaternion_)
                           : CanonicalizeQuaternion(q_start);
    Eigen::Quaterniond normalized_goal_quaternion =
        CanonicalizeQuaternion(q_goal, &start_quaternion);
    const bool same_goal_position = initialized_before &&
                                    (prev_goal_position - p_goal).norm() < 1e-9;
    const bool same_goal_quaternion =
        initialized_before &&
        SameGoalQuaternion(prev_goal_quaternion, normalized_goal_quaternion);

    if (initialized_before && same_goal_position && same_goal_quaternion) {
      return;
    }

    const double new_dt = 1.0 / static_cast<double>(rate);
    if (!otg_ || std::abs(dt_ - new_dt) > 1e-12) {
      dt_ = new_dt;
      otg_ = std::make_unique<ruckig::Ruckig<kDofs>>(dt_);
      input_ = ruckig::InputParameter<kDofs>();
      output_ = ruckig::OutputParameter<kDofs>();
      initialized_ = false;
      has_motion_state_ = false;
      last_velocity_ = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
      last_acceleration_ = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    }
    last_time_ = time_sec;
    initialized_ = true;

    const Eigen::Vector3d reset_position =
        (initialized_before && has_motion_state_) ? last_position_ : p_start;
    const Eigen::Quaterniond reset_quaternion =
        (initialized_before && has_motion_state_) ? last_quaternion_
                                                  : start_quaternion;
    goal_quaternion_ = CanonicalizeQuaternion(normalized_goal_quaternion,
                                              &reset_quaternion);
    const Eigen::Vector3d start_rotation =
        QuaternionToRotationVector(reset_quaternion);
    const Eigen::Vector3d goal_rotation =
        QuaternionToRotationVector(goal_quaternion_);

    input_.current_position = {reset_position[0], reset_position[1], reset_position[2],
                               start_rotation[0], start_rotation[1],
                               start_rotation[2]};
    input_.target_position = {p_goal[0], p_goal[1], p_goal[2], goal_rotation[0],
                              goal_rotation[1], goal_rotation[2]};
    input_.current_velocity =
        (initialized_before && has_motion_state_)
            ? last_velocity_
            : std::array<double, kDofs>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    input_.target_velocity = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    input_.current_acceleration =
        (initialized_before && has_motion_state_)
            ? last_acceleration_
            : std::array<double, kDofs>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    input_.target_acceleration = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    input_.max_velocity = {max_velocity_, max_velocity_, max_velocity_,
                           max_velocity_, max_velocity_, max_velocity_};
    input_.max_acceleration = {max_acceleration_, max_acceleration_,
                               max_acceleration_, max_acceleration_,
                               max_acceleration_, max_acceleration_};
    input_.max_jerk = {max_jerk_, max_jerk_, max_jerk_,
                       max_jerk_, max_jerk_, max_jerk_};
    input_.synchronization = ruckig::Synchronization::Time;

    const double minimum_duration =
        std::max(1.0 / static_cast<double>(policy_rate) *
                     traj_interpolator_time_fraction,
                 dt_);
    input_.minimum_duration = minimum_duration;

    goal_position_ = p_goal;
    last_position_ = reset_position;
    last_quaternion_ = reset_quaternion;
  }

  inline void GetNextStep(const double &time_sec, Eigen::Vector3d &p_t,
                          Eigen::Quaterniond &q_t) override {
    Eigen::Vector3d twist_trans_t = Eigen::Vector3d::Zero();
    Eigen::Vector3d twist_rot_t = Eigen::Vector3d::Zero();
    GetNextStep(time_sec, p_t, q_t, twist_trans_t, twist_rot_t);
  }

  inline void GetNextStep(const double &time_sec, Eigen::Vector3d &p_t,
                          Eigen::Quaterniond &q_t,
                          Eigen::Vector3d &twist_trans_t,
                          Eigen::Vector3d &twist_rot_t) override {
    std::lock_guard<std::mutex> lock(state_mutex_);
    Advance(time_sec);
    p_t = last_position_;
    q_t = last_quaternion_;
    twist_trans_t << last_velocity_[0], last_velocity_[1], last_velocity_[2];
    twist_rot_t << last_velocity_[3], last_velocity_[4], last_velocity_[5];
  }

  inline void
  SetConfig(const FrankaTrajInterpolatorConfig &traj_interpolator_config)
      override {
    if (traj_interpolator_config.max_velocity() > 0.0) {
      max_velocity_ = traj_interpolator_config.max_velocity();
    }
    if (traj_interpolator_config.max_acceleration() > 0.0) {
      max_acceleration_ = traj_interpolator_config.max_acceleration();
    }
    if (traj_interpolator_config.max_jerk() > 0.0) {
      max_jerk_ = traj_interpolator_config.max_jerk();
    }
  }

private:
  inline void Advance(const double &time_sec) {
    if (!initialized_ || !otg_) {
      return;
    }

    if (last_time_ + dt_ > time_sec) {
      return;
    }

    const auto result = otg_->update(input_, output_);
    if (result < 0) {
      last_position_ = goal_position_;
      last_quaternion_ = goal_quaternion_;
      last_velocity_ = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
      last_acceleration_ = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
      has_motion_state_ = false;
      last_time_ = time_sec;
      return;
    }

    last_position_ << output_.new_position[0], output_.new_position[1],
        output_.new_position[2];
    const Eigen::Vector3d rotation_vector(output_.new_position[3],
                                          output_.new_position[4],
                                          output_.new_position[5]);
    last_quaternion_ =
        CanonicalizeQuaternion(RotationVectorToQuaternion(rotation_vector),
                               &last_quaternion_);
    last_velocity_ = output_.new_velocity;
    last_acceleration_ = output_.new_acceleration;
    has_motion_state_ = true;

    if (result == ruckig::Result::Finished) {
      last_position_ = goal_position_;
      last_quaternion_ = goal_quaternion_;
      last_velocity_ = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
      last_acceleration_ = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
      has_motion_state_ = false;
    } else {
      output_.pass_to_input(input_);
    }

    last_time_ = time_sec;
  }
};
} // namespace traj_utils

#endif // DEOXYS_FRANKA_INTERFACE_INCLUDE_UTILS_TRAJ_INTERPOLATORS_RUCKIG_POSE_TRAJ_INTERPOLATOR_H_
