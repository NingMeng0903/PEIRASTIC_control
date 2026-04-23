# libfranka 0.9.0 → 0.10.0 映射与适配说明

基于 PEIRASTIC_control 当前使用的 API，对 0.9 与 0.10 的差异做逐项对照。

---

## 1. robot_state.h

| 项目 | 0.9 | 0.10 | 影响 |
|------|-----|------|------|
| RobotState 结构体 | 相同 | 相同 | 无 |
| RobotMode 枚举 | 相同 | 相同 | 无 |
| 新增 | - | `operator<<(ostream, RobotMode)` | 无影响 |

**结论**：无需改动，`LoadRobotStateToMsg` 可直接复用。

---

## 2. errors.h

| 项目 | 0.9 | 0.10 | 影响 |
|------|-----|------|------|
| franka::Errors 结构体 | 相同 | 相同 | 无 |

**结论**：`LoadErrorStateToMsg` 无需修改。

---

## 3. rate_limiting.h（重要）

| 常量/参数 | 0.9 (Panda) | 0.10 (FR3) | 说明 |
|-----------|-------------|------------|------|
| `kTolNumberPacketsLost` | 3.0 | 0.0 | FR3 假设无丢包 |
| `kMaxTorqueRate` | [1000]×7 | [1000]×7 | 相同 |
| `kMaxJointJerk` | [7500,3750,5000,...] | [5000]×7 | FR3 统一为 5000 |
| `kMaxJointAcceleration` | [15,7.5,10,...] | [10]×7 | FR3 统一为 10 |
| `kMaxJointVelocity` | 与 jerk/acc 相关 | [2,1,1.5,1.25,3,1.5,3] | FR3 新限速 |
| `kMaxTranslationalJerk` | 6500 | 4500 | 降低 |
| `kMaxTranslationalAcceleration` | 13 | 9 | 降低 |
| `kMaxTranslationalVelocity` | 2.0 | 3.0 | 提高 |
| `kMaxRotationalJerk` | 12500 | 8500 | 降低 |
| `kMaxRotationalAcceleration` | 25 | 17 | 降低 |
| `kMaxElbowVelocity` | ~2.175 | 1.5 | 降低 |

**API**：`limitRate(max_derivatives, commanded, last_commanded)` 签名一致。

**结论**：`torque_callback.h` 中 `franka::limitRate(franka::kMaxTorqueRate, ...)` 可直接用于 0.10，无需修改。`kMaxTorqueRate` 在 0.10 中仍为 1000。

---

## 4. robot.h

| 项目 | 0.9 | 0.10 | 影响 |
|------|-----|------|------|
| `Robot::control()` 等 `limit_rate` 默认值 | `true` | `false` | 0.10 默认关闭 rate limiting |
| `getVirtualWall()` | 存在 | 已移除 | PEIRASTIC 未使用 |
| `setFilters()` | 已弃用 | 已移除 | PEIRASTIC 未使用 |

**结论**：若希望沿用机器人侧的 rate limiting，在调用 `robot.control(...)` 时显式传入 `limit_rate=true`。PEIRASTIC 使用 `limitRate()` 做软件侧限速，行为与 0.9 类似。

---

## 5. 其他头文件（model.h, gripper.h, duration.h 等）

未发现与 franka-interface 相关的 breaking 变更。

---

## 6. PEIRASTIC_control 需检查的调用点

| 文件 | 调用 | 0.10 兼容性 |
|------|------|-------------|
| `torque_callback.h` | `franka::limitRate(kMaxTorqueRate, ...)` | 是 |
| `robot_utils.cpp` | `LoadErrorStateToMsg`, `LoadRobotStateToMsg` | 是 |
| `franka_control_node.cpp` | `franka::Robot`, `franka::Model` | 是 |
| `gripper_control_node.cpp` | `franka::Gripper` | 是 |
| `common_utils.cpp` | `setDefaultBehavior(robot)` | 需确认是否依赖已移除 API |

---

## 7. 升级步骤建议

1. 将 `PEIRASTIC_control/libfranka` 替换为 libfranka 0.10 源码
2. 重新编译，处理可能的 include 路径或小改动
3. 在 `robot.control()` 等调用中按需显式设置 `limit_rate=true`
4. 在 FR3 上做空载与轻载测试

---

## 8. 0.10 源码获取

```bash
git clone --depth 1 --branch 0.10.0 https://github.com/frankarobotics/libfranka.git
# 将 libfranka 目录内容替换 PEIRASTIC_control/libfranka
```
