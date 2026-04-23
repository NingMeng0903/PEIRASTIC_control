"""
SpaceNav teleoperation publisher.
Integrates SpaceNav twist input into an absolute EEF pose and publishes it
as geometry_msgs/PoseStamped on /panda_target_pose at PUBLISH_RATE Hz.

button[1] rising edge: desired pose snaps back to INIT_POS / INIT_QUAT.

依赖: ROS + spacenav_node（见 PEIRASTIC 文档）；启动前确保 /spacenav/twist、/spacenav/joy 有数据。
"""

import threading
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Joy

# Initial pose: position z
INIT_POS = np.array([0.450, -0.000, 0.430])

# Reference quat
INIT_QUAT = np.array([-0.8860310316085815, 0.46312281489372253, 0.015188909135758877, 0.015346580184996128])


PUBLISH_RATE = 100.0  # Hz

LIN_SCALE = 0.12
ROT_SCALE = 0.3
LIN_DEADZONE = 0.01
ROT_DEADZONE = 0.01

_lock = threading.Lock()
_sn_lin = np.zeros(3)
_sn_ang = np.zeros(3)
_reset_requested = False
_btn1_prev = 0


def _spacenav_cb(msg: Twist):
    global _sn_lin, _sn_ang
    lin = np.array([msg.linear.x, msg.linear.y, msg.linear.z])
    ang = np.array([msg.angular.x, msg.angular.y, msg.angular.z])
    if np.linalg.norm(lin) < LIN_DEADZONE:
        lin = np.zeros(3)
    if np.linalg.norm(ang) < ROT_DEADZONE:
        ang = np.zeros(3)
    with _lock:
        _sn_lin = lin
        _sn_ang = ang


def _joy_cb(msg: Joy):
    global _reset_requested, _btn1_prev
    if len(msg.buttons) < 2:
        return
    cur = msg.buttons[1]
    with _lock:
        if cur == 1 and _btn1_prev == 0:
            _reset_requested = True
        _btn1_prev = cur


def integrate_rotation(quat_xyzw, omega, dt):
    angle = np.linalg.norm(omega) * dt
    if angle < 1e-9:
        return quat_xyzw
    axis = omega / np.linalg.norm(omega)
    q = (R_scipy.from_quat(quat_xyzw) * R_scipy.from_rotvec(axis * angle)).as_quat()
    return -q if q[3] < 0 else q


def make_pose_stamped(pos, quat):
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "world"
    msg.pose.position.x = pos[0]
    msg.pose.position.y = pos[1]
    msg.pose.position.z = pos[2]
    msg.pose.orientation.x = quat[0]
    msg.pose.orientation.y = quat[1]
    msg.pose.orientation.z = quat[2]
    msg.pose.orientation.w = quat[3]
    return msg


def main():
    global _reset_requested

    rospy.init_node("spacenav_teleop_publisher", anonymous=False)
    rospy.Subscriber("/spacenav/twist", Twist, _spacenav_cb, queue_size=1)
    rospy.Subscriber("/spacenav/joy", Joy, _joy_cb, queue_size=1)

    pub = rospy.Publisher("/panda_target_pose", PoseStamped, queue_size=1)

    desired_pos = INIT_POS.copy()
    desired_quat = INIT_QUAT.copy()

    rospy.loginfo("[spacenav_pub] init pos  = [%.4f, %.4f, %.4f]",
                  desired_pos[0], desired_pos[1], desired_pos[2])
    rospy.loginfo("[spacenav_pub] init quat = [x=%.4f, y=%.4f, z=%.4f, w=%.4f]",
                  desired_quat[0], desired_quat[1], desired_quat[2], desired_quat[3])

    rate = rospy.Rate(PUBLISH_RATE)
    dt = 1.0 / PUBLISH_RATE

    while not rospy.is_shutdown():
        with _lock:
            sn_lin = _sn_lin.copy()
            sn_ang = _sn_ang.copy()
            do_reset = _reset_requested
            if do_reset:
                _reset_requested = False

        if do_reset:
            desired_pos = INIT_POS.copy()
            desired_quat = INIT_QUAT.copy()
            rospy.loginfo("[spacenav_pub] reset to init pose.")
        else:
            desired_pos += sn_lin * LIN_SCALE * dt
            omega = np.array([sn_ang[0], -sn_ang[1], -sn_ang[2]]) * ROT_SCALE
            desired_quat = integrate_rotation(desired_quat, omega, dt)

        pub.publish(make_pose_stamped(desired_pos, desired_quat))
        rate.sleep()


if __name__ == "__main__":
    main()
