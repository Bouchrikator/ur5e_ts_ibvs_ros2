#!/usr/bin/env python3
"""Move the UR5 to a viewing pose where the wrist camera sees the table.

Publishes a JointTrajectory to the active joint trajectory controller
(scaled_joint_trajectory_controller on the real UR driver,
joint_trajectory_controller in the Gazebo sim).

Usage: ros2 run ibvs_control move_to_view_pose.py [pose_name] [--sim]
"""

import sys
import time

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Poses optimised for manipulability, workspace coverage and a downward camera.
POSES = {
    "default": [0.0, -1.4, 1.4, -1.57, -1.57, 0.0],
    # Eye-to-hand: wrist_3 +90 deg turns the green marker toward the overview camera.
    "eth": [0.0, -1.4, 1.4, -1.57, -1.57, 1.57],
    "view": [0.0, -1.8, 2.0, -1.57, -1.57, 0.0],
    "high": [0.0, -1.4, 1.4, -1.57, -1.57, 3.14],
    "low": [0.0, -2.0, 2.2, -1.8, -1.57, 3.14],
    "left": [-0.5, -1.8, 1.8, -1.57, -1.57, 3.14],
    "right": [0.5, -1.8, 1.8, -1.57, -1.57, 3.14],
    "table": [0.0, -0.45, 1.85, -3.0, -1.57, 0.0],
}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pose_name = args[0] if args else "default"
    if pose_name not in POSES:
        print(f"Unknown pose '{pose_name}'. Available: {', '.join(POSES)}")
        sys.exit(1)

    # Real driver uses the scaled controller; the Gazebo sim uses the plain one.
    sim = "--sim" in sys.argv
    controller = ("/joint_trajectory_controller/joint_trajectory" if sim else
                  "/scaled_joint_trajectory_controller/joint_trajectory")

    rclpy.init()
    node = Node("move_to_view_pose")
    pub = node.create_publisher(JointTrajectory, controller, 10)

    # Wait for the controller to actually subscribe: from a fresh container
    # DDS discovery can take several seconds and an early publish is lost.
    deadline = time.time() + 15.0
    while pub.get_subscription_count() == 0 and time.time() < deadline:
        time.sleep(0.2)
    if pub.get_subscription_count() == 0:
        node.get_logger().error(
            f"No subscriber on {controller} — is the sim (or driver) running?")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    traj = JointTrajectory()
    traj.joint_names = JOINT_NAMES
    point = JointTrajectoryPoint()
    point.positions = [float(v) for v in POSES[pose_name]]
    point.time_from_start.sec = 5
    traj.points.append(point)

    pub.publish(traj)
    node.get_logger().info(f"Sent '{pose_name}' pose to {controller} (5 s motion)")
    time.sleep(1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
