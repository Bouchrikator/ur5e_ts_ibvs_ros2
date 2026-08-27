#!/usr/bin/env python3
"""Publish default joint states for gripper joints not managed by ros2_control."""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class GripperJointStatePublisher(Node):
    def __init__(self):
        super().__init__('gripper_joint_state_publisher')
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.1, self._publish)
        self.joint_names = [
            'gripper_finger1_joint',
            'gripper_finger2_joint',
            'gripper_finger1_inner_knuckle_joint',
            'gripper_finger2_inner_knuckle_joint',
            'gripper_finger1_finger_tip_joint',
            'gripper_finger2_finger_tip_joint',
        ]

    def _publish(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [0.0] * len(self.joint_names)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = GripperJointStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
