"""Cable reference projector: desired gripper pose -> IBVS feature target.

Step 10 of the plan. The outer cable loop says WHERE the gripper should go; the
inner eye-to-hand TS-IBVS loop only understands image features. This node is
the adapter between the two, and it is deliberately the exact same output
contract as the cube-based ``eth_reference_node`` — only the source of the goal
point differs, so the inner loop is reused unchanged.

It is also the reason both controllers can run at once without fighting: the
outer loop reaches Servo only through the inner loop, never directly.
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time

import tf2_ros
from geometry_msgs.msg import PoseStamped

from ibvs_msgs.msg import FeatureTarget


def quaternion_to_matrix(x, y, z, w):
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def project_marker_square(goal_camera, marker_size, fx, fy, cx, cy):
    """Corners of a fronto-parallel square centred on a camera-frame point.

    Corner order TL, TR, BR, BL to match the detector's ordering.
    Returns ``(features_normalized, corner_pixels, centroid_uv)``.
    """
    depth = float(goal_camera[2])
    if depth <= 0.0:
        raise ValueError("the goal point must be in front of the camera")

    x_star = float(goal_camera[0]) / depth
    y_star = float(goal_camera[1]) / depth
    half = (marker_size / 2.0) / depth

    xs = (x_star - half, x_star + half, x_star + half, x_star - half)
    ys = (y_star - half, y_star - half, y_star + half, y_star + half)

    normalized, pixels = [], []
    for x, y in zip(xs, ys):
        normalized.extend([x, y])
        pixels.extend([x * fx + cx, y * fy + cy])
    return normalized, pixels, (x_star * fx + cx, y_star * fy + cy)


class CableReferenceProjectorNode(Node):
    def __init__(self):
        super().__init__("cable_reference_projector_node")

        self.declare_parameter("desired_pose_topic", "/cable/desired_gripper_pose")
        self.declare_parameter("marker_topic", "/marker_detector/feature_target")
        self.declare_parameter("feature_target_topic",
                               "/cable/desired_feature_target")
        self.declare_parameter("camera_frame", "overview_optical_frame")
        self.declare_parameter("marker_size", 0.05)
        self.declare_parameter("min_depth_m", 0.05)

        self.camera_frame = self.get_parameter("camera_frame").value
        self.marker_size = float(self.get_parameter("marker_size").value)
        self.min_depth = float(self.get_parameter("min_depth_m").value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._marker = None

        self.pub = self.create_publisher(
            FeatureTarget, self.get_parameter("feature_target_topic").value, 10)
        self.create_subscription(
            FeatureTarget, self.get_parameter("marker_topic").value,
            self._on_marker, 10)
        self.create_subscription(
            PoseStamped, self.get_parameter("desired_pose_topic").value,
            self._on_desired_pose, 10)

        self.get_logger().info(
            f"cable reference projector: {self.marker_size * 1e3:.0f} mm marker, "
            f"camera={self.camera_frame}")

    # ------------------------------------------------------------------
    def _on_marker(self, msg):
        # Only kept for the camera intrinsics and image size, which the inner
        # loop expects to be echoed back in the reference.
        if msg.fx > 0.0:
            self._marker = msg

    def _to_camera_frame(self, pose):
        try:
            tfm = self.tf_buffer.lookup_transform(
                self.camera_frame, pose.header.frame_id, Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        t, q = tfm.transform.translation, tfm.transform.rotation
        rotation = quaternion_to_matrix(q.x, q.y, q.z, q.w)
        point = np.array([pose.pose.position.x, pose.pose.position.y,
                          pose.pose.position.z])
        return rotation @ point + np.array([t.x, t.y, t.z])

    def _on_desired_pose(self, msg):
        if self._marker is None:
            self.get_logger().warn(
                "no gripper marker seen yet; camera intrinsics unknown",
                throttle_duration_sec=5.0)
            return

        goal = self._to_camera_frame(msg)
        if goal is None:
            self.get_logger().warn(
                f"no transform {msg.header.frame_id} -> {self.camera_frame}",
                throttle_duration_sec=5.0)
            return

        goal[2] = max(goal[2], self.min_depth)
        source = self._marker
        normalized, pixels, centroid = project_marker_square(
            goal, self.marker_size, source.fx, source.fy, source.cx, source.cy)

        out = FeatureTarget()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.camera_frame
        out.detected = True
        out.image_width = source.image_width
        out.image_height = source.image_height
        out.features_normalized = normalized
        out.corner_pixels = pixels
        out.centroid_u, out.centroid_v = centroid
        out.area = 0.0
        out.pose_valid = True
        out.z_pnp = float(goal[2])
        out.depth_valid = False
        out.z_depth = 0.0
        out.fx, out.fy = source.fx, source.fy
        out.cx, out.cy = source.cx, source.cy
        self.pub.publish(out)


def main():
    rclpy.init()
    node = CableReferenceProjectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
