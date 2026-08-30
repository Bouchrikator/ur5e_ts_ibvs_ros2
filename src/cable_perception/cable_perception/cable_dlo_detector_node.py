"""Marker-free cable observation from the eye-to-hand camera.

Publishes the same ``/cable/observed_markers`` contract as the coloured-marker
tracker, but the points come from the detected centreline instead of fiducials:

    image -> colour mask -> skeleton -> branches -> table plane -> chain of
    fixed-length segments -> merge across occlusions -> resample at s/L

so the cable carries nothing but itself. Marker identity, which the coloured
spheres used to supply, comes from arc length measured from the clamped end.

Concerns: ROS transport and frame handling. The detection maths lives in
``dlo_detection`` and the ray/plane geometry in ``table_projection``; both stay
unit tested without a camera.
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time

import tf2_ros
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image

from cable_msgs.msg import CableMarker, CableMarkerArray

from cable_perception import dlo_detection as dlo
from cable_perception.table_projection import (
    project_pixels_to_plane, quaternion_to_matrix)


class CableDloDetectorNode(Node):
    def __init__(self):
        super().__init__("cable_dlo_detector_node")

        self.declare_parameter("image_topic", "/overview/image")
        self.declare_parameter("camera_info_topic", "/overview/camera_info")
        self.declare_parameter("observed_topic", "/cable/observed_markers")
        self.declare_parameter("camera_frame", "overview_camera_optical_frame")
        self.declare_parameter("reference_frame", "cable_fixture_frame")
        self.declare_parameter("marker_s_over_l",
                               [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0])
        self.declare_parameter("cable_length_m", 0.7)
        self.declare_parameter("n_segments", 30)
        # Segmentation is the one scene-dependent step: the paper is explicit
        # that any conservative method works, so it stays a parameter. Default
        # window is "dark and unsaturated", matching the grey cable used here.
        self.declare_parameter("cable_hsv_lower", [0.0, 0.0, 0.0])
        self.declare_parameter("cable_hsv_upper", [180.0, 90.0, 95.0])
        self.declare_parameter("min_branch_pixels", 5)
        self.declare_parameter("length_tolerance", 0.35)
        self.declare_parameter("position_std_m", 0.004)
        self.declare_parameter("plane_normal", [0.0, 0.0, 1.0])
        self.declare_parameter("plane_offset_m", 0.0)

        self.camera_frame = self.get_parameter("camera_frame").value
        self.reference_frame = self.get_parameter("reference_frame").value
        self.s_over_l = list(self.get_parameter("marker_s_over_l").value)
        self.cable_length = float(self.get_parameter("cable_length_m").value)
        n_segments = int(self.get_parameter("n_segments").value)
        if n_segments < 2:
            raise ValueError("n_segments must be >= 2")
        self.segment_length = self.cable_length / n_segments
        self.hsv_lower = list(self.get_parameter("cable_hsv_lower").value)
        self.hsv_upper = list(self.get_parameter("cable_hsv_upper").value)
        self.min_branch_pixels = int(self.get_parameter("min_branch_pixels").value)
        self.length_tolerance = float(self.get_parameter("length_tolerance").value)
        self.plane_normal = list(self.get_parameter("plane_normal").value)
        self.plane_offset = float(self.get_parameter("plane_offset_m").value)

        std = float(self.get_parameter("position_std_m").value)
        self._covariance = [std ** 2, 0.0, 0.0,
                            0.0, std ** 2, 0.0,
                            0.0, 0.0, std ** 2]

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.intrinsics = None

        self.pub = self.create_publisher(
            CableMarkerArray, self.get_parameter("observed_topic").value, 10)
        self.create_subscription(
            CameraInfo, self.get_parameter("camera_info_topic").value,
            self._on_camera_info, 10)
        self.create_subscription(
            Image, self.get_parameter("image_topic").value, self._on_image, 1)

        self.get_logger().info(
            f"cable DLO detector: {len(self.s_over_l)} samples along a "
            f"{self.cable_length:.3f} m chain of {n_segments} segments, "
            f"plane={self.reference_frame}, camera={self.camera_frame}")

    # ------------------------------------------------------------------
    def _on_camera_info(self, msg):
        k = msg.k
        self.intrinsics = (k[0], k[4], k[2], k[5])

    def _camera_pose(self):
        try:
            tfm = self.tf_buffer.lookup_transform(
                self.reference_frame, self.camera_frame, Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        t, q = tfm.transform.translation, tfm.transform.rotation
        return (quaternion_to_matrix(q.x, q.y, q.z, q.w),
                np.array([t.x, t.y, t.z]))

    def _publish(self, stamp, points):
        """``points`` is an (M, 3) array, or ``None`` when detection failed."""
        out = CableMarkerArray()
        out.header.stamp = stamp
        out.header.frame_id = self.reference_frame
        out.experiment_id = "camera_dlo"
        for index, s in enumerate(self.s_over_l):
            marker = CableMarker()
            marker.id = index
            marker.s_over_l = float(s)
            marker.valid = points is not None
            if marker.valid:
                marker.position.x = float(points[index][0])
                marker.position.y = float(points[index][1])
                marker.position.z = float(points[index][2])
                marker.covariance = self._covariance
                marker.confidence = 1.0
            out.markers.append(marker)
        self.pub.publish(out)

    def _metric_branches(self, mask, rotation, origin):
        """Skeleton branches projected onto the table plane, in metres."""
        branches = dlo.skeleton_branches(
            dlo.zhang_suen_thin(mask), self.min_branch_pixels)
        metric = []
        for branch in branches:
            pixels = branch[:, ::-1]  # (row, col) -> (u, v)
            points = project_pixels_to_plane(
                pixels, self.intrinsics, rotation, origin,
                plane_point=[0.0, 0.0, self.plane_offset],
                plane_normal=self.plane_normal)
            kept = np.array([p for p in points if p is not None])
            if len(kept) >= 2:
                metric.append(kept)
        return metric

    def _on_image(self, msg):
        if self.intrinsics is None:
            return
        pose = self._camera_pose()
        if pose is None:
            return
        rotation, origin = pose

        import cv2
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array(self.hsv_lower, np.uint8),
                           np.array(self.hsv_upper, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Fit fixed-length segments in metres, not pixels: the segment length is
        # then a real arc length and the chain inherits the cable's own scale.
        chains = [chain for branch in self._metric_branches(mask, rotation, origin)
                  for chain in dlo.fit_chain(branch, self.segment_length)]
        if not chains:
            self.get_logger().warn("no cable detected in the image",
                                   throttle_duration_sec=5.0)
            self._publish(msg.header.stamp, None)
            return

        chain = dlo.merge_chains(chains, self.segment_length)
        chain = dlo.orient_chain(chain, np.zeros(3))  # origin of the fixture frame
        length = float(np.sum(np.linalg.norm(np.diff(chain, axis=0), axis=1)))
        if abs(length - self.cable_length) > self.length_tolerance * self.cable_length:
            # A chain of the wrong length is a detection failure, not a shape:
            # resampling it would silently mislabel every arc-length identity.
            self.get_logger().warn(
                f"detected chain is {length:.3f} m, expected "
                f"{self.cable_length:.3f} m", throttle_duration_sec=5.0)
            self._publish(msg.header.stamp, None)
            return

        self._publish(msg.header.stamp, dlo.resample_chain(chain, self.s_over_l))


def main():
    rclpy.init()
    node = CableDloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
