"""Eye-to-hand camera tracker for the cable markers.

Detects the coloured markers in the overview image and reconstructs their 3-D
position by intersecting each optical ray with the known table plane, which is
what makes a monocular reconstruction well posed here.

Publishes the SAME contract as the synthetic source, so the estimator and the
controller cannot tell which observation source is running.

Concerns: image -> pixel detections -> ROS. The ray/plane geometry lives in
``table_projection``; the colour classes live in the marker palette config.
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time

import tf2_ros
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image

from cable_msgs.msg import CableMarker, CableMarkerArray

from cable_perception.table_projection import (
    project_pixels_to_plane, quaternion_to_matrix)


def detect_colour_blob(hsv, lower, upper, min_area_px):
    """Centroid of the largest blob matching an HSV range, or None."""
    import cv2

    mask = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area_px:
        return None
    moments = cv2.moments(largest)
    if moments["m00"] == 0.0:
        return None
    return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])


class CableMarkerTrackerNode(Node):
    def __init__(self):
        super().__init__("cable_marker_tracker_node")

        self.declare_parameter("image_topic", "/overview/image")
        self.declare_parameter("camera_info_topic", "/overview/camera_info")
        self.declare_parameter("observed_topic", "/cable/observed_markers")
        self.declare_parameter("camera_frame", "overview_camera_optical_frame")
        self.declare_parameter("reference_frame", "cable_fixture_frame")
        self.declare_parameter("marker_s_over_l",
                               [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0])
        # One HSV window per marker: [h_lo,s_lo,v_lo, h_hi,s_hi,v_hi].
        # Distinct colours are what give the markers a persistent identity.
        self.declare_parameter("marker_hsv_ranges", [
            0.0, 120.0, 80.0, 10.0, 255.0, 255.0,      # red
            11.0, 120.0, 80.0, 22.0, 255.0, 255.0,     # orange
            23.0, 120.0, 80.0, 34.0, 255.0, 255.0,     # yellow
            40.0, 100.0, 60.0, 80.0, 255.0, 255.0,     # green
            85.0, 120.0, 60.0, 100.0, 255.0, 255.0,    # cyan
            105.0, 120.0, 60.0, 125.0, 255.0, 255.0,   # blue
            130.0, 100.0, 60.0, 160.0, 255.0, 255.0,   # magenta
        ])
        self.declare_parameter("min_blob_area_px", 12.0)
        self.declare_parameter("position_std_m", 0.004)
        self.declare_parameter("plane_normal", [0.0, 0.0, 1.0])
        self.declare_parameter("plane_offset_m", 0.0)

        self.camera_frame = self.get_parameter("camera_frame").value
        self.reference_frame = self.get_parameter("reference_frame").value
        self.s_over_l = list(self.get_parameter("marker_s_over_l").value)
        self.min_area = float(self.get_parameter("min_blob_area_px").value)
        self.plane_normal = list(self.get_parameter("plane_normal").value)
        self.plane_offset = float(self.get_parameter("plane_offset_m").value)

        flat = list(self.get_parameter("marker_hsv_ranges").value)
        if len(flat) != 6 * len(self.s_over_l):
            raise ValueError(
                f"marker_hsv_ranges needs 6 values per marker: expected "
                f"{6 * len(self.s_over_l)}, got {len(flat)}")
        self.hsv_ranges = [(flat[i:i + 3], flat[i + 3:i + 6])
                           for i in range(0, len(flat), 6)]

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
            f"cable marker tracker: {len(self.s_over_l)} markers, "
            f"plane={self.reference_frame}, camera={self.camera_frame}")

    # ------------------------------------------------------------------
    def _on_camera_info(self, msg):
        k = msg.k
        self.intrinsics = (k[0], k[4], k[2], k[5])  # fx, fy, cx, cy

    def _camera_pose(self):
        """Camera rotation and origin expressed in the reference (plane) frame."""
        try:
            tfm = self.tf_buffer.lookup_transform(
                self.reference_frame, self.camera_frame, Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        t, q = tfm.transform.translation, tfm.transform.rotation
        return (quaternion_to_matrix(q.x, q.y, q.z, q.w),
                np.array([t.x, t.y, t.z]))

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

        pixels, detected_ids = [], []
        for index, (lower, upper) in enumerate(self.hsv_ranges):
            centroid = detect_colour_blob(hsv, lower, upper, self.min_area)
            if centroid is not None:
                pixels.append(centroid)
                detected_ids.append(index)

        points = project_pixels_to_plane(
            pixels, self.intrinsics, rotation, origin,
            plane_point=[0.0, 0.0, self.plane_offset],
            plane_normal=self.plane_normal) if pixels else []

        out = CableMarkerArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.reference_frame
        out.experiment_id = "camera"

        found = dict(zip(detected_ids, points))
        for index, s in enumerate(self.s_over_l):
            marker = CableMarker()
            marker.id = index
            marker.s_over_l = float(s)
            point = found.get(index)
            marker.valid = point is not None
            if marker.valid:
                marker.position.x = float(point[0])
                marker.position.y = float(point[1])
                marker.position.z = float(point[2])
                marker.covariance = self._covariance
                marker.confidence = 1.0
            out.markers.append(marker)

        self.pub.publish(out)


def main():
    rclpy.init()
    node = CableMarkerTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
