"""Synthetic cable marker observations.

Subscribes the SOFA truth markers, applies noise / dropout / delay and
republishes them on the same topic the camera tracker will use. This lets the
estimator and the outer controller be validated before any computer-vision
error is in the loop.

Concerns: ROS transport only — the corruption itself lives in
``observation_model`` so it stays testable.
"""

import rclpy
from rclpy.node import Node

from cable_msgs.msg import CableMarker, CableMarkerArray

from cable_perception.observation_model import ObservationCorruptor, DelayLine


def _identity_of(marker):
    """Copy the identity fields of a marker, leaving the measurement blank."""
    out = CableMarker()
    out.id = marker.id
    out.s_over_l = marker.s_over_l
    return out


class SyntheticMarkerNode(Node):
    def __init__(self):
        super().__init__("synthetic_marker_node")

        self.declare_parameter("truth_topic", "/cable/truth/markers")
        self.declare_parameter("observed_topic", "/cable/observed_markers")
        self.declare_parameter("noise_std_m", 0.002)
        self.declare_parameter("dropout_probability", 0.0)
        self.declare_parameter("delay_s", 0.0)
        self.declare_parameter("seed", 0)

        seed = int(self.get_parameter("seed").value)
        self.corruptor = ObservationCorruptor(
            noise_std_m=float(self.get_parameter("noise_std_m").value),
            dropout_probability=float(
                self.get_parameter("dropout_probability").value),
            seed=seed if seed >= 0 else None)
        self.delay = DelayLine(float(self.get_parameter("delay_s").value))
        self._covariance = self.corruptor.covariance()

        self.pub = self.create_publisher(
            CableMarkerArray, self.get_parameter("observed_topic").value, 10)
        self.sub = self.create_subscription(
            CableMarkerArray, self.get_parameter("truth_topic").value,
            self._on_truth, 10)
        # Released on a timer rather than on arrival so the delay is honoured
        # even when the truth plant stalls.
        self.timer = self.create_timer(0.005, self._release)

        self.get_logger().info(
            f"synthetic markers: noise={self.corruptor.noise_std_m * 1e3:.2f} mm, "
            f"dropout={self.corruptor.dropout_probability:.2f}, "
            f"delay={self.delay.delay_s * 1e3:.0f} ms, seed={seed}")

    def _now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_truth(self, msg):
        positions = [[m.position.x, m.position.y, m.position.z] for m in msg.markers]
        if not positions:
            return
        noisy, valid = self.corruptor.corrupt(positions)

        out = CableMarkerArray()
        out.header = msg.header
        out.experiment_id = msg.experiment_id
        for marker, position, is_valid in zip(msg.markers, noisy, valid):
            observed = _identity_of(marker)
            observed.position.x = float(position[0])
            observed.position.y = float(position[1])
            observed.position.z = float(position[2])
            observed.covariance = self._covariance
            observed.valid = bool(marker.valid and is_valid)
            observed.confidence = 1.0 if observed.valid else 0.0
            out.markers.append(observed)

        self.delay.push(self._now_s(), out)

    def _release(self):
        for _stamp, msg in self.delay.pop_ready(self._now_s()):
            self.pub.publish(msg)


def main():
    rclpy.init()
    node = SyntheticMarkerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
