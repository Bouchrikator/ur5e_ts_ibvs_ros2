"""Cable state reducer: observed markers -> reduced modal state.

Turns the raw marker cloud into the ``x = [q; qdot; g]`` the TS-PDC controller
expects, expressed in the fixture frame so the state is independent of where
the robot base happens to be. ``g`` is the gripper displacement, read from TF:
the input is gripper VELOCITY, so without it a displaced but stationary gripper
keeps the cable deformed while contributing nothing to the state, and the
reduced description is not Markov.

Concerns: ROS transport and frame handling. The projection maths lives in
``modal_basis`` and stays unit tested.
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.time import Time

import tf2_ros

from cable_msgs.msg import CableMarkerArray, CableState

from cable_ts_control.modal_basis import ModalBasis
from cable_ts_control.state_filter import ModalVelocityFilter


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


class CableStateReducerNode(Node):
    def __init__(self):
        super().__init__("cable_state_reducer_node")

        self.declare_parameter("modal_basis_file", "")
        self.declare_parameter("observed_topic", "/cable/observed_markers")
        self.declare_parameter("state_topic", "/cable/reduced_state")
        self.declare_parameter("reference_frame", "cable_fixture_frame")
        self.declare_parameter("gripper_frame", "cable_grasp_frame")
        self.declare_parameter("velocity_filter_alpha", 0.4)
        self.declare_parameter("max_reconstruction_rmse_m", 0.05)

        basis_file = self.get_parameter("modal_basis_file").value
        if not basis_file:
            raise ValueError("modal_basis_file is required")
        self.basis = ModalBasis.load(basis_file)

        self.reference_frame = self.get_parameter("reference_frame").value
        self.gripper_frame = self.get_parameter("gripper_frame").value
        self.alpha = float(self.get_parameter("velocity_filter_alpha").value)
        self.max_rmse = float(self.get_parameter("max_reconstruction_rmse_m").value)
        self.boundary_reference = (
            np.zeros(2) if self.basis.boundary_reference is None
            else np.asarray(self.basis.boundary_reference, dtype=float))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub = self.create_publisher(
            CableState, self.get_parameter("state_topic").value, 10)
        self.create_subscription(
            CableMarkerArray, self.get_parameter("observed_topic").value,
            self._on_markers, 10)

        self._previous_stamp = None
        self._filter = ModalVelocityFilter(self.basis.n_modes, self.alpha)

        self.get_logger().info(
            f"cable state reducer: {self.basis.n_modes} modes over "
            f"{self.basis.dimension // 2} planar markers, frame="
            f"{self.reference_frame}, gripper={self.gripper_frame}")

    # ------------------------------------------------------------------
    def _gripper_displacement(self):
        """Gripper position in the reference frame, minus the basis origin."""
        try:
            tfm = self.tf_buffer.lookup_transform(
                self.reference_frame, self.gripper_frame, Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        t = tfm.transform.translation
        return np.array([t.x, t.y]) - self.boundary_reference

    # ------------------------------------------------------------------
    def _to_reference_frame(self, msg):
        """Marker positions expressed in the reference frame, as an (M, 3) array."""
        positions = np.array(
            [[m.position.x, m.position.y, m.position.z] for m in msg.markers])
        if msg.header.frame_id == self.reference_frame:
            return positions
        try:
            tfm = self.tf_buffer.lookup_transform(
                self.reference_frame, msg.header.frame_id, Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        t, q = tfm.transform.translation, tfm.transform.rotation
        rotation = quaternion_to_matrix(q.x, q.y, q.z, q.w)
        return positions @ rotation.T + np.array([t.x, t.y, t.z])

    def _publish_invalid(self, header, markers_used, rmse=0.0):
        msg = CableState()
        msg.header = header
        msg.valid = False
        msg.markers_used = markers_used
        msg.reconstruction_rmse = rmse
        self.pub.publish(msg)

    def _on_markers(self, msg):
        expected = self.basis.dimension // 2
        if len(msg.markers) != expected:
            self.get_logger().warn(
                f"expected {expected} markers, got {len(msg.markers)}",
                throttle_duration_sec=5.0)
            return

        positions = self._to_reference_frame(msg)
        if positions is None:
            self.get_logger().warn(
                f"no transform to {self.reference_frame}",
                throttle_duration_sec=5.0)
            return

        # Planar shaping: the reduced description only uses x and y.
        shape = positions[:, :2].reshape(-1)
        valid = np.repeat([m.valid for m in msg.markers], 2)
        markers_used = int(sum(m.valid for m in msg.markers))

        gripper = self._gripper_displacement()
        if gripper is None:
            self.get_logger().warn(
                f"no transform {self.reference_frame} -> {self.gripper_frame}",
                throttle_duration_sec=5.0)
            return
        boundary = gripper if self.basis.psi is not None else None

        try:
            q = self.basis.project(shape, valid, boundary)
        except ValueError:
            self.get_logger().warn(
                f"too few valid markers ({markers_used}) to identify "
                f"{self.basis.n_modes} modes", throttle_duration_sec=5.0)
            self._publish_invalid(msg.header, markers_used)
            return

        residual = (shape - self.basis.reconstruct(q, boundary))[valid]
        rmse = float(np.sqrt(np.mean(residual ** 2))) if residual.size else 0.0

        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        dt = stamp - self._previous_stamp if self._previous_stamp else 0.0
        velocity = self._filter.update(q, dt)
        self._previous_stamp = stamp

        if rmse > self.max_rmse:
            # The shape left the span of the identified basis: the reduced
            # state is meaningless, so say so instead of feeding the controller.
            self.get_logger().warn(
                f"reconstruction rmse {rmse * 1e3:.1f} mm exceeds "
                f"{self.max_rmse * 1e3:.1f} mm", throttle_duration_sec=5.0)
            self._publish_invalid(msg.header, markers_used, rmse)
            return

        state = CableState()
        state.header = msg.header
        state.header.frame_id = self.reference_frame
        state.modal_coordinates = q.tolist()
        state.modal_velocities = velocity.tolist()
        state.state = np.concatenate([q, velocity, gripper]).tolist()
        state.reconstruction_rmse = rmse
        state.markers_used = markers_used
        state.valid = True
        self.pub.publish(state)


def main():
    rclpy.init()
    node = CableStateReducerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
