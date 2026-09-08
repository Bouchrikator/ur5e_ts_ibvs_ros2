"""Visual observer of the Cosserat-modal state: markers -> ``x = [a, a_dot, p_g - p_g0]``.

Same ``/cable/reduced_state`` contract as ``cable_state_reducer_node`` but the
coordinates are those of the ONE strain POD. The observation model is the ROM
graph (``ModelOrderReductionMapping`` -> ``DiscreteCosseratMapping``), used
kinematically; the prior of each update is the TS prediction when a model is
given, otherwise the previous estimate. A target published in marker space is
converted to ``a*`` with the same observation model.
"""

import numpy as np
import rclpy
import tf2_ros
import yaml
from rclpy.node import Node
from rclpy.time import Time

from cable_identification import cosserat_model as cm
from cable_identification.modal_observation import ModalObservationModel
from cable_identification.strain_basis import ReductionSpec
from cable_msgs.msg import CableMarkerArray, CableShapeTarget, CableState
from cable_ts_control.cable_state_reducer_node import quaternion_to_matrix
from cable_ts_control.modal_contract import COORDINATES, build_modal_state, check_model_contract
from cable_ts_control.modal_observer import ModalObserver
from cable_ts_control.state_filter import ModalVelocityFilter
from cable_ts_control.ts_model import CableTsModel


class CableModalObserverNode(Node):
    def __init__(self):
        super().__init__("cable_modal_observer_node")
        self.declare_parameter("cable_config", "")
        self.declare_parameter("strain_modes_file", "")
        self.declare_parameter("strain_modes_metadata", "")
        self.declare_parameter("ts_model_file", "")
        self.declare_parameter("observed_topic", "/cable/observed_markers")
        self.declare_parameter("state_topic", "/cable/reduced_state")
        self.declare_parameter("target_topic", "/cable/target_shape")
        self.declare_parameter("reference_frame", "cable_fixture_frame")
        self.declare_parameter("gripper_frame", "cable_grasp_frame")
        self.declare_parameter("marker_std_m", 0.002)
        self.declare_parameter("prior_std", 0.2)
        self.declare_parameter("max_iterations", 3)
        self.declare_parameter("max_residual_rmse_m", 0.05)

        with open(self.get_parameter("strain_modes_metadata").value) as stream:
            self.metadata = yaml.safe_load(stream)
        cfg = cm.load_config(self.get_parameter("cable_config").value or None)
        self.observation = ModalObservationModel(cfg, ReductionSpec(
            self.get_parameter("strain_modes_file").value,
            self.get_parameter("strain_modes_metadata").value, int(self.metadata["n_modes"])))
        self.n_modes = self.observation.n_modes
        self.model = None
        self.gripper_reference = np.zeros(2)
        self.alpha = 0.4
        model_file = self.get_parameter("ts_model_file").value
        if model_file:
            with open(model_file) as stream:
                section = yaml.safe_load(stream)["cable_ts_model"]
            check_model_contract(section, self.metadata, expected_coordinates=COORDINATES)
            self.model = CableTsModel.load(model_file)
            self.gripper_reference = np.asarray(section["gripper_reference"], dtype=float)
            self.alpha = float(section["velocity_alpha"])
        self.observer = ModalObserver(
            self.observation.markers, self.n_modes, float(self.get_parameter("marker_std_m").value),
            float(self.get_parameter("prior_std").value), int(self.get_parameter("max_iterations").value))
        self.max_residual = float(self.get_parameter("max_residual_rmse_m").value)
        self.reference_frame = self.get_parameter("reference_frame").value
        self.gripper_frame = self.get_parameter("gripper_frame").value
        self.filter = ModalVelocityFilter(self.n_modes, self.alpha)
        self._a = np.zeros(self.n_modes)
        self._state = None
        self._command = np.zeros(2)
        self._previous = None  # (stamp_s, gripper)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(CableState, self.get_parameter("state_topic").value, 10)
        self.target_pub = self.create_publisher(
            CableShapeTarget, self.get_parameter("target_topic").value, 10)
        self.create_subscription(CableMarkerArray, self.get_parameter("observed_topic").value,
                                 self._on_markers, 10)
        self.create_subscription(CableShapeTarget, self.get_parameter("target_topic").value,
                                 self._on_target, 10)
        self.get_logger().info(
            f"cable modal observer: r={self.n_modes} strain modes, {2 * self.observation.n_markers} "
            f"planar measurements, prior={'TS model' if self.model else 'previous estimate'}, "
            f"basis {self.metadata['strain_basis_sha256'][:12]}")

    def _lookup(self, target, source, stamp):
        try:
            return self.tf_buffer.lookup_transform(target, source, stamp)
        except tf2_ros.ExtrapolationException:
            return self.tf_buffer.lookup_transform(target, source, Time())

    def _positions(self, msg):
        positions = np.array([[m.position.x, m.position.y, m.position.z] for m in msg.markers])
        if msg.header.frame_id == self.reference_frame:
            return positions
        tfm = self._lookup(self.reference_frame, msg.header.frame_id, Time.from_msg(msg.header.stamp))
        t, q = tfm.transform.translation, tfm.transform.rotation
        return positions @ quaternion_to_matrix(q.x, q.y, q.z, q.w).T + np.array([t.x, t.y, t.z])

    def _on_markers(self, msg):
        if len(msg.markers) != self.observation.n_markers:
            self.get_logger().warn(f"expected {self.observation.n_markers} markers, got {len(msg.markers)}",
                                   throttle_duration_sec=5.0)
            return
        stamp = Time.from_msg(msg.header.stamp)
        try:
            positions = self._positions(msg)
            tfm = self._lookup(self.reference_frame, self.gripper_frame, stamp)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.get_logger().warn("missing transform for markers or gripper", throttle_duration_sec=5.0)
            return
        gripper = np.array([tfm.transform.translation.x, tfm.transform.translation.y])
        stamp_s = stamp.nanoseconds * 1e-9
        dt = stamp_s - self._previous[0] if self._previous else 0.0
        if self._previous and dt > 1e-6:
            # The gripper integrates the command, so the applied u is its finite difference.
            self._command = (gripper - self._previous[1]) / dt
        prior = self._a
        if self.model is not None and self._state is not None:
            prior = self.model.predict(self._state, self._command)[:self.n_modes]
        valid = np.array([m.valid for m in msg.markers], dtype=bool)
        a, info = self.observer.update(positions[:, :2].ravel(), valid, prior)
        state = CableState()
        state.header = msg.header
        state.header.frame_id = self.reference_frame
        state.markers_used = int(valid.sum())
        state.reconstruction_rmse = float(info["residual_rmse_m"]) if valid.any() else 0.0
        if not valid.any() or info["residual_rmse_m"] > self.max_residual:
            state.valid = False
            self.pub.publish(state)
            self.get_logger().warn(f"observer residual {info['residual_rmse_m'] * 1e3:.1f} mm rejected",
                                   throttle_duration_sec=5.0)
            return
        self._a = a
        velocity = self.filter.update(a, dt)
        self._state = build_modal_state(a, velocity, gripper, self.gripper_reference)
        self._previous = (stamp_s, gripper)
        state.modal_coordinates = a.tolist()
        state.modal_velocities = velocity.tolist()
        state.state = self._state.tolist()
        state.valid = True
        self.pub.publish(state)

    def _on_target(self, msg):
        """Marker-space targets are converted to a* with the same observation model."""
        markers = np.asarray(msg.marker_positions, dtype=float)
        if len(msg.modal_coordinates) in (self.n_modes, self.n_modes + 2) or markers.size == 0:
            return
        if markers.size != 3 * self.observation.n_markers:
            self.get_logger().warn("target has neither a* nor the full marker set", throttle_duration_sec=5.0)
            return
        a_star = np.zeros(self.n_modes)
        for _ in range(5):
            a_star, _ = self.observer.update(markers.reshape(-1, 3)[:, :2].ravel(),
                                             np.ones(self.observation.n_markers, dtype=bool), a_star)
        out = CableShapeTarget()
        out.header = msg.header
        out.target_name = f"{msg.target_name}@{COORDINATES}"
        out.modal_coordinates = a_star.tolist()
        out.marker_positions = msg.marker_positions
        self.target_pub.publish(out)


def main():
    rclpy.init()
    node = CableModalObserverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.observation.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
