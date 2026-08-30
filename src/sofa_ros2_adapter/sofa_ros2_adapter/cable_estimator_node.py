"""Online cable parameter identification against the SOFA estimator model.

This is the second SOFA instance of the identification setup: same geometry and
boundary conditions as the truth plant, deliberately wrong physical parameters,
and an unscented filter that corrects them from marker observations.

Contract (plan step 5):

    in   /cable/observed_markers            noisy markers, any frame on TF
    out  /cable/model/frames                predicted centerline
         /cable/model/predicted_markers     predicted markers
         /cable/parameter_estimate          one message per identified parameter
         /cable/innovation                  prediction error [m]
         /cable/marker_rmse                 residual after the correction [m]
         /cable/estimator_status            accepted / why it was rejected

The estimator never drives the experiment. It reads the same boundary poses as
the truth plant and owns no grasp state, which is what keeps the identification
from being circular.

Each sigma point is evaluated by replaying the cable under trial parameters and
rewinding it, so the estimator is left exactly where the rollout found it.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Float64, String
from cable_msgs.msg import CableMarker, CableMarkerArray, CableParameterEstimate

import tf2_ros

from cable_identification.parameter_bounds import bounds_for
from cable_identification.parameter_filter import LogParameterUKF


def _rotate(quaternion, points):
    """Rotate an (N, 3) array by [qx, qy, qz, qw] without a quaternion dep."""
    x, y, z, w = quaternion
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return points @ rotation.T


class CableEstimatorNode(Node):
    def __init__(self):
        super().__init__("cable_estimator")

        self.declare_parameter("cable_config", "")
        self.declare_parameter("identified_parameters", ["EI"])
        self.declare_parameter("parameter_bounds_file", "")
        self.declare_parameter("initial_relative_std", [0.6])
        self.declare_parameter("process_relative_std", [0.01])
        self.declare_parameter("measurement_std_m", 0.002)
        self.declare_parameter("max_innovation_m", 0.05)
        self.declare_parameter("max_relative_std", 1.5)
        self.declare_parameter("rollout_steps", 4)
        self.declare_parameter("update_rate_hz", 8.0)
        self.declare_parameter("observation_topic", "/cable/observed_markers")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("grasp_frame", "cable_grasp_frame")
        self.declare_parameter("fixture_frame", "cable_fixture_frame")
        self.declare_parameter("max_steps_per_cycle", 20)

        self.base_frame = self.get_parameter("base_frame").value
        self.grasp_frame = self.get_parameter("grasp_frame").value
        self.fixture_frame = self.get_parameter("fixture_frame").value
        self.rollout_steps = int(self.get_parameter("rollout_steps").value)
        self.max_steps = int(self.get_parameter("max_steps_per_cycle").value)
        self.measurement_std = float(self.get_parameter("measurement_std_m").value)
        self.max_innovation = float(self.get_parameter("max_innovation_m").value)
        self.max_relative_std = float(self.get_parameter("max_relative_std").value)

        # --- SOFA estimator scene (in-process, same interpreter as rclpy) ---
        import Sofa.Core
        import Sofa.Simulation
        from cable_identification import cosserat_model as cm
        from cable_identification.coupling import GraspCoupling

        self._sim = Sofa.Simulation
        self.cfg = cm.load_config(self.get_parameter("cable_config").value or None)
        self.table_mode = bool(self.cfg.get("grasp_tip"))
        self.marker_s = list(self.cfg["marker_s_over_l"])

        self.root = Sofa.Core.Node("root")
        cm.prepare_root(self.root, self.cfg)
        self.cable = cm.build_cable(self.root, self.cfg)
        Sofa.Simulation.init(self.root)
        self.dt = self.root.dt.value

        # The estimator mirrors the grasp kinematics but owns no grasp state:
        # it latches on the truth plant's published state, never on its own.
        self.coupling = GraspCoupling(self.cable) if self.table_mode else None

        # --- filter ---
        names = tuple(self.get_parameter("identified_parameters").value)
        initial = tuple(self.cable.get_parameter(n) for n in names)
        bounds = self._load_bounds(names)
        self.filter = LogParameterUKF(
            names=names,
            initial_values=initial,
            initial_relative_std=self._vector("initial_relative_std", len(names)),
            process_relative_std=self._vector("process_relative_std", len(names)),
            bounds=bounds,
        )
        self.get_logger().info(
            f"identifying {list(names)} from {initial} "
            f"(bounds={'none' if bounds is None else bounds})")

        # --- ROS interfaces ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.frames_pub = self.create_publisher(PoseArray, "/cable/model/frames", 10)
        self.markers_pub = self.create_publisher(
            CableMarkerArray, "/cable/model/predicted_markers", 10)
        self.estimate_pub = self.create_publisher(
            CableParameterEstimate, "/cable/parameter_estimate", 10)
        self.innovation_pub = self.create_publisher(Float64, "/cable/innovation", 10)
        self.rmse_pub = self.create_publisher(Float64, "/cable/marker_rmse", 10)
        self.status_pub = self.create_publisher(String, "/cable/estimator_status", 10)

        self._observation = None
        self._sim_time = None
        self.create_subscription(
            CableMarkerArray, self.get_parameter("observation_topic").value,
            self._on_observation, 10)
        rate = float(self.get_parameter("update_rate_hz").value)
        self.timer = self.create_timer(1.0 / rate, self._cycle)

    # ------------------------------------------------------------------
    def _vector(self, name, size):
        values = [float(v) for v in self.get_parameter(name).value]
        if len(values) == 1:
            return values * size
        if len(values) != size:
            raise ValueError(f"{name} must have 1 or {size} entries")
        return values

    def _load_bounds(self, names):
        path = self.get_parameter("parameter_bounds_file").value
        if not path:
            return None
        return bounds_for(path, names)

    def _on_observation(self, msg):
        self._observation = msg

    def _lookup(self, target, source=None):
        try:
            tfm = self.tf_buffer.lookup_transform(
                source or self.base_frame, target, Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        t, q = tfm.transform.translation, tfm.transform.rotation
        return [t.x, t.y, t.z, q.x, q.y, q.z, q.w]

    # ------------------------------------------------------------------
    def _drive_boundary(self, now_s):
        """Follow the same boundary poses as the truth plant."""
        if self.table_mode:
            if not self.coupling.fixture_set:
                fixture = self._lookup(self.fixture_frame)
                if fixture is None:
                    return False
                self.coupling.on_fixture(fixture)
            grasp = self._lookup(self.grasp_frame)
            if grasp is not None:
                self.coupling.update_grasp(grasp, now_s)
            return True
        grasp = self._lookup(self.grasp_frame)
        if grasp is None:
            return False
        self.cable.set_base_pose(grasp)
        return True

    def _step_to(self, now_s):
        if self._sim_time is None:
            self._sim_time = now_s
        n = int((now_s - self._sim_time) / self.dt)
        for _ in range(min(n, self.max_steps)):
            self._sim.animate(self.root, self.dt)
        self._sim_time += n * self.dt

    def _to_frame(self, points, frame):
        """Express base-frame points in the observation frame."""
        if not frame or frame == self.base_frame:
            return points
        pose = self._lookup(self.base_frame, source=frame)
        if pose is None:
            return points
        return _rotate(pose[3:7], points) + np.asarray(pose[:3])

    def _predict_markers(self, theta, frame):
        """Marker prediction for trial parameters, leaving the state untouched."""
        state = self.cable.save_state()
        for name, value in zip(self.filter.names, theta):
            self.cable.set_parameter(name, value)
        for _ in range(self.rollout_steps):
            self._sim.animate(self.root, self.dt)
        points = np.asarray(self.cable.marker_positions(), dtype=float)
        self.cable.restore_state(state)
        return self._to_frame(points, frame).ravel()

    def _observation_vector(self, msg):
        count = len(self.marker_s)
        z = np.zeros(3 * count)
        mask = np.zeros(3 * count, dtype=bool)
        for marker in msg.markers:
            if not marker.valid or not 0 <= marker.id < count:
                continue
            i = 3 * marker.id
            z[i:i + 3] = (marker.position.x, marker.position.y, marker.position.z)
            mask[i:i + 3] = True
        return z, mask

    # ------------------------------------------------------------------
    def _identify(self, stamp):
        msg, self._observation = self._observation, None
        z, mask = self._observation_vector(msg)
        frame = msg.header.frame_id

        self.filter.predict()
        result = self.filter.update(
            lambda theta: self._predict_markers(theta, frame),
            z, self.measurement_std, mask=mask, point_dim=3,
            max_relative_std=self.max_relative_std,
            max_innovation=self.max_innovation)

        if result.accepted:
            for name, value in zip(self.filter.names, self.filter.values):
                self.cable.set_parameter(name, value)

        for name, value, std in zip(self.filter.names, self.filter.values,
                                    self.filter.std_dev):
            estimate = CableParameterEstimate()
            estimate.header.stamp = stamp
            estimate.header.frame_id = self.base_frame
            estimate.parameter_name = name
            estimate.value = float(value)
            estimate.std_dev = float(std)
            estimate.innovation = float(result.innovation)
            estimate.marker_rmse = float(result.marker_rmse)
            self.estimate_pub.publish(estimate)

        self.innovation_pub.publish(Float64(data=float(result.innovation)))
        self.rmse_pub.publish(Float64(data=float(result.marker_rmse)))
        self.status_pub.publish(String(
            data="accepted" if result.accepted else f"rejected: {result.reason}"))

    def _publish_prediction(self, stamp):
        poses = self.cable.frame_poses()
        array = PoseArray()
        array.header.stamp = stamp
        array.header.frame_id = self.base_frame
        for p in poses:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, p[:3])
            (pose.orientation.x, pose.orientation.y,
             pose.orientation.z, pose.orientation.w) = map(float, p[3:7])
            array.poses.append(pose)
        self.frames_pub.publish(array)

        markers = CableMarkerArray()
        markers.header = array.header
        markers.experiment_id = "estimator"
        for mid, (s, idx) in enumerate(zip(self.marker_s, self.cable.marker_indices)):
            marker = CableMarker()
            marker.id = mid
            marker.s_over_l = float(s)
            (marker.position.x, marker.position.y,
             marker.position.z) = map(float, poses[idx][:3])
            marker.confidence = 1.0
            marker.valid = True
            markers.markers.append(marker)
        self.markers_pub.publish(markers)

    def _cycle(self):
        now = self.get_clock().now()
        if now.nanoseconds == 0:
            return  # sim clock not up yet
        if not self._drive_boundary(now.nanoseconds * 1e-9):
            return

        self._step_to(now.nanoseconds * 1e-9)
        stamp = now.to_msg()
        self._publish_prediction(stamp)
        if self._observation is not None:
            self._identify(stamp)


def main():
    rclpy.init()
    node = CableEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
