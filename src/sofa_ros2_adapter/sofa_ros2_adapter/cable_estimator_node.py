"""Online cable parameter identification with Optimus (ROUKF) inside SOFA.

This is the second SOFA instance of the identification setup: same geometry and
boundary conditions as the truth plant, deliberately wrong physical parameters,
and Optimus' reduced-order unscented Kalman filter correcting them from marker
observations. The estimator scene is the plant of `cosserat_model` with the
Optimus components inserted around it (`optimus_scene.build_optimus_cable`).

Contract (plan step 5, unchanged):

    in   /cable/observed_markers            noisy markers, any frame on TF
         /cable/grasp_state                 truth plant grasp state (table mode)
    out  /cable/model/frames                predicted centerline
         /cable/model/predicted_markers     predicted markers
         /cable/parameter_estimate          one message per identified parameter
         /cable/innovation                  prediction error [m]
         /cable/marker_rmse                 residual after the correction [m]
         /cable/estimator_status            what the last step did

The estimator never drives the experiment. It reads the same boundary poses as
the truth plant and owns no grasp state: it latches when the truth reports
ATTACHED, which is what keeps the identification from being circular.

Every SOFA step is a filter step (FilteringAnimationLoop): p + 1 sigma-point
propagations from the same state and the same boundary pose, then a correction
if a complete observation is available for that step, otherwise prediction only
(protocol §8.2, §8.4). Parameters are estimated in log space and stay positive.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Float64, String
from cable_msgs.msg import CableMarker, CableMarkerArray, CableParameterEstimate, GraspState

import tf2_ros

from cable_identification.parameter_bounds import bounds_for


def _rotate(quaternion, points):
    """Rotate an (N, 3) array by [qx, qy, qz, qw] without a quaternion dep."""
    x, y, z, w = quaternion
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return points @ rotation.T


def _rms_per_point(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=-1))))


class CableEstimatorNode(Node):
    def __init__(self):
        super().__init__("cable_estimator")

        self.declare_parameter("cable_config", "")
        self.declare_parameter("identified_parameters", ["EI"])
        self.declare_parameter("parameter_bounds_file", "")
        self.declare_parameter("initial_relative_std", [0.6])
        self.declare_parameter("measurement_std_m", 0.002)
        self.declare_parameter("max_innovation_m", 0.05)
        self.declare_parameter("update_rate_hz", 8.0)
        self.declare_parameter("observation_topic", "/cable/observed_markers")
        self.declare_parameter("grasp_state_topic", "/cable/grasp_state")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("grasp_frame", "cable_grasp_frame")
        self.declare_parameter("fixture_frame", "cable_fixture_frame")
        self.declare_parameter("max_steps_per_cycle", 20)

        self.base_frame = self.get_parameter("base_frame").value
        self.grasp_frame = self.get_parameter("grasp_frame").value
        self.fixture_frame = self.get_parameter("fixture_frame").value
        self.max_steps = int(self.get_parameter("max_steps_per_cycle").value)
        self.measurement_std = float(self.get_parameter("measurement_std_m").value)
        self.max_innovation = float(self.get_parameter("max_innovation_m").value)

        # --- SOFA estimator scene (in-process, same interpreter as rclpy) ---
        import Sofa.Core
        import Sofa.Simulation
        from cable_identification import cosserat_model as cm
        from cable_identification import optimus_scene as osc
        from cable_identification.coupling import GraspCoupling

        self._sim = Sofa.Simulation
        self.cfg = cm.load_config(self.get_parameter("cable_config").value or None)
        self.table_mode = bool(self.cfg.get("grasp_tip"))
        self.marker_s = list(self.cfg["marker_s_over_l"])
        self.names = tuple(self.get_parameter("identified_parameters").value)
        relative_std = self._vector("initial_relative_std", len(self.names))

        self.root = Sofa.Core.Node("root")
        cm.prepare_root(self.root, self.cfg, animation_loop=None)
        self.est = osc.build_optimus_cable(
            self.root, self.cfg, parameters=self.names, relative_std=relative_std,
            observation_std=self.measurement_std)
        Sofa.Simulation.init(self.root)
        self.cable = self.est.cable
        self.dt = self.root.dt.value

        # The estimator mirrors the grasp kinematics but owns no grasp state:
        # it latches on the truth plant's published state, never on its own.
        self.coupling = GraspCoupling(self.cable, attach_mode="explicit") if self.table_mode else None

        self.bounds = self._load_bounds(self.names)
        initial = self.est.estimates()
        self.get_logger().info(
            f"Optimus ROUKF identifying {list(self.names)} from {initial}, "
            f"prior relative std {relative_std}, observation std {self.measurement_std} m "
            f"(bounds={'none' if self.bounds is None else self.bounds})")

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
        self._truth_grasp_state = None
        self._sim_time = None
        self.create_subscription(
            CableMarkerArray, self.get_parameter("observation_topic").value,
            self._on_observation, 10)
        if self.table_mode:
            self.create_subscription(
                GraspState, self.get_parameter("grasp_state_topic").value,
                self._on_grasp_state, 10)
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
        return dict(zip(names, bounds_for(path, names)))

    def _on_observation(self, msg):
        self._observation = msg

    def _on_grasp_state(self, msg):
        self._truth_grasp_state = msg.state

    def _lookup(self, target, source=None, when=None):
        """Pose of `target` in `source` (default base_frame) at time `when` [s]
        (interpolated by the TF buffer; latest if None or unavailable)."""
        for stamp in ((Time(seconds=when),) if when is not None else ()) + (Time(),):
            try:
                tfm = self.tf_buffer.lookup_transform(source or self.base_frame, target, stamp)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                continue
            t, q = tfm.transform.translation, tfm.transform.rotation
            return [t.x, t.y, t.z, q.x, q.y, q.z, q.w]
        return None

    # ------------------------------------------------------------------
    def _ensure_fixture(self):
        """Table mode: clamp the cable base on the fixture once."""
        if not self.table_mode or self.coupling.fixture_set:
            return True
        fixture = self._lookup(self.fixture_frame)
        if fixture is None:
            return False
        self.coupling.on_fixture(fixture)
        self.est.set_boundary_pose(fixture)
        self.cable.refresh_mapping()
        return True

    def _drive_boundary(self, t_s):
        """Same boundary poses as the truth plant, sampled on TF at the step time.

        Applied before the filter step; the pose then stays constant for every
        sigma point of that step (protocol §8.2)."""
        grasp = self._lookup(self.grasp_frame, when=t_s)
        if grasp is None:
            return False
        if self.table_mode:
            self._mirror_grasp_state()
            self.coupling.update_grasp(grasp, t_s)
        else:
            self.est.set_boundary_pose(grasp)
        return True

    def _mirror_grasp_state(self):
        """Arm/release the local coupling from the truth plant's grasp state."""
        truth = self._truth_grasp_state
        if truth is None:
            return
        if truth == GraspState.ATTACHED and not self.coupling.latched:
            self.coupling.request_attach()
            if not self.coupling.in_range():
                self.get_logger().warn(
                    "truth plant is ATTACHED but the estimator tip is out of range "
                    f"(distance {self.coupling.distance_to_tip:.3f} m); waiting",
                    throttle_duration_sec=5.0)
        elif truth == GraspState.DETACHED and self.coupling.state != GraspState.DETACHED:
            self.coupling.request_detach()

    def _observation_in_base(self, msg):
        """Complete, ordered marker set in the model frame, or None."""
        count = len(self.marker_s)
        points = np.full((count, 3), np.nan)
        for marker in msg.markers:
            if marker.valid and 0 <= marker.id < count:
                points[marker.id] = (marker.position.x, marker.position.y, marker.position.z)
        if not np.all(np.isfinite(points)):
            return None, f"incomplete observation ({int(np.isfinite(points[:, 0]).sum())}/{count} markers)"
        frame = msg.header.frame_id
        if frame and frame != self.base_frame:
            pose = self._lookup(frame)   # base_frame <- frame
            if pose is None:
                return None, f"no TF {self.base_frame} <- {frame}"
            points = _rotate(pose[3:7], points) + np.asarray(pose[:3])
        return points, ""

    def _advance_to(self, t_target, msg, budget):
        """Filter steps up to `t_target`; the step ending nearest to it consumes
        `msg` (gated first). Returns (steps done, points used or None, reason)."""
        points, reason, steps = None, "no observation", 0
        while steps < budget and self._sim_time + self.dt <= t_target + 0.5 * self.dt:
            t_step = self._sim_time + self.dt
            if not self._drive_boundary(t_step):
                break
            last = self._sim_time + 2 * self.dt > t_target + 0.5 * self.dt
            if last and msg is not None:
                points, reason = self._gate_observation(msg)
            self.est.set_observation(points if (last and points is not None) else None,
                                     valid=last and points is not None)
            self._sim.animate(self.root, self.dt)
            self._sim_time = t_step
            steps += 1
        return steps, points, reason

    # ------------------------------------------------------------------
    def _gate_observation(self, msg):
        """Complete observation in the model frame, unless it fails the innovation
        gate against the current prediction (outlier / wrong frame)."""
        points, reason = self._observation_in_base(msg)
        if points is not None:
            predicted = np.asarray(self.cable.marker_positions(), dtype=float)
            gate = _rms_per_point(points, predicted)
            if gate > self.max_innovation:
                points, reason = None, f"innovation {gate:.3f} m > max_innovation_m {self.max_innovation}"
        return points, reason

    def _publish_estimate(self, stamp, corrected, reason, observation):
        estimates = self.est.estimates()
        std_devs = self.est.std_devs()
        log_var = self.est.log_variances()
        innovation = self.est.innovation_rms() if corrected else float("nan")
        rmse = (_rms_per_point(observation, self.cable.marker_positions())
                if corrected else float("nan"))

        out_of_bounds = []
        for name, value in estimates.items():
            estimate = CableParameterEstimate()
            estimate.header.stamp = stamp
            estimate.header.frame_id = self.base_frame
            estimate.parameter_name = name
            estimate.value = float(value)
            estimate.std_dev = float(std_devs[name])
            estimate.innovation = float(innovation)
            estimate.marker_rmse = float(rmse)
            self.estimate_pub.publish(estimate)
            if self.bounds and not (self.bounds[name][0] <= value <= self.bounds[name][1]):
                out_of_bounds.append(f"{name}={value:.4g} outside {self.bounds[name]}")

        if corrected:
            self.innovation_pub.publish(Float64(data=float(innovation)))
            self.rmse_pub.publish(Float64(data=float(rmse)))
        # protocol §12 per-iteration log, one line
        fields = ", ".join(
            f"{n}={v:.5g} (log-std {math.sqrt(max(log_var[n], 0.0)):.3g}, "
            f"[{v * math.exp(-2 * math.sqrt(max(log_var[n], 0.0))):.4g}, "
            f"{v * math.exp(2 * math.sqrt(max(log_var[n], 0.0))):.4g}])"
            for n, v in estimates.items())
        status = ("corrected" if corrected else f"prediction only: {reason}") + f"; {fields}"
        if corrected:
            status += f"; innovation {innovation:.4f} m; rmse {rmse:.4f} m"
        if out_of_bounds:
            status += "; OUT OF BOUNDS " + "; ".join(out_of_bounds)
            self.get_logger().warn("estimate outside the parameter envelope: " + "; ".join(out_of_bounds),
                                   throttle_duration_sec=5.0)
        self.status_pub.publish(String(data=status))

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
        now_s = now.nanoseconds * 1e-9
        if not self._ensure_fixture():
            return
        if self._sim_time is None:
            if self._lookup(self.grasp_frame) is None:
                return
            self._sim_time = now_s
        stamp = now.to_msg()

        # 1. step to the observation's own time and correct there (no latency bias),
        # 2. then predict up to now. A stale observation (older than the model) is
        #    applied at the next step; a future stamp is treated as now.
        msg, self._observation = self._observation, None
        budget = self.max_steps
        steps, points, reason = 0, None, "no observation"
        if msg is not None:
            t_obs = min(now_s, max(self._sim_time + self.dt,
                                   Time.from_msg(msg.header.stamp).nanoseconds * 1e-9))
            steps, points, reason = self._advance_to(t_obs, msg, budget)
            budget -= steps
            if steps == 0:
                reason = "observation older than the model state"
        more, _, _ = self._advance_to(now_s, None, budget)
        steps += more
        if self._sim_time < now_s - self.dt * self.max_steps:
            self.get_logger().warn(
                f"estimator {now_s - self._sim_time:.2f} s behind real time "
                f"(max_steps_per_cycle={self.max_steps}); dropping the backlog",
                throttle_duration_sec=5.0)
            self._sim_time = now_s

        self._publish_prediction(stamp)
        if msg is not None:
            self._publish_estimate(stamp, corrected=points is not None, reason=reason,
                                   observation=points)


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
