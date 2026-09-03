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
with the markers observed for that step (a missing marker has zero weight),
otherwise prediction only (protocol §8.2, §8.4). Parameters are estimated in
log space and stay positive.

Time is the model's: it advances only by filter steps. Observations are
processed in stamp order, each one at the step ending nearest its stamp, with
the boundary pose read from TF at every step time; an observation the model
has already passed is dropped, never assigned to a later step. The innovation
gate runs inside the filter after the prediction (NIS vs chi-square for the
observed coordinates), so it sees the innovation at the observation time.
"""

import math
from collections import deque

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Float64, String
from cable_msgs.msg import CableMarker, CableMarkerArray, CableParameterEstimate, GraspState

import tf2_ros

from cable_identification.parameter_bounds import bounds_for

TF_CACHE_S = 10.0   # boundary history the model can still step through


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
        self.declare_parameter("innovation_gate_sigma", 3.0)
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
            observation_std=self.measurement_std,
            innovation_gate_sigma=float(self.get_parameter("innovation_gate_sigma").value))
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
        if "EI" in self.names and self.table_mode and not any(self.cfg["gravity"]):
            L, ei = float(self.cfg["length_m"]), float(self.cfg["EI_Nm2"])
            self.get_logger().warn(
                "table setup without gravity: the only absolute force scales are the attachment "
                f"springs (k_g L^3/EI = {float(self.cfg['grasp_stiffness']) * L ** 3 / ei:.3g}, "
                f"k_theta L/EI = {float(self.cfg['grasp_angular_stiffness']) * L / ei:.3g}) and "
                "inertia, so EI is identified relative to the modelled clamp compliance; on the "
                "real cable calibrate the clamp or identify EI hanging under gravity")

        # --- ROS interfaces ---
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=TF_CACHE_S))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.frames_pub = self.create_publisher(PoseArray, "/cable/model/frames", 10)
        self.markers_pub = self.create_publisher(
            CableMarkerArray, "/cable/model/predicted_markers", 10)
        self.estimate_pub = self.create_publisher(
            CableParameterEstimate, "/cable/parameter_estimate", 10)
        self.innovation_pub = self.create_publisher(Float64, "/cable/innovation", 10)
        self.rmse_pub = self.create_publisher(Float64, "/cable/marker_rmse", 10)
        self.status_pub = self.create_publisher(String, "/cable/estimator_status", 10)

        self._pending = deque(maxlen=64)   # observations in arrival order; oldest dropped if flooded
        self._truth_grasp_state = None
        self._sim_time = None
        self.create_subscription(
            CableMarkerArray, self.get_parameter("observation_topic").value,
            lambda msg: self._pending.append(msg), 10)   # rclpy inspects the signature: no builtin
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

    def _on_grasp_state(self, msg):
        self._truth_grasp_state = msg.state

    def _lookup(self, target, when=None):
        """Pose of `target` in base_frame at time `when` [s] (TF interpolation;
        latest if None). None if TF has no answer for that time: the caller waits."""
        try:
            tfm = self.tf_buffer.lookup_transform(
                self.base_frame, target, Time(seconds=when) if when is not None else Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        t, q = tfm.transform.translation, tfm.transform.rotation
        return [t.x, t.y, t.z, q.x, q.y, q.z, q.w]

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

    def _observation_in_base(self, msg, t_obs):
        """Per-marker positions (NaN = not observed) and variances in the model
        frame, or (None, None, reason). The frame transform is taken at the
        observation stamp: a wrist camera moves between the stamp and now."""
        count = len(self.marker_s)
        points = np.full((count, 3), np.nan)
        variances = np.full((count, 3), self.measurement_std ** 2)
        for marker in msg.markers:
            if marker.valid and 0 <= marker.id < count:
                points[marker.id] = (marker.position.x, marker.position.y, marker.position.z)
                diag = np.asarray(marker.covariance, dtype=float).reshape(3, 3).diagonal()
                if np.all(np.isfinite(diag)) and np.all(diag > 0.0):
                    variances[marker.id] = diag
        n_valid = int(np.isfinite(points[:, 0]).sum())
        if n_valid == 0:
            return None, None, "no valid marker"
        frame = msg.header.frame_id
        if frame and frame != self.base_frame:
            pose = self._lookup(frame, when=t_obs)   # base_frame <- frame
            if pose is None:
                return None, None, f"no TF {self.base_frame} <- {frame} at the observation stamp"
            rotation = _rotate(pose[3:7], np.eye(3)).T      # p_base = rotation @ p + t
            points = _rotate(pose[3:7], points) + np.asarray(pose[:3])
            variances = variances @ (rotation ** 2).T        # diagonal of R diag(v) R'
        return points, variances, f"{n_valid}/{count} markers"

    def _step(self, observation=None):
        """One filter step ending at _sim_time + dt with the boundary sampled
        there; `observation` = (points, variances) or None for prediction only.
        False (nothing done) when TF has no pose for that time yet."""
        t = self._sim_time + self.dt
        if not self._drive_boundary(t):
            return False
        if observation is None:
            self.est.set_observation(None, valid=False)
        else:
            self.est.set_observation(*observation)
        self._sim.animate(self.root, self.dt)
        self._sim_time = t
        return True

    # ------------------------------------------------------------------
    def _publish_estimate(self, stamp, observation, reason):
        """`observation`: the per-marker points given to the step just done, None
        when the step had none (or no step ran: stale message)."""
        estimates = self.est.estimates()
        std_devs = self.est.std_devs()
        log_var = self.est.log_variances()
        corrected = observation is not None and self.est.correction_applied()
        innovation, (nis, dof) = float("nan"), (float("nan"), 0)
        if observation is not None:
            innovation, (nis, dof) = self.est.innovation_rms(), self.est.nis()
        rmse = float("nan")
        if corrected:
            mask = self.est.observed
            rmse = _rms_per_point(observation[mask], np.asarray(self.cable.marker_positions())[mask])

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
        if corrected:
            status = f"corrected ({reason}); {fields}; innovation {innovation:.4f} m; rmse {rmse:.4f} m"
        elif observation is not None and dof:
            status = f"rejected by the innovation gate ({reason}); innovation {innovation:.4f} m; {fields}"
        else:
            status = f"prediction only: {reason}; {fields}"
        if dof:
            status += f"; NIS {nis:.1f} / {dof} dof"
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
        if not self._ensure_fixture():
            return
        budget = self.max_steps
        while self._pending and budget > 0:
            msg = self._pending[0]
            t_obs = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
            if self._sim_time is None:
                self._sim_time = t_obs - self.dt   # the first observation lands on step 1
            n = round((t_obs - self._sim_time) / self.dt)   # step ending nearest the stamp
            if n < 1:
                self._pending.popleft()
                self._publish_estimate(msg.header.stamp, None,
                                       f"stale: stamped {self._sim_time - t_obs:.3f} s before the model time")
                continue
            points, reason, observation = None, "", None
            if n <= budget:
                points, variances, reason = self._observation_in_base(msg, t_obs)
                observation = (points, variances) if points is not None else None
            done = 0
            while done < min(n, budget) and self._step(observation if done == n - 1 else None):
                done += 1
            budget -= done
            if done < n:
                t_next = self._sim_time + self.dt
                if self.get_clock().now().nanoseconds * 1e-9 - t_next > TF_CACHE_S:
                    # ponytail: the shape is not re-settled at the new boundary (one transient
                    # pollutes the next corrections); upgrade = reset the scene and re-latch
                    self.get_logger().error(
                        f"boundary history at {t_next:.2f} s is older than the TF cache "
                        f"({TF_CACHE_S:.0f} s): re-anchoring the model at the observation time")
                    self._sim_time = t_obs - self.dt
                    continue
                break   # TF not there yet or budget spent: resume at the next cycle
            self._pending.popleft()
            self._publish_estimate(msg.header.stamp, points, reason)
        if self._sim_time is None:
            return
        if self._pending:
            behind = Time.from_msg(self._pending[-1].header.stamp).nanoseconds * 1e-9 - self._sim_time
            if behind > self.dt * self.max_steps:
                self.get_logger().warn(
                    f"estimator {behind:.2f} s behind the observations "
                    f"(max_steps_per_cycle={self.max_steps}, {len(self._pending)} queued)",
                    throttle_duration_sec=5.0)
        self._publish_prediction(Time(seconds=self._sim_time).to_msg())


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
