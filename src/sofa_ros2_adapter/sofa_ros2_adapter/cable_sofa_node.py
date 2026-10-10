"""SOFA Cosserat cable <-> ROS 2 adapter node.

SofaPython3 and rclpy run in the SAME interpreter. Two coupling modes,
selected by the cable config:

  grasp_tip: false  (hanging)  TF grasp frame drives the cable BASE, tip free.
  grasp_tip: true   (table)    base clamped at the fixture frame; the TIP is
                               bilaterally constrained to a target that latches
                               onto the gripper when it comes close, then
                               tracks it (planar shaping setup).

Each cycle: TF in -> SOFA steps to /clock -> centerline frames + markers out.
Cable mechanics are owned exclusively by SOFA (Gazebo cable is visual-only).

The same node serves both roles of the identification setup, selected by the
``role`` parameter, which sets the default output namespace:

  role: truth      -> /cable/truth/{frames,markers}            simulated reality
  role: estimator  -> /cable/model/{frames,predicted_markers}   the prediction

Running two instances with different configs is what keeps the identification
from being circular. Only the truth instance owns the grasp state.
"""

import math
import time

from cable_identification.coupling import IncompatibleGraspCommand
from cable_identification.dynamics_dump import pose_delta
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from cable_msgs.msg import CableMarker, CableMarkerArray, GraspState

import tf2_ros

ROLE_TOPICS = {
    "truth": ("/cable/truth/frames", "/cable/truth/markers"),
    "estimator": ("/cable/model/frames", "/cable/model/predicted_markers"),
}


class CableSofaNode(Node):
    def __init__(self):
        super().__init__("cable_sofa_node")

        self.declare_parameter("cable_config", "")
        self.declare_parameter("role", "truth")
        self.declare_parameter("frames_topic", "")
        self.declare_parameter("markers_topic", "")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("grasp_frame", "cable_grasp_frame")
        self.declare_parameter("fixture_frame", "cable_fixture_frame")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("max_steps_per_cycle", 0)  # 0: derived from timestep and rate

        self.base_frame = self.get_parameter("base_frame").value
        self.grasp_frame = self.get_parameter("grasp_frame").value
        self.fixture_frame = self.get_parameter("fixture_frame").value
        self.max_steps = int(self.get_parameter("max_steps_per_cycle").value)

        self.role = self.get_parameter("role").value
        if self.role not in ROLE_TOPICS:
            raise ValueError(
                f"role must be one of {sorted(ROLE_TOPICS)}, got {self.role!r}")
        default_frames, default_markers = ROLE_TOPICS[self.role]
        frames_topic = self.get_parameter("frames_topic").value or default_frames
        markers_topic = self.get_parameter("markers_topic").value or default_markers

        # --- SOFA scene (in-process) ---
        import Sofa.Core
        import Sofa.Simulation
        from cable_identification import cosserat_model as cm
        from cable_identification.coupling import GraspCoupling

        self._sofa_sim = Sofa.Simulation
        cfg_path = self.get_parameter("cable_config").value or None
        self.cfg = cm.load_config(cfg_path)
        self.table_mode = bool(self.cfg.get("grasp_tip"))

        self.root = Sofa.Core.Node("root")
        cm.prepare_root(self.root, self.cfg)
        self.cable = cm.build_cable(self.root, self.cfg)
        Sofa.Simulation.init(self.root)

        self.coupling = GraspCoupling(self.cable) if self.table_mode else None
        self.get_logger().info(
            f"SOFA cable up [{self.role}]: L={self.cfg['length_m']} m, "
            f"EI={self.cfg['EI_Nm2']}, GJ={self.cfg['GJ_Nm2']}, "
            f"mode={'table' if self.table_mode else 'hanging'} -> {frames_topic}")

        # --- ROS interfaces ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.frames_pub = self.create_publisher(PoseArray, frames_topic, 10)
        self.markers_pub = self.create_publisher(CableMarkerArray, markers_topic, 10)
        # [compute_ms, backlog_steps, stepped]: real-time health of the plant (a rising
        # backlog = integration trails the clock), recorded alongside the control metrics.
        self.solver_pub = self.create_publisher(
            Float64MultiArray, f"/cable/{self.role}/solver_stats", 10)

        # The grasp is a property of the physical setup: only the truth plant
        # owns it, so the estimator can never influence the experiment.
        self.grasp_pub = None
        if self.table_mode and self.role == "truth":
            self.grasp_pub = self.create_publisher(
                GraspState, "/cable/grasp_state", 10)
            self._attach_srv = self.create_service(
                Trigger, "/cable/attach", self._on_attach)
            self._detach_srv = self.create_service(
                Trigger, "/cable/detach", self._on_detach)

        self._sim_ns = None
        self._first_grasp_ns = None
        self._logged_track = False
        rate = float(self.get_parameter("publish_rate_hz").value)
        # real-time clock (ibvs_world real_time_factor 1): 1/rate s of steps per callback
        required = math.ceil(1.0 / (rate * self.root.dt.value))
        self.max_steps = self.max_steps or 2 * required  # derived: 2x real-time catch-up
        if self.max_steps < required:
            raise ValueError(f"max_steps_per_cycle {self.max_steps} < {required} steps per callback "
                             f"needed at timestep {self.root.dt.value} s and {rate} Hz")
        self.timer = self.create_timer(1.0 / rate, self._cycle)

    # ------------------------------------------------------------------
    def _on_attach(self, _request, response):
        accepted = self.coupling.request_attach()
        response.success = accepted
        response.message = ("attach requested; waiting for the gripper to be in range"
                            if accepted else "already attached")
        return response

    def _on_detach(self, _request, response):
        released = self.coupling.request_detach()
        response.success = released
        response.message = "tip released" if released else "already detached"
        return response

    def _publish_grasp_state(self, stamp):
        msg = GraspState()
        msg.header.stamp = stamp
        msg.header.frame_id = self.base_frame
        msg.state = self.coupling.state
        msg.distance_to_tip_m = float(self.coupling.distance_to_tip)
        msg.height_above_plane_m = float(self.coupling.height_above_plane)
        msg.in_range = bool(self.coupling.in_range())
        self.grasp_pub.publish(msg)

    def _lookup(self, target, stamp_ns=None):
        """Pose of ``target`` in base_frame: latest, or interpolated by TF at ``stamp_ns``; None
        when TF cannot evaluate it (never extrapolated)."""
        stamp = Time() if stamp_ns is None else Time(nanoseconds=stamp_ns)
        try:
            tfm = self.tf_buffer.lookup_transform(self.base_frame, target, stamp)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        t, q = tfm.transform.translation, tfm.transform.rotation
        return [t.x, t.y, t.z, q.x, q.y, q.z, q.w]

    def _cycle(self):
        now = self.get_clock().now()
        if now.nanoseconds == 0:
            return  # sim clock not up yet
        self._advance(now.nanoseconds)

    def _advance(self, now_ns):
        """Integrate from the ROS time of the SOFA state (``_sim_ns``: set once by _start_epoch,
        then advanced by executed steps only) towards ``now_ns``, at most max_steps substeps. Each
        substep takes the gripper boundary TF interpolates over it; an interval TF cannot evaluate
        yet stays pending (backlog kept, never extrapolated)."""
        if self.table_mode and not self.coupling.fixture_set:
            fixture = self._lookup(self.fixture_frame)
            if fixture is None:
                return
            self.coupling.on_fixture(fixture)
            self.get_logger().info(
                f"cable clamped at {self.fixture_frame}: "
                f"[{fixture[0]:.3f} {fixture[1]:.3f} {fixture[2]:.3f}]")
        dt = self.root.dt.value
        dt_ns = round(dt * 1e9)
        if self._sim_ns is None:
            self._sim_ns = self._start_epoch(now_ns, dt_ns)
            if self._sim_ns is None:
                return
        stepped = 0
        started = time.perf_counter()
        while stepped < self.max_steps and self._sim_ns + dt_ns <= now_ns:
            start = self._lookup(self.grasp_frame, self._sim_ns)
            end = self._lookup(self.grasp_frame, self._sim_ns + dt_ns)
            if start is None or end is None:
                break
            self._apply_boundary(start, end, self._sim_ns + dt_ns, dt)
            self._sofa_sim.animate(self.root, dt)
            self._sim_ns += dt_ns
            stepped += 1
        compute_ms = (time.perf_counter() - started) * 1e3
        self.solver_pub.publish(Float64MultiArray(
            data=[compute_ms, float((now_ns - self._sim_ns) // dt_ns), float(stepped)]))
        self._publish(Time(nanoseconds=self._sim_ns).to_msg())

    def _start_epoch(self, now_ns, dt_ns):
        """ROS time of the SOFA state before any step: the stamp of the first gripper transform
        seen, once TF evaluates both ends of the first interval and it ends by ``now_ns``."""
        if self._first_grasp_ns is None:
            try:
                first = self.tf_buffer.lookup_transform(self.base_frame, self.grasp_frame, Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                return None
            self._first_grasp_ns = Time.from_msg(first.header.stamp).nanoseconds
        start = self._first_grasp_ns
        if self._lookup(self.grasp_frame, start) is None:
            self._first_grasp_ns = None  # aged out of the TF cache: pick the next stamp seen
            return None
        if start + dt_ns > now_ns or self._lookup(self.grasp_frame, start + dt_ns) is None:
            return None
        return start

    def _apply_boundary(self, start, end, end_ns, dt):
        """Gripper pose at the substep end and its twist over the substep, from one TF motion."""
        if not self.table_mode:
            self.cable.set_base_pose(end)
            if not self._logged_track:
                self._logged_track = True
                self.get_logger().info(
                    f"tracking {self.base_frame} -> {self.grasp_frame}")
            return
        twist = pose_delta(np.asarray(end), np.asarray(start)) / dt
        try:
            if self.coupling.update_grasp(end, end_ns * 1e-9, twist):
                self.get_logger().info("gripper latched onto the cable end")
        except IncompatibleGraspCommand as exc:
            self.coupling.hold()
            self.get_logger().error(
                f"gripper command rejected, end target held: {exc}", throttle_duration_sec=1.0)

    def _publish(self, stamp):
        poses = self.cable.frame_poses()

        pa = PoseArray()
        pa.header.stamp = stamp
        pa.header.frame_id = self.base_frame
        for p in poses:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, p[:3])
            (pose.orientation.x, pose.orientation.y,
             pose.orientation.z, pose.orientation.w) = map(float, p[3:7])
            pa.poses.append(pose)
        self.frames_pub.publish(pa)

        ma = CableMarkerArray()
        ma.header = pa.header
        ma.experiment_id = self.role
        for mid, (s, idx) in enumerate(
                zip(self.cfg["marker_s_over_l"], self.cable.marker_indices)):
            m = CableMarker()
            m.id = mid
            m.s_over_l = float(s)
            m.position.x, m.position.y, m.position.z = map(float, poses[idx][:3])
            m.confidence = 1.0
            m.valid = True
            ma.markers.append(m)
        self.markers_pub.publish(ma)

        if self.grasp_pub is not None:
            self._publish_grasp_state(stamp)


def main():
    rclpy.init()
    node = CableSofaNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
