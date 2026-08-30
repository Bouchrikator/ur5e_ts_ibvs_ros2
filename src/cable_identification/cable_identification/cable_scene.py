"""runSofa GUI scene for the ROS-coupled cable.

Loads the same Cosserat cable as cable_sofa_node, but inside the runSofa GUI:
a scene controller drives the base from TF (base_link -> cable_grasp_frame)
and publishes /cable/sofa_frames + /cable/predicted_markers, so the Gazebo
visual cable stays in sync while you inspect the physics in SOFA.

Run (inside the container, sim already up with sofa_gui:=true):
    runSofa <this file>
Without ROS running it degrades to a hanging cable under gravity.
"""

import os

import Sofa
import Sofa.Core

from cable_identification import cosserat_model as cm


def _default_config_path():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("cable_identification"),
                            "config", "cable_initial.yaml")
    except Exception:
        return None


class RosCouplingController(Sofa.Core.Controller):
    """Per-step: TF -> cable boundaries; frames/markers -> ROS topics."""

    def __init__(self, cable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "RosCouplingController"
        self.cable = cable
        self.ros_ok = False
        from cable_identification.coupling import GraspCoupling
        self.coupling = GraspCoupling(cable) if cable.cfg.get("grasp_tip") else None
        try:
            import rclpy
            from rclpy.node import Node as RclpyNode
            import tf2_ros
            from geometry_msgs.msg import PoseArray, Pose
            from cable_msgs.msg import CableMarker, CableMarkerArray

            self._PoseArray, self._Pose = PoseArray, Pose
            self._CableMarker, self._CableMarkerArray = CableMarker, CableMarkerArray
            self._tf2 = tf2_ros

            # runSofa evaluates createScene more than once per session
            if not rclpy.ok():
                rclpy.init()
            self.node = RclpyNode(
                f"cable_sofa_gui_{os.getpid()}_{id(self) % 10000}",
                parameter_overrides=[rclpy.parameter.Parameter(
                    "use_sim_time", rclpy.Parameter.Type.BOOL, True)])
            self.tf_buffer = tf2_ros.Buffer()
            # spin_thread: TF fills in the background, SOFA thread just reads
            self.tf_listener = tf2_ros.TransformListener(
                self.tf_buffer, self.node, spin_thread=True)
            self.frames_pub = self.node.create_publisher(
                PoseArray, "/cable/sofa_frames", 10)
            self.markers_pub = self.node.create_publisher(
                CableMarkerArray, "/cable/predicted_markers", 10)
            self.ros_ok = True
            print("[cable_scene] ROS coupling active (TF + publishers)")
        except Exception as exc:  # ROS absent -> standalone GUI physics
            print(f"[cable_scene] ROS unavailable, standalone mode: {exc}")

    def _lookup(self, target):
        from rclpy.time import Time
        try:
            tfm = self.tf_buffer.lookup_transform("base_link", target, Time())
            t, q = tfm.transform.translation, tfm.transform.rotation
            return [t.x, t.y, t.z, q.x, q.y, q.z, q.w]
        except Exception:
            return None

    def onAnimateBeginEvent(self, _):
        if not self.ros_ok:
            return
        if self.coupling is not None:  # table shaping mode
            if not self.coupling.fixture_set:
                f = self._lookup("cable_fixture_frame")
                if f is None:
                    return
                self.coupling.on_fixture(f)
                print("[cable_scene] cable clamped at cable_fixture_frame")
            g = self._lookup("cable_grasp_frame")
            now_s = self.node.get_clock().now().nanoseconds * 1e-9
            if g is not None and self.coupling.update_grasp(g, now_s):
                print("[cable_scene] gripper latched onto the cable end")
        else:  # hanging mode: gripper drives the base
            g = self._lookup("cable_grasp_frame")
            if g is not None:
                self.cable.set_base_pose(g)

    def onAnimateEndEvent(self, _):
        if not self.ros_ok:
            return
        stamp = self.node.get_clock().now().to_msg()
        poses = self.cable.frame_poses()

        pa = self._PoseArray()
        pa.header.stamp = stamp
        pa.header.frame_id = "base_link"
        for p in poses:
            pose = self._Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, p[:3])
            (pose.orientation.x, pose.orientation.y,
             pose.orientation.z, pose.orientation.w) = map(float, p[3:7])
            pa.poses.append(pose)
        self.frames_pub.publish(pa)

        ma = self._CableMarkerArray()
        ma.header = pa.header
        for mid, (s, idx) in enumerate(zip(self.cable.cfg["marker_s_over_l"],
                                           self.cable.marker_indices)):
            m = self._CableMarker()
            m.id = mid
            m.s_over_l = float(s)
            m.position.x, m.position.y, m.position.z = map(float, poses[idx][:3])
            m.confidence = 1.0
            m.valid = True
            ma.markers.append(m)
        self.markers_pub.publish(ma)


def createScene(root):
    cfg = cm.load_config(os.environ.get("CABLE_CONFIG") or _default_config_path())

    cm.prepare_root(root, cfg)
    root.addObject("RequiredPlugin", pluginName=["Sofa.Component.Visual"])
    root.addObject("VisualStyle",
                   displayFlags="showBehaviorModels showMechanicalMappings")

    cable = cm.build_cable(root, cfg, show=True)
    root.addObject(RosCouplingController(cable))
    return root
