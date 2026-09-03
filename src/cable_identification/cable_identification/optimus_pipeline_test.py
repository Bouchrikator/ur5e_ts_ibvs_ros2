"""Gazebo-free Gate I harness: scripted gripper TF + truth plant + synthetic markers
+ Optimus estimator node, judged on /cable/parameter_estimate.

Everything but the robot: the fixture TF is the URDF's table clamp, the gripper
TF starts at the straight cable tip (1 cm above the table, inside the attach
window), the truth is asked to attach, and the gripper then drags the tip along
a slow tangential/normal sweep inside the reachable disk. The estimator starts
at cable_estimator_initial.yaml (EI 0.006) and must approach cable_truth.yaml
(EI 0.010) using only the published topics.

    ros2 run cable_identification optimus_pipeline_test [--duration 90] [--tolerance 0.05]
"""

import argparse
import math
import os
import signal
import subprocess
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
from cable_msgs.msg import CableParameterEstimate, GraspState

FIXTURE = (-0.25, -0.35, 0.03, 0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))  # URDF clamp, yaw 90 deg
TRUE_EI = 0.010


class GripperDriver(Node):
    """Publishes cable_fixture_frame (static) and a scripted cable_grasp_frame."""

    def __init__(self, length_m):
        super().__init__("optimus_pipeline_test")
        self.tf = TransformBroadcaster(self)
        self.length = length_m
        self.t_attach = None
        self.grasp_state = None
        self.estimates = []      # (t, EI, std, innovation, rmse)
        self.create_subscription(GraspState, "/cable/grasp_state", self._on_grasp, 10)
        self.create_subscription(CableParameterEstimate, "/cable/parameter_estimate",
                                 self._on_estimate, 10)
        self.attach = self.create_client(Trigger, "/cable/attach")
        self.t0 = time.monotonic()
        self.create_timer(1.0 / 60.0, self._tick)

    def _on_grasp(self, msg):
        if msg.state == GraspState.ATTACHED and self.t_attach is None:
            self.t_attach = time.monotonic()
            self.get_logger().info("truth plant attached")
        self.grasp_state = msg.state

    def _on_estimate(self, msg):
        if msg.parameter_name == "EI":
            self.estimates.append((time.monotonic() - self.t0, msg.value, msg.std_dev,
                                   msg.innovation, msg.marker_rmse))

    def gripper_pose(self):
        # straight tip: fixture + L along the clamp's x axis (world +y for yaw 90 deg)
        tip = np.array([FIXTURE[0], FIXTURE[1] + self.length, FIXTURE[2]])
        if self.t_attach is None:
            offset = np.zeros(3)
        else:
            t = time.monotonic() - self.t_attach
            ramp = min(1.0, t / 5.0)   # ease in after the latch
            # lateral (world x) sweep + shortening along the cable (world y), inside 0.99 L
            offset = ramp * np.array([0.10 * math.sin(2 * math.pi * 0.10 * t),
                                      -0.06 - 0.04 * math.sin(2 * math.pi * 0.07 * t), 0.0])
        p = tip + offset
        return [p[0], p[1], FIXTURE[2] + 0.005, FIXTURE[3], FIXTURE[4], FIXTURE[5], FIXTURE[6]]

    def _tick(self):
        stamp = self.get_clock().now().to_msg()
        for child, pose in (("cable_fixture_frame", FIXTURE), ("cable_grasp_frame", self.gripper_pose())):
            tfm = TransformStamped()
            tfm.header.stamp = stamp
            tfm.header.frame_id = "base_link"
            tfm.child_frame_id = child
            (tfm.transform.translation.x, tfm.transform.translation.y,
             tfm.transform.translation.z) = map(float, pose[:3])
            (tfm.transform.rotation.x, tfm.transform.rotation.y,
             tfm.transform.rotation.z, tfm.transform.rotation.w) = map(float, pose[3:7])
            self.tf.sendTransform(tfm)


def spawn(cmd):
    # own process group: `ros2 run` exits on SIGTERM but its python child would survive
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                            start_new_session=True)


def stop(procs):
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)


def live_stack():
    """Names of pipeline processes already running (a second stack corrupts the test)."""
    out = subprocess.run(["pgrep", "-fa", "cable_sofa_node|cable_estimator_node|synthetic_marker_node"],
                         capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if "pgrep" not in line]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--duration", type=float, default=90.0, help="seconds after attach")
    ap.add_argument("--tolerance", type=float, default=0.05, help="relative EI tolerance")
    ap.add_argument("--noise", type=float, default=0.002)
    args = ap.parse_args(argv)

    from ament_index_python.packages import get_package_share_directory
    from cable_identification import cosserat_model as cm
    share = get_package_share_directory("cable_identification")
    truth_cfg = f"{share}/config/cable_truth.yaml"
    est_cfg = f"{share}/config/cable_estimator_initial.yaml"
    length = float(cm.load_config(truth_cfg)["length_m"])

    stale = live_stack()
    if stale:
        print("refusing to start: pipeline nodes already running:\n  " + "\n  ".join(stale))
        print("OPTIMUS_PIPELINE_TEST_FAILED")
        return 2

    procs = [
        spawn(["ros2", "run", "sofa_ros2_adapter", "cable_sofa_node", "--ros-args",
               "-p", "role:=truth", "-p", f"cable_config:={truth_cfg}", "-r", "__node:=cable_truth"]),
        spawn(["ros2", "run", "cable_perception", "synthetic_marker_node", "--ros-args",
               "-p", "truth_topic:=/cable/truth/markers", "-p", f"noise_std_m:={args.noise}"]),
        spawn(["ros2", "run", "sofa_ros2_adapter", "cable_estimator_node", "--ros-args",
               "-p", f"cable_config:={est_cfg}", "-p", "identified_parameters:=[EI]",
               "-p", f"measurement_std_m:={args.noise}", "-p", "update_rate_hz:=8.0"]),
    ]
    rclpy.init()
    node = GripperDriver(length)
    ok = False
    try:
        t0 = time.monotonic()
        while rclpy.ok() and node.grasp_state is None and time.monotonic() - t0 < 60:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.grasp_state is None:
            print("truth plant never published /cable/grasp_state")
        else:
            while not node.attach.wait_for_service(timeout_sec=1.0):
                rclpy.spin_once(node, timeout_sec=0.1)
            future = node.attach.call_async(Trigger.Request())
            while rclpy.ok() and not future.done():
                rclpy.spin_once(node, timeout_sec=0.1)
            t0 = time.monotonic()
            while rclpy.ok() and node.t_attach is None and time.monotonic() - t0 < 30:
                rclpy.spin_once(node, timeout_sec=0.1)
            if node.t_attach is None:
                print("truth plant did not latch (gripper out of range?)")
            else:
                while rclpy.ok() and time.monotonic() - node.t_attach < args.duration:
                    rclpy.spin_once(node, timeout_sec=0.1)
                est = np.array(node.estimates)
                if len(est) == 0:
                    print("no /cable/parameter_estimate received")
                else:
                    print(f"{len(est)} estimates; EI trace (t[s], EI): " + ", ".join(
                        f"({t:.0f}, {v:.5f})" for t, v in est[::max(1, len(est) // 12), :2]))
                    final = float(np.median(est[-10:, 1]))
                    err = abs(final - TRUE_EI) / TRUE_EI
                    corrected = est[np.isfinite(est[:, 3])]
                    print(f"final EI (median of last 10) = {final:.6f}, truth {TRUE_EI}, error {100 * err:.2f} %; "
                          f"{len(corrected)}/{len(est)} steps corrected; last std {est[-1, 2]:.3g}; "
                          f"median innovation {1e3 * np.nanmedian(corrected[:, 3]) if len(corrected) else float('nan'):.2f} mm")
                    ok = err < args.tolerance and len(corrected) > 10
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        stop(procs)
    print("OPTIMUS_PIPELINE_TEST_" + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
