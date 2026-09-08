"""Mission sequencing for the cable shaping experiment.

Step 10 of the plan. The supervisor owns the ORDER of things — approach, grasp,
shape, hold — and nothing else: it does not compute control and does not touch
the physics. Keeping the sequencing separate is what lets the same controllers
run in the direct and cascade architectures unchanged.

States: INITIALIZE -> APPROACH -> GRASP -> SHAPE -> HOLD, with FAULT reachable
from anywhere.
"""

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from std_srvs.srv import Trigger

from cable_msgs.msg import CableShapeTarget, CableState, GraspState

INITIALIZE = "INITIALIZE"
APPROACH = "APPROACH"
GRASP = "GRASP"
SHAPE = "SHAPE"
HOLD = "HOLD"
FAULT = "FAULT"


class CableSupervisorNode(Node):
    def __init__(self):
        super().__init__("cable_supervisor_node")

        self.declare_parameter("state_topic", "/cable/reduced_state")
        self.declare_parameter("grasp_state_topic", "/cable/grasp_state")
        self.declare_parameter("target_topic", "/cable/target_shape")
        self.declare_parameter("status_topic", "/cable/supervisor_state")
        self.declare_parameter("target_modal_coordinates", [0.0, 0.0, 0.0])
        self.declare_parameter("target_file", "")
        self.declare_parameter("shape_tolerance_m", 0.010)
        self.declare_parameter("hold_time_s", 2.0)
        self.declare_parameter("state_timeout_s", 3.0)
        self.declare_parameter("recover_time_s", 3.0)
        self.declare_parameter("attach_service", "/cable/attach")
        self.declare_parameter("auto_start", True)

        self.target = np.asarray(
            self.get_parameter("target_modal_coordinates").value, dtype=float)
        self.target_name = "supervisor"
        self.tolerance = float(self.get_parameter("shape_tolerance_m").value)
        self.hold_time = float(self.get_parameter("hold_time_s").value)
        self.state_timeout = float(self.get_parameter("state_timeout_s").value)
        self.recover_time = float(self.get_parameter("recover_time_s").value)
        self._load_target_file()

        self.state = INITIALIZE
        self._reduced_state = None
        self._reduced_stamp = None
        self._grasp = None
        self._attach_pending = False
        self._settled_since = None
        self._recover_since = None

        self.status_pub = self.create_publisher(
            String, self.get_parameter("status_topic").value, 10)
        self.target_pub = self.create_publisher(
            CableShapeTarget, self.get_parameter("target_topic").value, 10)

        self.create_subscription(
            CableState, self.get_parameter("state_topic").value,
            self._on_state, 10)
        self.create_subscription(
            GraspState, self.get_parameter("grasp_state_topic").value,
            self._on_grasp, 10)
        # Adopt externally latched targets (e.g. a settled-equilibrium latch)
        # instead of clobbering them with the configured one every tick.
        self.create_subscription(
            CableShapeTarget, self.get_parameter("target_topic").value,
            self._on_external_target, 10)

        self.attach_client = self.create_client(
            Trigger, self.get_parameter("attach_service").value)

        self.timer = self.create_timer(0.1, self._tick)
        self.get_logger().info(
            f"cable supervisor: target q*={self.target.tolist()}, "
            f"tolerance={self.tolerance * 1e3:.1f} mm")

    # ------------------------------------------------------------------
    def _load_target_file(self):
        """A target file, when given, defines the experiment and wins."""
        path = self.get_parameter("target_file").value
        if not path:
            return
        with open(path, "r") as handle:
            data = yaml.safe_load(handle) or {}
        section = data.get("cable_target", data)
        if "modal_coordinates" in section:
            self.target = np.asarray(section["modal_coordinates"], dtype=float)
        self.tolerance = float(section.get("shape_tolerance_m", self.tolerance))
        self.hold_time = float(section.get("hold_time_s", self.hold_time))
        self.get_logger().info(f"target loaded from {path}")

    def _now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_state(self, msg):
        if not msg.valid:
            return
        self._reduced_state = np.asarray(msg.modal_coordinates, dtype=float)
        self._reduced_stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9

    def _on_grasp(self, msg):
        self._grasp = msg

    def _on_external_target(self, msg):
        coords = np.asarray(msg.modal_coordinates, dtype=float)
        if coords.shape != self.target.shape:
            return
        # Own re-publications carry the adopted name AND coordinates.
        if (msg.target_name == self.target_name
                and np.allclose(coords, self.target, atol=1e-12)):
            return
        self.target = coords
        self.target_name = msg.target_name
        self._settled_since = None
        self.get_logger().info(
            f"adopted external target '{msg.target_name}': "
            f"{np.round(coords, 5).tolist()}")

    def _transition(self, new_state, reason):
        if new_state == self.state:
            return
        self.get_logger().info(f"{self.state} -> {new_state}: {reason}")
        self.state = new_state

    def _publish_target(self):
        msg = CableShapeTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.target_name = self.target_name
        msg.modal_coordinates = self.target.tolist()
        self.target_pub.publish(msg)

    def _request_attach(self):
        if self._attach_pending:
            return
        if not self.attach_client.service_is_ready():
            self.get_logger().warn("attach service not available yet",
                                   throttle_duration_sec=5.0)
            return
        self._attach_pending = True
        future = self.attach_client.call_async(Trigger.Request())
        future.add_done_callback(self._on_attach_response)

    def _on_attach_response(self, future):
        self._attach_pending = False
        try:
            response = future.result()
        except Exception as error:  # service call failed outright
            self._transition(FAULT, f"attach request failed: {error}")
            return
        self.get_logger().info(f"attach request: {response.message}")

    def _shape_error(self):
        if self._reduced_state is None:
            return None
        # A [a*, g*] target: the shape error is on the modal block only.
        return float(np.linalg.norm(self._reduced_state - self.target[:self._reduced_state.size]))

    def _tick(self):
        self.status_pub.publish(String(data=self.state))

        stale = (self._reduced_stamp is None
                 or self._now_s() - self._reduced_stamp > self.state_timeout)

        if self.state == FAULT:
            # Recoverable: occlusion dropouts are transient (audit 9.1); a
            # terminal FAULT forced operators to kill the supervisor.
            if stale or self._grasp is None:
                self._recover_since = None
                return
            if self._recover_since is None:
                self._recover_since = self._now_s()
            elif self._now_s() - self._recover_since >= self.recover_time:
                self._recover_since = None
                self._transition(INITIALIZE, "reduced state healthy again")
            return

        if self.state == INITIALIZE:
            if not stale and self._grasp is not None:
                self._transition(APPROACH, "reduced state and grasp state alive")
            return

        if stale:
            self._transition(FAULT, "reduced cable state lost")
            return

        if self.state == APPROACH:
            if self._grasp.state == GraspState.ATTACHED:
                self._transition(GRASP, "gripper already holding the cable")
            elif self._grasp.in_range:
                self._request_attach()
                self._transition(GRASP, "gripper in range; attach requested")
            return

        if self.state == GRASP:
            if self._grasp.state == GraspState.ATTACHED:
                self._transition(SHAPE, "cable attached")
            return

        # SHAPE and HOLD both drive the same target; only the report differs.
        self._publish_target()
        error = self._shape_error()

        if self._grasp.state != GraspState.ATTACHED:
            self._transition(FAULT, "cable released while shaping")
            return

        if self.state == SHAPE:
            if error is not None and error <= self.tolerance:
                if self._settled_since is None:
                    self._settled_since = self._now_s()
                elif self._now_s() - self._settled_since >= self.hold_time:
                    self._transition(HOLD, f"shape error {error * 1e3:.1f} mm held")
            else:
                self._settled_since = None
            return

        if self.state == HOLD and error is not None and error > 3.0 * self.tolerance:
            self._settled_since = None
            self._transition(SHAPE, f"shape drifted to {error * 1e3:.1f} mm")


def main():
    rclpy.init()
    node = CableSupervisorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
