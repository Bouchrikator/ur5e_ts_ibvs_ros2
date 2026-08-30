"""Approach v3: TF go-to-point to the basis boundary reference in the fixture
frame (mid premise box), settle, then latch settled q as the shape target."""
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from cable_msgs.msg import CableState, CableShapeTarget
import tf2_ros

G_REF = np.array([0.64055, -0.01286])  # modal basis boundary_reference
TOL = 0.004
SPEED = 0.02
SETTLE_TICKS = 100  # 4 s at the reference before testing stationarity
# A premature latch is NOT an equilibrium: the zero-g-feedback law then chases
# it out of the premise box. Require q stationary over a window, latch the mean.
Q_WINDOW_S = 10.0     # stationarity window
Q_PTP = 0.004         # max peak-to-peak per modal coordinate over the window
Q_MIN_SAMPLES = 60
STATIONARY_TIMEOUT_S = 180.0  # latch the mean anyway (with a warning) after this


class Approach3(Node):
    def __init__(self):
        super().__init__("cable_approach3")
        self.buf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buf, self)
        self.pub = self.create_publisher(
            TwistStamped, "/servo_node/delta_twist_cmds", 10)
        self.target_pub = self.create_publisher(
            CableShapeTarget, "/cable/target_shape", 10)
        self.q = None
        self.q_valid_stamp = -1.0
        self.q_hist = deque()  # (t, q) valid samples, pruned to Q_WINDOW_S
        self.settled_since = None
        self.create_subscription(
            CableState, "/cable/reduced_state", self.on_s, 10)
        self.timer = self.create_timer(0.04, self.tick)
        self.done = False
        self.settle = 0
        self._n = 0

    def on_s(self, msg):
        if msg.valid:
            self.q = np.asarray(msg.state[:3])
            now = self.get_clock().now().nanoseconds * 1e-9
            self.q_valid_stamp = now
            self.q_hist.append((now, self.q))
            while self.q_hist and now - self.q_hist[0][0] > Q_WINDOW_S:
                self.q_hist.popleft()

    def _stationary_mean(self, now):
        """Window mean if q is stationary (ptp < Q_PTP per coord), else None."""
        if len(self.q_hist) < Q_MIN_SAMPLES:
            return None
        if now - self.q_hist[0][0] < 0.8 * Q_WINDOW_S:
            return None
        qs = np.array([q for _, q in self.q_hist])
        ptp = qs.max(axis=0) - qs.min(axis=0)
        if np.all(ptp < Q_PTP):
            return qs.mean(axis=0)
        if self._n % 100 == 0:
            self.get_logger().info(f"waiting for stationarity: ptp={np.round(ptp, 5).tolist()}")
        return None

    def tick(self):
        try:
            g = self.buf.lookup_transform(
                "cable_fixture_frame", "cable_grasp_frame", rclpy.time.Time())
            b = self.buf.lookup_transform(
                "base_link", "cable_fixture_frame", rclpy.time.Time())
        except Exception:
            return
        p = np.array([g.transform.translation.x, g.transform.translation.y])
        err = G_REF - p
        dist = float(np.linalg.norm(err))
        fore = 1.0 - np.linalg.norm(p) / 0.7
        if dist < TOL:
            self.settle += 1
            self._n += 1
            self.pub.publish(self._zero())
            if self.settle < SETTLE_TICKS:
                return
            now = self.get_clock().now().nanoseconds * 1e-9
            if self.settled_since is None:
                self.settled_since = now
                self.q_hist.clear()  # only average samples taken at rest
                self.get_logger().info(
                    "at reference; waiting for the modal state to become stationary")
            if self.q is None or now - self.q_valid_stamp > 1.0:
                if self._n % 100 == 0:
                    self.get_logger().warn("settled but no fresh valid q; waiting")
                return
            q_eq = self._stationary_mean(now)
            if (q_eq is None and now - self.settled_since > STATIONARY_TIMEOUT_S
                    and len(self.q_hist) >= Q_MIN_SAMPLES):
                q_eq = np.array([q for _, q in self.q_hist]).mean(axis=0)
                self.get_logger().warn(
                    "stationarity timeout; latching the window mean anyway")
            if q_eq is None:
                return
            t = CableShapeTarget()
            t.header.stamp = self.get_clock().now().to_msg()
            t.target_name = "equilibrium_latch"
            t.modal_coordinates = [float(v) for v in q_eq]
            for _ in range(3):
                self.target_pub.publish(t)
            self.get_logger().info(
                f"settled at |p|={np.linalg.norm(p):.4f} fore={fore:.4f} "
                f"bearing={np.arctan2(p[1], p[0]):+.4f}; latched stationary q_eq="
                f"{np.round(q_eq, 5).tolist()}")
            self.done = True
            return
        self.settle = 0
        self.settled_since = None
        v_fix = SPEED * err / dist if dist > 0.02 else 1.0 * err
        q = b.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z)],
        ])
        v_base = R @ v_fix
        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "base_link"
        out.twist.linear.x, out.twist.linear.y = float(v_base[0]), float(v_base[1])
        self.pub.publish(out)
        self._n += 1
        if self._n % 100 == 1:
            self.get_logger().info(
                f"|p|={np.linalg.norm(p):.4f} fore={fore:.4f} dist_to_ref={dist:.4f}")

    def _zero(self):
        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "base_link"
        return out


def main():
    rclpy.init()
    node = Approach3()
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.get_logger().info("approach3 complete")


if __name__ == "__main__":
    main()
