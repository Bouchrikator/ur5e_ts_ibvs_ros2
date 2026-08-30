"""Bridge the certified outer TS twist to MoveIt Servo while the visual
inner loop has no marker lock (wrist faces away from the overview camera).

Envelope-keeping supervisor: inside the certified premise box the certified
twist passes through; outside it (where the gains are meaningless) a gentle
TF go-to-reference twist steers back into the box, with hysteresis."""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
import tf2_ros
import numpy as np

# Certified premise box (cable_qs_gains_narrow.yaml) and re-entry hysteresis.
# Re-entry deep in the box: the no-g-feedback law drifts taut-ward (audit 9.1),
# so handing back at the edge chatters at ~3 s period.
FORE_LO, FORE_HI = 0.06, 0.11
BEAR_LIM = 0.25
REENTRY_FORE = (0.080, 0.095)
REENTRY_BEAR = 0.10
G_REF = np.array([0.64055, -0.01286])  # modal basis boundary_reference
RECOVER_SPEED = 0.01  # m/s, envelope-recovery go-to-ref


def quat_to_rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class TwistRelay(Node):
    def __init__(self):
        super().__init__("cable_twist_relay")
        self.buf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buf, self)
        self.pub = self.create_publisher(
            TwistStamped, "/servo_node/delta_twist_cmds", 10)
        self.sub = self.create_subscription(
            TwistStamped, "/cable/desired_gripper_twist", self.cb, 10)
        self.n = 0
        self.recovering = False

    def cb(self, msg):
        v = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        try:
            tfm = self.buf.lookup_transform(
                "base_link", msg.header.frame_id, rclpy.time.Time())
            bf = self.buf.lookup_transform(
                "base_link", "cable_fixture_frame", rclpy.time.Time())
            g = self.buf.lookup_transform(
                "cable_fixture_frame", "cable_grasp_frame", rclpy.time.Time())
        except Exception:
            return
        p = np.array([g.transform.translation.x, g.transform.translation.y])
        fore = 1.0 - np.linalg.norm(p) / 0.7
        bearing = float(np.arctan2(p[1], p[0]))
        inside = FORE_LO <= fore <= FORE_HI and abs(bearing) <= BEAR_LIM
        re_entered = (REENTRY_FORE[0] <= fore <= REENTRY_FORE[1]
                      and abs(bearing) <= REENTRY_BEAR)
        if self.recovering and re_entered:
            self.recovering = False
            self.get_logger().info(
                f"re-entered certified box (fore={fore:.4f}); resuming certified law")
        elif not self.recovering and not inside:
            self.recovering = True
            self.get_logger().warn(
                f"fore={fore:.4f} bearing={bearing:+.4f} outside certified box; "
                "RECOVERY: steering back to the basis reference")
        if self.recovering:
            err = G_REF - p
            d = np.linalg.norm(err)
            v_fix = RECOVER_SPEED * err / d if d > 1e-6 else np.zeros(2)
            v = np.array([v_fix[0], v_fix[1], 0.0])
            vb = quat_to_rot(bf.transform.rotation) @ v
        else:
            vb = quat_to_rot(tfm.transform.rotation) @ v
        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "base_link"
        out.twist.linear.x, out.twist.linear.y, out.twist.linear.z = vb
        self.pub.publish(out)
        self.n += 1
        if self.n % 100 == 1:
            self.get_logger().info(
                f"relay #{self.n}: fixture v=({v[0]:+.4f},{v[1]:+.4f}) -> "
                f"base v=({vb[0]:+.4f},{vb[1]:+.4f},{vb[2]:+.4f})")


def main():
    rclpy.init()
    rclpy.spin(TwistRelay())


if __name__ == "__main__":
    main()
