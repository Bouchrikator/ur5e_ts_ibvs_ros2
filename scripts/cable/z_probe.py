"""Measure the CERTIFIED coordinate z = U^T (q - q*) through a pulse:
the certificate claims z -> 0, not full-state error -> 0."""
import time
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64MultiArray
import tf2_ros

g = yaml.safe_load(open("/ros2_ws/artifacts/cable_qs_gains_narrow.yaml"))
U = np.asarray(g["quasi_static"]["output_map_u"], dtype=float)  # 3x2


class ZProbe(Node):
    def __init__(self):
        super().__init__("z_probe")
        self.buf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buf, self)
        self.pub = self.create_publisher(
            TwistStamped, "/servo_node/delta_twist_cmds", 10)
        self.err = None
        self.create_subscription(
            Float64MultiArray, "/cable/shape_error", self.on_e, 10)

    def on_e(self, msg):
        self.err = np.asarray(msg.data[:8])

    def z(self):
        if self.err is None:
            return None
        return U.T @ self.err[:3]

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def spin_for(self, dur, pulse=None):
        t0 = self.now()
        while self.now() - t0 < dur:
            rclpy.spin_once(self, timeout_sec=0.02)
            if pulse is not None:
                try:
                    b = self.buf.lookup_transform(
                        "base_link", "cable_fixture_frame", rclpy.time.Time())
                except Exception:
                    continue
                q = b.transform.rotation
                x, y, zz, w = q.x, q.y, q.z, q.w
                R = np.array([
                    [1 - 2 * (y * y + zz * zz), 2 * (x * y - zz * w)],
                    [2 * (x * y + zz * w), 1 - 2 * (x * x + zz * zz)]])
                vb = R @ np.asarray(pulse)
                out = TwistStamped()
                out.header.stamp = self.get_clock().now().to_msg()
                out.header.frame_id = "base_link"
                out.twist.linear.x, out.twist.linear.y = float(vb[0]), float(vb[1])
                self.pub.publish(out)
                time.sleep(0.03)

    def sample(self, label):
        z = self.z()
        if z is None:
            print(f"[{label:12s}] no data", flush=True)
            return
        print(f"[{label:12s}] ||z||={np.linalg.norm(z):.6f} "
              f"z=({z[0]:+.5f},{z[1]:+.5f})", flush=True)


def main():
    rclpy.init()
    n = ZProbe()
    n.spin_for(3.0)
    n.sample("baseline")
    n.spin_for(1.5, pulse=(0.0, 0.015))
    n.sample("pulse end")
    for k in range(8):
        n.spin_for(2.0)
        n.sample(f"recover +{2*(k+1)}s")


if __name__ == "__main__":
    main()
