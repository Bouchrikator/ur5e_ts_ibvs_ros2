"""Disturbance-rejection experiment: tangential pulse, then record recovery."""
import time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64MultiArray
import tf2_ros


class Experiment(Node):
    def __init__(self):
        super().__init__("disturbance_experiment")
        self.buf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buf, self)
        self.pub = self.create_publisher(
            TwistStamped, "/servo_node/delta_twist_cmds", 10)
        self.err = None
        self.lyap = None
        self.create_subscription(
            Float64MultiArray, "/cable/shape_error", self.on_e, 10)
        self.create_subscription(
            Float64MultiArray, "/cable/lyapunov", self.on_l, 10)

    def on_e(self, msg):
        self.err = msg.data[-1]

    def on_l(self, msg):
        self.lyap = (msg.data[0], msg.data[3])

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
                x, y, z, w = q.x, q.y, q.z, q.w
                R = np.array([
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z)]])
                vb = R @ np.asarray(pulse)
                out = TwistStamped()
                out.header.stamp = self.get_clock().now().to_msg()
                out.header.frame_id = "base_link"
                out.twist.linear.x, out.twist.linear.y = float(vb[0]), float(vb[1])
                self.pub.publish(out)
                time.sleep(0.03)

    def sample(self, label):
        e = f"{self.err:.5f}" if self.err is not None else "n/a"
        v = f"{self.lyap[0]:.3e} dec={int(self.lyap[1])}" if self.lyap else "n/a"
        print(f"[{label:12s}] err={e} V={v}", flush=True)


def main():
    rclpy.init()
    node = Experiment()
    node.spin_for(2.0)
    node.sample("baseline")
    # 2 s tangential pulse in the fixture frame: bearing disturbance, in-box
    node.spin_for(2.0, pulse=(0.0, 0.025))
    node.sample("pulse end")
    for k in range(10):
        node.spin_for(2.0)
        node.sample(f"recover +{2*(k+1)}s")


if __name__ == "__main__":
    main()
