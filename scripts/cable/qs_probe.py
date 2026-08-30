"""Post-latch probe: log fore/bearing (TF), q, u, V once per second.
Decides between (a) controller feedback pushing the gripper out of the box
and (b) an external mover / measurement drift with u ~ 0."""
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64MultiArray
from cable_msgs.msg import CableState
import tf2_ros

DURATION_S = 45.0


class Probe(Node):
    def __init__(self):
        super().__init__("qs_probe")
        self.buf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buf, self)
        self.q = None
        self.prem = None
        self.u = None
        self.V = None
        self.create_subscription(CableState, "/cable/reduced_state", self.on_s, 10)
        self.create_subscription(TwistStamped, "/cable/desired_gripper_twist", self.on_u, 10)
        self.create_subscription(Float64MultiArray, "/cable/lyapunov", self.on_l, 10)

    def on_s(self, m):
        if m.valid:
            self.q = np.asarray(m.state[:3])
            self.prem = np.asarray(m.state[6:8])

    def on_u(self, m):
        self.u = (m.twist.linear.x, m.twist.linear.y)

    def on_l(self, m):
        self.V = (m.data[0], m.data[2])

    def fore(self):
        try:
            g = self.buf.lookup_transform(
                "cable_fixture_frame", "cable_grasp_frame", rclpy.time.Time())
            p = np.hypot(g.transform.translation.x, g.transform.translation.y)
            return 1.0 - p / 0.7
        except Exception:
            return float("nan")


def main():
    rclpy.init()
    n = Probe()
    t0 = time.time()
    last = t0
    print("t  fore_tf  prem(fore,bear)  q  u(mm/s)  V,err")
    while time.time() - t0 < DURATION_S:
        rclpy.spin_once(n, timeout_sec=0.05)
        if time.time() - last >= 1.0:
            last = time.time()
            q = None if n.q is None else np.round(n.q, 4).tolist()
            prem = None if n.prem is None else np.round(n.prem, 4).tolist()
            u = None if n.u is None else [round(1e3 * v, 3) for v in n.u]
            V = None if n.V is None else [f"{n.V[0]:.3e}", f"{n.V[1]:.4f}"]
            print(f"{time.time()-t0:5.1f}  {n.fore():+.4f}  {prem}  {q}  {u}  {V}",
                  flush=True)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
