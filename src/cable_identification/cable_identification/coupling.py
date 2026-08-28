"""Grasp coupling for the table-shaping setup (shared by node and GUI scene).

Cable base = table fixture (static). Cable tip = bilaterally constrained to a
kinematic target. The target stays at the straight tip until the gripper
comes within `attach_distance_m` of it, then latches and ramps to the gripper
pose over `attach_ramp_s` (seamless: orientation offset captured at latch),
after which it tracks the gripper 1:1.
"""

import math


def _q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def _q_conj(q):
    return (-q[0], -q[1], -q[2], q[3])


def _q_slerp(a, b, t):
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0.0:
        b = tuple(-x for x in b)
        dot = -dot
    if dot > 0.9995:
        out = [x + t * (y - x) for x, y in zip(a, b)]
    else:
        th = math.acos(max(-1.0, min(1.0, dot)))
        sa, sb = math.sin((1 - t) * th), math.sin(t * th)
        out = [(sa * x + sb * y) / math.sin(th) for x, y in zip(a, b)]
    n = math.sqrt(sum(x * x for x in out)) or 1.0
    return tuple(x / n for x in out)


def _q_rot(q, v):
    qv = _q_mul(_q_mul(q, (v[0], v[1], v[2], 0.0)), _q_conj(q))
    return (qv[0], qv[1], qv[2])


class GraspCoupling:
    def __init__(self, cable):
        self.cable = cable
        self.attach_distance = float(cable.cfg.get("attach_distance_m", 0.15))
        self.attach_height = float(cable.cfg.get("attach_height_m", 0.01))
        self.ramp_s = float(cable.cfg.get("attach_ramp_s", 0.5))
        self.fixture_set = False
        self.latched = False
        self._fixture_z = None
        self._t_latch = None
        self._tip0 = None      # tip pose at latch
        self._q_offset = None  # grasp_q^-1 * tip_q at latch (seamless)

    def on_fixture(self, pose7):
        """Relocate the (still straight) cable onto the fixture, once.

        The grasp target is set to the ANALYTIC straight tip (fixture pose
        composed with [L,0,0]) — the mapped frames are stale until the next
        physics step, so reading them here would yank the cable violently.
        """
        self.cable.set_base_pose(list(pose7))
        L = float(self.cable.cfg["length_m"])
        q = tuple(pose7[3:7])
        d = _q_rot(q, (L, 0.0, 0.0))
        tip = [pose7[0] + d[0], pose7[1] + d[1], pose7[2] + d[2],
               q[0], q[1], q[2], q[3]]
        self.cable.set_grasp_pose(tip)
        self._fixture_z = float(pose7[2])
        self.fixture_set = True

    def sync_target_to_tip(self):
        self.cable.set_grasp_pose(self.cable.tip_pose())

    def update_grasp(self, grasp_pose7, now_s):
        """Feed the current gripper pose; returns True on the latch edge."""
        if not self.fixture_set:
            return False
        tip = self.cable.tip_pose()

        if not self.latched:
            # snap only when the gripper is at table height (within
            # attach_height of the cable plane) AND near the free end
            dxy = math.dist(grasp_pose7[:2], tip[:2])
            dz = grasp_pose7[2] - self._fixture_z
            if dxy > self.attach_distance or dz > self.attach_height or dz < -0.05:
                return False
            self.latched = True
            self._t_latch = now_s
            self._tip0 = tip
            gq = tuple(grasp_pose7[3:7])
            self._q_offset = _q_mul(_q_conj(gq), tuple(tip[3:7]))
            return True

        a = 1.0 if self.ramp_s <= 0 else min(
            1.0, (now_s - self._t_latch) / self.ramp_s)
        gq = tuple(grasp_pose7[3:7])
        goal_q = _q_mul(gq, self._q_offset)
        pos = [self._tip0[i] + a * (grasp_pose7[i] - self._tip0[i]) for i in range(3)]
        quat = _q_slerp(tuple(self._tip0[3:7]), goal_q, a)
        self.cable.set_grasp_pose(pos + list(quat))
        return False
