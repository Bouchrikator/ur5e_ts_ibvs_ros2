"""Grasp coupling for the table-shaping setup (shared by node and GUI scene).

Cable base = table fixture (static). Cable tip = bilaterally constrained to a
kinematic target. The target stays at the straight tip until the grasp is
engaged, then ramps to the gripper pose over `attach_ramp_s` (seamless:
orientation offset captured at latch), after which it tracks the gripper 1:1.

Two engagement policies, selected by `attach_mode`:

  "proximity"  legacy: latches by itself as soon as the gripper is in range.
  "explicit"   an external supervisor calls request_attach(); the latch only
               fires once the geometric condition is ALSO satisfied. This is
               the deterministic policy used for identification experiments.

Only the latch DECISION differs between the two; the ramp/tracking mechanics
are shared.
"""

import math

DETACHED = 0
REQUESTED = 1
ATTACHED = 2


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
    def __init__(self, cable, attach_mode=None):
        self.cable = cable
        self.attach_distance = float(cable.cfg.get("attach_distance_m", 0.15))
        self.attach_height = float(cable.cfg.get("attach_height_m", 0.01))
        self.ramp_s = float(cable.cfg.get("attach_ramp_s", 0.5))
        self.attach_mode = attach_mode or cable.cfg.get("attach_mode", "proximity")
        self.fixture_set = False
        self.state = DETACHED
        self._fixture_z = None
        self._fixture_xy = None
        self._t_latch = None
        self._tip0 = None      # tip pose at latch
        self._q_offset = None  # grasp_q^-1 * tip_q at latch (seamless)
        self._distance = float("inf")
        self._height = float("inf")
        # Detached = free tip. The spring only exists while the gripper holds
        # the cable: left engaged it slowly pumps energy into the rod (its
        # geometric stiffness on the mapped frames is invisible to the
        # implicit solver) until the cable coils up.
        self.cable.set_grasp_spring_enabled(False)

    @property
    def latched(self):
        return self.state == ATTACHED

    @property
    def distance_to_tip(self):
        return self._distance

    @property
    def height_above_plane(self):
        return self._height

    def request_attach(self):
        """Arm the latch (explicit mode). Returns False if already attached."""
        if self.state == ATTACHED:
            return False
        self.state = REQUESTED
        return True

    def request_detach(self):
        """Release the tip; it stays where the physics leaves it."""
        if self.state == DETACHED:
            return False
        self.state = DETACHED
        self._t_latch = None
        self._tip0 = None
        self._q_offset = None
        self.sync_target_to_tip()
        self.cable.set_grasp_spring_enabled(False)
        return True

    def in_range(self):
        """Geometric attach condition, evaluated on the last update_grasp call."""
        return (self._distance <= self.attach_distance
                and -0.05 <= self._height <= self.attach_height)

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
        self._fixture_xy = (float(pose7[0]), float(pose7[1]))
        self.fixture_set = True

    def sync_target_to_tip(self):
        self.cable.set_grasp_pose(self.cable.tip_pose())

    def _clamp_reachable(self, pos):
        """Project the grasp target into the disk the rod can reach.

        The rod is inextensible: a target beyond length_m of the fixture puts
        the attachment spring in sustained tension, and tension through the
        Cosserat mapping (whose geometric stiffness the implicit solver never
        sees) flips transverse modes unstable — the rod coils up. Physically,
        fingers pulling a taut cable SLIP: cap the anchor at 99% of reach.
        """
        fx, fy = self._fixture_xy
        r_max = 0.99 * float(self.cable.cfg["length_m"])
        dx, dy = pos[0] - fx, pos[1] - fy
        r = math.hypot(dx, dy)
        if r <= r_max:
            return pos
        s = r_max / r
        return [fx + dx * s, fy + dy * s, pos[2]]

    def update_grasp(self, grasp_pose7, now_s):
        """Feed the current gripper pose; returns True on the latch edge."""
        if not self.fixture_set:
            return False
        tip = self.cable.tip_pose()
        self._distance = math.dist(grasp_pose7[:2], tip[:2])
        self._height = grasp_pose7[2] - self._fixture_z

        if self.state != ATTACHED:
            # "explicit" waits for a supervisor request; "proximity" self-arms
            armed = self.state == REQUESTED or self.attach_mode == "proximity"
            if not armed or not self.in_range():
                return False
            self.state = ATTACHED
            self._t_latch = now_s
            self._tip0 = tip
            gq = tuple(grasp_pose7[3:7])
            self._q_offset = _q_mul(_q_conj(gq), tuple(tip[3:7]))
            # Engage the attachment: target synced to the current tip so the
            # spring starts at rest, then the ramp walks it to the gripper.
            self.cable.set_grasp_pose(tip)
            self.cable.set_grasp_spring_enabled(True)
            return True

        a = 1.0 if self.ramp_s <= 0 else min(
            1.0, (now_s - self._t_latch) / self.ramp_s)
        gq = tuple(grasp_pose7[3:7])
        goal_q = _q_mul(gq, self._q_offset)
        pos = [self._tip0[i] + a * (grasp_pose7[i] - self._tip0[i]) for i in range(3)]
        pos = self._clamp_reachable(pos)
        quat = _q_slerp(tuple(self._tip0[3:7]), goal_q, a)
        self.cable.set_grasp_pose(pos + list(quat))
        return False
