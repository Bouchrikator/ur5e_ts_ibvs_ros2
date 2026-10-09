"""Grasp coupling for the table-shaping setup (shared by node and GUI scene).

Cable base = table fixture (static). The gripper grasps the cable at the material
point under its fingers: on the latch edge the frame nearest to the gripper becomes
the grasped frame (the cable beyond it stays free, so the effective free length is
the rest arc length up to that frame) and the relative pose gripper -> frame is
captured. From then on the attachment target is ``gripper (+) relative pose``, so
the commanded position AND orientation are transmitted without slip and the spring
starts exactly at rest (no ramp, no reach clamp: pulling past the rest length
stretches the extensible rod, which is the physics, not a slip law).

Two engagement policies, selected by `attach_mode`:

  "proximity"  legacy: latches by itself as soon as the gripper is in range.
  "explicit"   an external supervisor calls request_attach(); the latch only
               fires once the geometric condition is ALSO satisfied. This is
               the deterministic policy used for identification experiments.

Only the latch DECISION differs between the two; the tracking is shared.
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


def _q_rot(q, v):
    qv = _q_mul(_q_mul(q, (v[0], v[1], v[2], 0.0)), _q_conj(q))
    return (qv[0], qv[1], qv[2])


class GraspCoupling:
    def __init__(self, cable, attach_mode=None):
        self.cable = cable
        self.attach_distance = float(cable.cfg.get("attach_distance_m", 0.15))
        self.attach_height = float(cable.cfg.get("attach_height_m", 0.01))
        self.attach_mode = attach_mode or cable.cfg.get("attach_mode", "proximity")
        self.fixture_set = False
        self.state = DETACHED
        self._fixture_z = None
        self._p_offset = None  # grasped frame origin in the gripper frame, at latch
        self._q_offset = None  # grasp_q^-1 * frame_q at latch
        self._distance = float("inf")
        self._height = float("inf")
        # Detached = free end: the spring only exists while the fingers are closed.
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

    @property
    def grasp_index(self):
        """Frame index of the grasped material point (tip until the first latch)."""
        return self.cable.grasp_index

    def request_attach(self):
        """Arm the latch (explicit mode). Returns False if already attached."""
        if self.state == ATTACHED:
            return False
        self.state = REQUESTED
        return True

    def request_detach(self):
        """Release the cable; it stays where the physics leaves it."""
        if self.state == DETACHED:
            return False
        self.state = DETACHED
        self._p_offset = None
        self._q_offset = None
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
        self.fixture_set = True

    def grasp_target(self, grasp_pose7):
        """Attachment pose = gripper pose composed with the relative pose met at latch."""
        gq = tuple(grasp_pose7[3:7])
        d = _q_rot(gq, self._p_offset)
        return [grasp_pose7[0] + d[0], grasp_pose7[1] + d[1], grasp_pose7[2] + d[2],
                *_q_mul(gq, self._q_offset)]

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
            # No slip: the fingers close on the material point under them and keep
            # the relative pose they met, so the spring engages exactly at rest.
            index = self.cable.nearest_frame(grasp_pose7[:3])
            pose = self.cable.frame_pose(index)
            gq = tuple(grasp_pose7[3:7])
            self._p_offset = _q_rot(_q_conj(gq), tuple(pose[i] - grasp_pose7[i] for i in range(3)))
            self._q_offset = _q_mul(_q_conj(gq), tuple(pose[3:7]))
            self.cable.set_grasp_point(index)
            self.cable.set_grasp_pose(pose)
            self.cable.set_grasp_spring_enabled(True)
            self.state = ATTACHED
            return True

        self.cable.set_grasp_pose(self.grasp_target(grasp_pose7))
        return False
