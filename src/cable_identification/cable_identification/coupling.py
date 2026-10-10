"""Grasp coupling for the table-shaping setup (shared by node and GUI scene).

Cable base = table fixture (static, placed once per run). Planar table scene
(``cable.attachment``): on the latch edge the material end s = L is attached by the three
bilateral rows ``[x, y, yaw]`` of the Lagrange attachment; the relative pose
``T_offset = T_gripper^-1 T_end`` is captured once and the target is
``T_gripper T_offset`` (with the matching twist) until release. Commands the plane cannot
realise raise ``IncompatibleGraspCommand``; they are never projected silently. Release
removes the rows and leaves positions and velocities untouched.

Legacy branch (no attachment, non-planar grasp): the frame nearest to the gripper is held
by the penalty spring, enabled only while ATTACHED.

Two engagement policies, selected by `attach_mode`:

  "proximity"  latches by itself as soon as the gripper is in range.
  "explicit"   an external supervisor calls request_attach(); the latch only
               fires once the geometric condition is ALSO satisfied. This is
               the deterministic policy used for identification experiments.

Only the latch DECISION differs between the two; the tracking is shared.
"""

import math

import numpy as np

DETACHED = 0
REQUESTED = 1
ATTACHED = 2


class IncompatibleGraspCommand(ValueError):
    """A gripper command the planar attachment cannot realise (out of plane or tilted)."""


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


def _rotvec(q):
    """Rotation vector of a unit quaternion (atan2: no digits lost near the identity)."""
    q = np.asarray(q, dtype=float)
    if q[3] < 0.0:
        q = -q
    s = np.linalg.norm(q[:3])
    return q[:3] * (2.0 * math.atan2(s, q[3]) / s if s > 0.0 else 2.0)


def planar_residual(frame7, target7, plane_q):
    """``[dx, dy, dyaw]`` in the plane axes, the formula of PlanarAttachmentConstraint: the
    in-plane position error and the twist of ``q_frame q_target^-1`` about the plane normal."""
    axes = [np.asarray(_q_rot(plane_q, e)) for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    dp = np.asarray(frame7[:3], dtype=float) - np.asarray(target7[:3], dtype=float)
    relative = np.asarray(_q_mul(frame7[3:7], _q_conj(target7[3:7])))
    if relative[3] < 0.0:
        relative = -relative
    return np.array([dp @ axes[0], dp @ axes[1],
                     2.0 * math.atan2(float(relative[:3] @ axes[2]), float(relative[3]))])


class GraspCoupling:
    def __init__(self, cable, attach_mode=None):
        self.cable = cable
        self.attach_distance = float(cable.cfg["attach_distance_m"])
        self.attach_tilt = float(cable.cfg["attach_tilt_rad"])
        self.attach_mode = attach_mode or cable.cfg.get("attach_mode", "proximity")
        self.exact = cable.attachment is not None
        self.fixture_set = False
        self.state = DETACHED
        self._fixture = None   # fixture pose7 of this run
        self._p_offset = None  # attached frame origin in the gripper frame, at latch
        self._q_offset = None  # grasp_q^-1 * frame_q at latch
        self._previous = None  # (target pose7, time) of the last command: twist fallback
        self._distance = float("inf")
        self._height = float("inf")
        # Detached = free end (pre-init safe: only Data writes, no reinit)
        if self.exact:
            self.cable.set_attachment_active(False)
        else:
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

    @property
    def offset(self):
        """``(p_offset, q_offset)`` captured at latch (None while detached)."""
        return self._p_offset, self._q_offset

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
        self._previous = None
        if self.exact:
            self.cable.set_attachment_active(False)
        else:
            self.cable.set_grasp_spring_enabled(False)
        return True

    def in_range(self):
        """Geometric attach condition (3D distance to the end), on the last update_grasp."""
        return self._distance <= self.attach_distance

    def on_fixture(self, pose7):
        """Place the straight cable on the fixture, once per run.

        Accepted any time before the run starts (also after Simulation.init, when the
        fixture TF arrives late). Repeats of the same pose are no-ops; a different pose
        raises: a moved fixture is a new run (``reset_fixture``).
        """
        pose7 = [float(v) for v in pose7]
        if self.fixture_set:
            if np.max(np.abs(np.subtract(pose7, self._fixture))) > 1e-9:
                raise ValueError(f"fixture moved during a run: {self._fixture} -> {pose7}; "
                                 "call reset_fixture() to start a new run")
            return
        # The base is projected (fixed) in the planar scene: written here only, at rest.
        self.cable.set_base_pose(pose7)
        with self.cable.base_mo.velocity.writeable() as v:
            v[:] = 0.0
        # The mapped frames are stale until the next step: analytic straight end
        L = float(self.cable.cfg["length_m"])
        q = tuple(pose7[3:7])
        d = _q_rot(q, (L, 0.0, 0.0))
        tip = [pose7[0] + d[0], pose7[1] + d[1], pose7[2] + d[2], q[0], q[1], q[2], q[3]]
        if self.exact:
            self.cable.attachment.planeOrientation.value = list(q)
        self.cable.set_grasp_pose(tip)
        self._fixture = pose7
        self.fixture_set = True

    def reset_fixture(self, pose7):
        """New run on a (possibly moved) fixture: release, straight cable at rest, place."""
        self.request_detach()
        independent = self.cable.modal_mo if self.cable.modal_mo is not None else self.cable.strain_mo
        with independent.position.writeable() as x:
            x[:] = 0.0 if self.cable.modal_mo is not None else self.cable.strain_mo.rest_position.value
        with independent.velocity.writeable() as v:
            v[:] = 0.0
        self.fixture_set = False
        self.on_fixture(pose7)
        self.cable.refresh_mapping()

    def grasp_target(self, grasp_pose7):
        """Attachment pose = gripper pose composed with the relative pose met at latch."""
        gq = tuple(grasp_pose7[3:7])
        d = _q_rot(gq, self._p_offset)
        return [grasp_pose7[0] + d[0], grasp_pose7[1] + d[1], grasp_pose7[2] + d[2],
                *_q_mul(gq, self._q_offset)]

    def target_twist(self, grasp_pose7, grasp_twist6):
        """Twist of the rigidly offset target: v + omega x (R_g p_offset), omega."""
        omega = np.asarray(grasp_twist6[3:6], dtype=float)
        lever = np.asarray(_q_rot(tuple(grasp_pose7[3:7]), self._p_offset))
        return np.r_[np.asarray(grasp_twist6[:3], dtype=float) + np.cross(omega, lever), omega]

    def check_compatible(self, target7):
        """Raise IncompatibleGraspCommand when the commanded end pose leaves the fixture plane
        by more than attach_distance or tilts its section normal by more than attach_tilt."""
        q = tuple(self._fixture[3:7])
        normal = np.asarray(_q_rot(q, (0.0, 0.0, 1.0)))
        height = float(normal @ (np.asarray(target7[:3]) - np.asarray(self._fixture[:3])))
        section = np.asarray(_q_rot(tuple(target7[3:7]), (0.0, 0.0, 1.0)))
        tilt = math.atan2(np.linalg.norm(np.cross(section, normal)), float(section @ normal))
        if abs(height) > self.attach_distance or tilt > self.attach_tilt:
            raise IncompatibleGraspCommand(
                f"commanded end pose leaves the fixture plane: height {height:.4f} m "
                f"(limit {self.attach_distance}), tilt {tilt:.4f} rad (limit {self.attach_tilt})")

    def _command(self, grasp_pose7, now_s, grasp_twist6):
        target = self.grasp_target(grasp_pose7)
        if not self.exact:
            self.cable.set_grasp_pose(target)
            return
        self.check_compatible(target)
        if grasp_twist6 is not None:
            twist = self.target_twist(grasp_pose7, grasp_twist6)
        elif self._previous is not None and now_s > self._previous[1]:
            # no gripper twist given: backward difference of the target poses
            previous, before = self._previous
            dt = now_s - before
            twist = np.r_[(np.asarray(target[:3]) - np.asarray(previous[:3])) / dt,
                          _rotvec(_q_mul(target[3:7], _q_conj(previous[3:7]))) / dt]
        else:
            twist = np.zeros(6)
        self.cable.set_grasp_pose(target, twist)
        self._previous = (target, float(now_s))

    def update_grasp(self, grasp_pose7, now_s, grasp_twist6=None):
        """Feed the current gripper pose (and optionally its twist [v, omega], world);
        returns True on the latch edge."""
        if not self.fixture_set:
            return False
        if self.exact and self.state != ATTACHED:
            # latch decisions use the material end on the current strains and base
            self.cable.refresh_mapping()
        tip = self.cable.tip_pose()
        self._distance = math.dist(grasp_pose7[:3], tip[:3])
        normal = _q_rot(tuple(self._fixture[3:7]), (0.0, 0.0, 1.0))
        self._height = sum(n * (g - f) for n, g, f in zip(normal, grasp_pose7[:3], self._fixture[:3]))

        if self.state != ATTACHED:
            # "explicit" waits for a supervisor request; "proximity" self-arms
            armed = self.state == REQUESTED or self.attach_mode == "proximity"
            if not armed or not self.in_range():
                return False
            if self.exact:
                index = len(self.cable.frames_mo.position.value) - 1   # the material end s = L
            else:
                index = self.cable.nearest_frame(grasp_pose7[:3])
            pose = self.cable.frame_pose(index)
            # No slip: keep the relative pose met at latch, so the target starts at the frame.
            gq = tuple(grasp_pose7[3:7])
            self._p_offset = _q_rot(_q_conj(gq), tuple(pose[i] - grasp_pose7[i] for i in range(3)))
            self._q_offset = _q_mul(_q_conj(gq), tuple(pose[3:7]))
            self.cable.set_grasp_point(index)
            try:
                self._command(grasp_pose7, now_s, grasp_twist6)
            except IncompatibleGraspCommand:
                self._p_offset = self._q_offset = None
                return False
            if self.exact:
                self.cable.set_attachment_active(True)
            else:
                self.cable.set_grasp_spring_enabled(True)
            self.state = ATTACHED
            return True

        self._command(grasp_pose7, now_s, grasp_twist6)
        return False

    def checkpoint(self):
        """Replay state of everything below the rollout driver: mechanics, SOFA time, fixture,
        attachment (rows on/off, frame index, offset, target pose and twist, last
        multipliers) and the twist fallback history. The constraint solver restarts cold
        every step (no stored multipliers), so it carries no warm-start state."""
        if not self.exact:
            raise NotImplementedError("rollout checkpoints cover the planar attachment scene")
        attachment = self.cable.attachment
        return {
            "mechanics": self.cable.save_state(),
            "time": float(self.cable.solver_node.getRoot().time.value),
            "coupling": (self.state, self.fixture_set, self._fixture, self._p_offset, self._q_offset,
                         self._previous, self._distance, self._height, self.cable.grasp_index),
            "attachment": {key: np.array(attachment.findData(key).value, copy=True)
                           for key in ("active", "index", "target", "targetVelocity",
                                       "planeOrientation", "lambda", "violation")},
        }

    def restore(self, checkpoint):
        """Return the scene and the coupling exactly to ``checkpoint``."""
        attachment = self.cable.attachment
        self.cable.restore_state(checkpoint["mechanics"])

        def set_time(node):  # every context, as the step's context update leaves them
            node.time.value = checkpoint["time"]
            for child in node.children:
                set_time(child)

        set_time(self.cable.solver_node.getRoot())
        (self.state, self.fixture_set, self._fixture, self._p_offset, self._q_offset,
         self._previous, self._distance, self._height, self.cable.grasp_index) = checkpoint["coupling"]
        for key, value in checkpoint["attachment"].items():
            attachment.findData(key).value = value.tolist()
