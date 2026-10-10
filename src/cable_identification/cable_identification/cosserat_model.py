"""Parameterized SOFA Cosserat cable (piecewise-constant strain, full 6-strain state).

Canonical node structure (mirrors the Cosserat v25.12 plugin examples):

    <parent>
      cable_solver          EulerImplicitSolver + SparseLDLSolver
        rigidBase           Rigid3d base frame (driven boundary) + anchor spring
        cosseratCoordinate  Vec6d strains + BeamHookeLawForceField + Kelvin-Voigt damping
        frames              Rigid3d centerline frames + UniformMass + DiscreteCosseratMapping

Planar table scene (``planar`` and ``grasp_tip``): the base is the fixed fixture
(``FixedProjectiveConstraint``, no anchor spring) and the material end s = L is held by the
three bilateral rows ``[x, y, yaw]`` of ``PlanarAttachmentConstraint`` (cosserat-patches/0004)
in fixture axes, solved by FreeMotionAnimationLoop + GenericConstraintCorrection. Its
multiplier is an impulse: the reaction on the cable is ``lambda / h``.

Strain state per section: ``(kappa_x, kappa_y, kappa_z, eps_x, eps_y, eps_z)`` = torsion,
bending about y and z, axial extension and the two shears, all as increments over the
straight rest configuration: the plugin's exponential adds the unit axial stretch itself,
so ``eps_x = 0`` is the undeformed rod and ``xi = (kappa, 1 + eps_x, eps_y, eps_z)`` is the
twist of Renda et al. 2016.

Constitutive law (per section of rest length ``l``): linear Hooke
``f = -diag(GJ, EI, EI, EA, GA, GA) l (q - q0)`` plus the Kelvin-Voigt damping
``-rayleigh_stiffness_s diag(GJ, EI, EI, EA, GA, GA) l q_dot`` (Renda 2016 eq. 15 with
``Upsilon`` proportional to ``Sigma``). It is intended for small strains (|kappa| L of
order 1, |eps| << 1); the nominal parameters are not experimentally calibrated.

The Cosserat data field ``GI`` receives the torsional rigidity GJ [N.m^2].
Units: SI (m, kg, s, N).
"""

import math
import os

import numpy as np
import yaml

# Section-major order of the strain state; the rest value of every component is 0.
STRAIN_COMPONENTS = ("kappa_x", "kappa_y", "kappa_z", "epsilon_x", "epsilon_y", "epsilon_z")
# Planar rod in the base xy-plane: in-plane bending, axial extension, in-plane shear.
PLANAR_ACTIVE_COMPONENTS = (2, 3, 4)

DEFAULT_CONFIG = {
    "length_m": 0.5,
    "radius_m": 0.004,
    "mass_kg": 0.05,
    "EI_Nm2": 0.01,
    "GJ_Nm2": 0.008,
    "EA_N": 5000.0,
    "GA_N": 2000.0,
    # Mass-proportional viscous damping of the frames [1/s]: the only friction-like
    # term (table/air). The table model neglects friction: 0.
    "rayleigh_mass_per_s": 0.0,
    # Kelvin-Voigt internal damping [s]: Upsilon = rayleigh_stiffness_s * Sigma, a real
    # force on the strain rates (stiffness-proportional, hence the historical name).
    "rayleigh_stiffness_s": 0.02,
    "number_of_sections": 12,
    "number_of_frames": 30,
    "timestep_s": 0.01,
    "marker_s_over_l": [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0],
    # scene gravity; [0,0,0] for planar table shaping (table carries the rod)
    "gravity": [0.0, 0.0, -9.81],
    # False: base driven by the gripper, tip free (hanging cable)
    # True:  base clamped at the fixture, tip attached to gripper on latch
    "grasp_tip": False,
    # latch decision only: gripper within this 3D distance of the cable end
    "attach_distance_m": 0.010,
    # attached: the composed cable target T_gripper T_offset may leave the fixture plane by at
    # most attach_plane_tolerance_m and tilt its section normal by at most attach_tilt_rad
    # (numerical tolerances of the ideal planar benchmark, not gripper clearances)
    "attach_plane_tolerance_m": 1e-8,
    "attach_tilt_rad": 1e-8,
    # "proximity" latches on its own; "explicit" waits for a supervisor request
    "attach_mode": "proximity",
    # Legacy penalty grip (non-planar grasp_tip only; the planar scene uses the exact
    # Lagrange attachment): springs stiff enough that the attachment error stays far
    # below the 1 mm / 0.02 rad identifiability budget under the rod's mN / mN.m loads.
    "grasp_stiffness": 1.0e5,
    "grasp_angular_stiffness": 10.0,
    # Constrain the cable to the z = base plane (table shaping without contact)
    "planar": False,
    # Velocity-dependent inertia of the mapped frame masses, -m (dJ/dt q'), which
    # SOFA's re-assembled M(q) q'' omits (add_convective_inertia). Measured in the
    # declared excitation envelope: 0.2 % of the elastic force (median), 3 % at
    # worst, below the 5 % strain tolerance; its explicit treatment is unstable at
    # h = 0.01 s when the softest cable snaps taut (stable at 0.005 s). Off by
    # default; enable with timestep_s <= 0.005 for the full-fidelity transient.
    "convective_inertia": False,
}

# Core SOFA component plugins the scene needs (plus Cosserat itself)
REQUIRED_PLUGINS = [
    "Sofa.Component.AnimationLoop",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.LinearSystem",
    "Sofa.Component.StateContainer",
    "Sofa.Component.SolidMechanics.Spring",
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.Constraint.Lagrangian.Solver",
    "Sofa.Component.Constraint.Lagrangian.Correction",
    "Sofa.Component.Mass",
    "Sofa.Component.MechanicalLoad",
    "Cosserat",
]

# Physical parameters the estimator is allowed to update online.
IDENTIFIABLE_PARAMETERS = ("EI", "GJ", "EA", "GA",
                           "rayleigh_stiffness", "rayleigh_mass")


def hooke_diagonal(cfg):
    """Section stiffness per unit length, in STRAIN_COMPONENTS order [N.m^2, N]."""
    return np.array([cfg["GJ_Nm2"], cfg["EI_Nm2"], cfg["EI_Nm2"],
                     cfg["EA_N"], cfg["GA_N"], cfg["GA_N"]], dtype=float)


def strain_dimension(cfg):
    return len(STRAIN_COMPONENTS) * int(cfg["number_of_sections"])


def strain_weights(cfg):
    """Diagonal of the strain-energy inner product ``q^T W q = 2 V_elastic`` (6 ns,).

    Curvatures [1/m] and extensions [-] have different units; the Hooke stiffness
    ``diag(Sigma) l`` of the nominal configuration is the fixed, physically meaningful
    scaling shared by the POD, its projection and the ROM compatibility checks.
    """
    section_len = float(cfg["length_m"]) / int(cfg["number_of_sections"])
    return np.tile(hooke_diagonal(cfg) * section_len, int(cfg["number_of_sections"]))


def active_components(cfg):
    """Strain dofs that are free (flat index into the section-major 6 ns state)."""
    dimension = len(STRAIN_COMPONENTS) * int(cfg["number_of_sections"])
    if not cfg["planar"]:
        return list(range(dimension))
    return [index for index in range(dimension)
            if index % len(STRAIN_COMPONENTS) in PLANAR_ACTIVE_COMPONENTS]


def frame_rigid_mass(cfg):
    """``UniformMass.vertexMass`` of one frame: a rod segment of the lumped mass.

    SOFA's ``totalMass`` alone leaves ``inertiaMatrix`` at identity, i.e. a rotational
    inertia numerically equal to the mass (1.7e-3 kg m^2 per frame here against the
    physical 5e-8), so the segment inertia is given explicitly (specific inertia I/m,
    body axes: x along the rod).
    """
    n_frames = int(cfg["number_of_frames"]) + 1
    mass = float(cfg["mass_kg"]) / n_frames
    segment = float(cfg["length_m"]) / int(cfg["number_of_frames"])
    radius = float(cfg["radius_m"])
    axial = radius ** 2 / 2.0
    transverse = (3.0 * radius ** 2 + segment ** 2) / 12.0
    volume = math.pi * radius ** 2 * segment
    return f"{mass!r} {volume!r} {axial!r} 0 0 0 {transverse!r} 0 0 0 {transverse!r}"


def load_config(path=None, _seen=None):
    """Load a cable config.

    A config may declare ``base: <path>`` (relative to itself) to inherit
    another file and override only some keys. Truth and estimator configs use
    this so they cannot drift apart on geometry — only on the physical
    parameters that are being identified.
    """
    cfg = dict(DEFAULT_CONFIG)
    if not path:
        return cfg

    path = os.path.abspath(path)
    _seen = _seen or []
    if path in _seen:
        raise ValueError(f"circular cable config include: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    section = data.get("cable", data)

    base = section.pop("base", None)
    if base:
        parent = os.path.join(os.path.dirname(path), base)
        cfg.update(load_config(parent, _seen + [path]))
    cfg.update(section)
    return cfg


def add_required_plugins(root):
    root.addObject("RequiredPlugin", name="cable_plugins", pluginName=REQUIRED_PLUGINS)
    # SOFA v25.12 requires an explicit animation loop at the root
    root.addObject("DefaultAnimationLoop")


def uses_planar_attachment(cfg):
    """Planar table scene: fixed fixture + Lagrange end attachment (needs FreeMotionAnimationLoop)."""
    return bool(cfg.get("planar") and cfg.get("grasp_tip"))


def prepare_root(root, cfg, animation_loop="auto", extra_plugins=()):
    """Plugins + gravity/dt + exactly one animation loop.

    "auto": FreeMotionAnimationLoop and its constraint solver for the Lagrange end
    attachment, DefaultAnimationLoop for force-only scenes; None: the caller adds its own.
    """
    root.gravity = [float(g) for g in cfg["gravity"]]
    root.dt = float(cfg["timestep_s"])
    root.addObject("RequiredPlugin", name="cable_plugins",
                   pluginName=REQUIRED_PLUGINS + list(extra_plugins))
    if animation_loop == "auto":
        animation_loop = ("FreeMotionAnimationLoop" if uses_planar_attachment(cfg)
                          else "DefaultAnimationLoop")
    if animation_loop:
        root.addObject(animation_loop)
    if animation_loop == "FreeMotionAnimationLoop":
        # One 3x3 bilateral block, inverted exactly per Gauss-Seidel sweep; the tolerance
        # only stops the second sweep. Cold start every step (no stored multipliers).
        root.addObject("BlockGaussSeidelConstraintSolver", name="constraintSolver",
                       maxIterations=100, tolerance=1e-14, computeConstraintForces=True)


def build_geometry(cfg):
    """Sections/frames arrays following the plugin's BuildCosseratGeometry convention."""
    length = float(cfg["length_m"])
    ns = int(cfg["number_of_sections"])
    nf = int(cfg["number_of_frames"])

    section_len = length / ns
    strains = [[0.0] * len(STRAIN_COMPONENTS) for _ in range(ns)]
    section_lengths = [section_len] * ns
    curv_abs_input = [0.0] + [(i + 1) * section_len for i in range(ns)]
    curv_abs_input[-1] = length

    frame_len = length / nf
    frames = [[i * frame_len, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] for i in range(nf)]
    frames.append([length, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    curv_abs_output = [i * frame_len for i in range(nf)] + [length]
    return strains, section_lengths, curv_abs_input, frames, curv_abs_output


def marker_frame_indices(cfg):
    """Map marker s/L values to nearest centerline frame indices."""
    nf = int(cfg["number_of_frames"])
    return [min(nf, max(0, round(float(s) * nf))) for s in cfg["marker_s_over_l"]]


def add_convective_inertia(cable, frames_node, eps=1e-6):
    """Coriolis/centrifugal force of the lumped frame masses as a frame wrench.

    SOFA rebuilds ``M(q) = J^T diag(m, I) J`` every step but drops ``d(J q')/dt`` at fixed
    ``q'``, so the frames' acceleration misses ``a_c = (dJ/dt) q'``. The wrench
    ``(-m a_c, -I_w alpha_c - omega x I_w omega)`` is written to a ConstantForceField on the
    frames at the start of every step (centered difference of applyJ along the current
    independent velocity) and reaches the independent dofs through the mapping chain as ``-J^T m a_c``
    (and ``-Phi^T J^T m a_c`` for the ROM: the Galerkin projection is automatic). The term is
    explicit (no B/K contribution): a few percent of the elastic force in fast transients,
    and at h = 0.01 s the explicit feedback through the under-resolved axial mode (~80 Hz)
    diverges when the softest cable snaps taut (vertex EI 0.005, c_R 0.022, excitation
    trajectory 1, t = 7.6 s); at h = 0.005 s the same episode is stable. Optional
    (``convective_inertia``), off by default.
    """
    import Sofa.Core

    n_frames = len(cable.frames_mo.position.value)
    rigid_mass = [float(v) for v in frame_rigid_mass(cable.cfg).split()]
    mass, inertia = rigid_mass[0], rigid_mass[0] * np.array([rigid_mass[2], rigid_mass[6], rigid_mass[10]])
    force_field = frames_node.addObject(
        "ConstantForceField", template="Rigid3d", name="convectiveInertia",
        indices=list(range(n_frames)), forces=[[0.0] * 6] * n_frames, showArrowSize=0.0)
    cable.convective_eps = eps

    class ConvectiveInertia(Sofa.Core.Controller):
        def onAnimateBeginEvent(self, event):
            cable.refresh_mapping()
            v0 = np.array(cable.frames_mo.velocity.value, dtype=float)
            state = cable.save_state()
            key = "modal" if cable.modal_mo is not None else "strain"
            independent = cable.modal_mo if cable.modal_mo is not None else cable.strain_mo
            velocities = []
            try:
                for sign in (1.0, -1.0):  # centered along the current independent velocity
                    with independent.position.writeable() as x:
                        x[:] = state[key] + sign * eps * state[f"{key}_velocity"]
                    cable.refresh_mapping()
                    velocities.append(np.array(cable.frames_mo.velocity.value, dtype=float))
            finally:
                cable.restore_state(state)
            acceleration = (velocities[0] - velocities[1]) / (2 * eps)
            poses = np.asarray(cable.frames_mo.position.value)
            x, y, z, w = poses[:, 3], poses[:, 4], poses[:, 5], poses[:, 6]
            rotation = np.empty((n_frames, 3, 3))
            rotation[:, 0] = np.c_[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]
            rotation[:, 1] = np.c_[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]
            rotation[:, 2] = np.c_[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
            inertia_world = np.einsum("fij,j,fkj->fik", rotation, inertia, rotation)
            omega = v0[:, 3:]
            torque = (-np.einsum("fij,fj->fi", inertia_world, acceleration[:, 3:])
                      - np.cross(omega, np.einsum("fij,fj->fi", inertia_world, omega)))
            wrench = np.c_[-mass * acceleration[:, :3], torque]
            force_field.forces.value = wrench.tolist()
            cable.convective_wrench = wrench

    frames_node.addObject(ConvectiveInertia(name="convectiveInertiaUpdate"))
    return force_field


class CableHandles:
    """Live handles into the built scene used by the adapter/identification code."""

    def __init__(self, solver_node, base_mo, strain_mo, force_field, frames_mo, cfg,
                 grasp_target_mo=None, ode_solver=None, grasp_spring=None, mapping=None,
                 modal_mo=None, mor_mapping=None, linear_systems=None, damping=None,
                 attachment=None, fixture=None):
        self.solver_node = solver_node
        self.base_mo = base_mo
        self.strain_mo = strain_mo
        self.force_field = force_field
        self.frames_mo = frames_mo
        self.cfg = cfg
        self.grasp_target_mo = grasp_target_mo
        self.ode_solver = ode_solver
        self.grasp_spring = grasp_spring
        self.mapping = mapping
        self.modal_mo = modal_mo
        self.mor_mapping = mor_mapping
        self.damping = damping
        # planar table scene: PlanarAttachmentConstraint on the last frame, fixture projection
        self.attachment = attachment
        self.fixture = fixture
        self.kelvin_voigt_s = float(cfg["rayleigh_stiffness_s"])
        # {"A": MatrixLinearSystem, "M": ..., "B": ..., "K": ...} when built with expose_matrices
        self.linear_systems = linear_systems or {}
        self.marker_indices = marker_frame_indices(cfg)
        self.grasp_index = len(frames_mo.position.value) - 1
        # per-frame wrench of the convective inertia at the start of the last step (zeros if off)
        self.convective_wrench = np.zeros((len(frames_mo.position.value), 6))

    def system_matrices(self):
        """The implicit system SOFA solved in the last step, as numpy/scipy.

        EulerImplicitSolver (v25.12 source): ``A dv = b`` with
        ``A = (1 + h rM) M - h B - h (h + rK) K`` and ``b = h (f + ((h + rK) K - rM M) v)``,
        K = df/dq (negative for a spring), B = df/dv (the Kelvin-Voigt damping, negative
        definite). The solver's ``rayleighStiffness`` and the Hooke law's own
        ``rayleighStiffness`` are 0 in this model (all physical damping is the explicit
        damping force field), so every K factor is ``-h^2``; the general factors are kept
        so the identities hold for any setting. Observers assemble one term each
        (pre-multiplied); ``K_direct`` skips the mapped components (grasp spring, frame
        mass), which splits K into the Hooke law (strain block), the base clamp (base
        block) and the projected grasp spring ``J^T K_g J``. Needs ``expose_matrices=True``
        + a step.
        """
        if not self.linear_systems:
            raise RuntimeError("build the cable with expose_matrices=True")
        h = float(self.solver_node.getRoot().dt.value)
        rM = float(self.ode_solver.rayleighMass.value)
        rK = float(self.ode_solver.rayleighStiffness.value)
        rK_ff = float(self.force_field.rayleighStiffness.value)
        # FullMatrix comes back as a view on SOFA's buffer: copy before the next step
        # (or the scene unload) overwrites it.
        terms = {key: np.array(self.linear_systems[key].A(), dtype=float, copy=True)
                 for key in ("M", "B", "K", "K_direct")}
        kF_springs, kF_hooke = -h * (h + rK), -h * (h + rK + rK_ff)
        nb = self.base_mo.position.value.size - 1  # Rigid3d: 7 coords, 6 dofs
        k_clamp = np.zeros_like(terms["K"])
        k_clamp[:nb, :nb] = terms["K_direct"][:nb, :nb] / kF_springs
        out = {
            "A": self.linear_systems["A"].A(), "b": self.linear_systems["A"].b(),
            "x": self.linear_systems["A"].x(),
            "M_term": terms["M"], "B_term": terms["B"], "K_term": terms["K"],
            "K_direct_term": terms["K_direct"],
            "M": terms["M"] / (1.0 + h * rM), "B": terms["B"] / -h, "K_clamp": k_clamp,
            "factors": {"h": h, "rM": rM, "rK": rK, "rK_ff": rK_ff, "mF": 1.0 + h * rM,
                        "bF": -h, "kF_springs": kF_springs, "kF_hooke": kF_hooke},
        }
        if self.modal_mo is None:
            k_int = np.zeros_like(terms["K"])
            k_int[nb:, nb:] = terms["K_direct"][nb:, nb:] / kF_hooke
            k_grasp = (terms["K"] - terms["K_direct"]) / kF_springs
            out.update(K_int=k_int, K_grasp=k_grasp, K=k_int + k_clamp + k_grasp)
        else:
            # ROM: the Hooke law sits on the mapped strain state, so it and the grasp
            # spring both reach the modal dofs through Phi with their own factors.
            out["K_mapped_term"] = terms["K"] - terms["K_direct"]
        return out

    def set_base_pose(self, pose7):
        """Kinematically drive/relocate the cable base: [x y z qx qy qz qw].

        The mapped frames are only recomputed by the next solve's propagation,
        so the forces of that solve still see the previous base pose (one-step
        boundary lag). Call `refresh_mapping()` afterwards when the solve must
        use the new pose immediately (e.g. to match the Optimus estimator, whose
        sigma-point propagations always start from the current boundary).
        """
        with self.base_mo.position.writeable() as p:
            p[0] = pose7
        with self.base_mo.rest_position.writeable() as rp:
            rp[0] = pose7

    def refresh_mapping(self):
        """Re-apply the Cosserat mapping now (frames <- current strains + base)."""
        if self.mor_mapping is not None:
            self.mor_mapping.init()
        if self.mapping is not None:
            self.mapping.init()

    def set_grasp_pose(self, pose7, twist6=None):
        """Prescribe the pose (and twist [v, omega], world) of the grasped frame's target.

        The legacy spring target is a pose only; the attachment also takes the twist (used
        when the constraint solver works on velocities).
        """
        if self.attachment is not None:
            self.attachment.target.value = [float(v) for v in pose7]
            self.attachment.targetVelocity.value = [float(v) for v in (
                np.zeros(6) if twist6 is None else twist6)]
            return
        with self.grasp_target_mo.position.writeable() as p:
            p[0] = pose7

    def set_attachment_active(self, active):
        """Create/remove the three attachment rows; the cable state is not touched."""
        if active and not any(obj.getClassName() == "FreeMotionAnimationLoop"
                              for obj in self.solver_node.getRoot().objects):
            raise RuntimeError("the Lagrange end attachment needs FreeMotionAnimationLoop at the "
                               "root (prepare_root(..., animation_loop='auto'))")
        self.attachment.active.value = bool(active)

    def attachment_active(self):
        return bool(self.attachment.active.value)

    def attachment_target(self):
        return np.array(self.attachment.target.value, dtype=float)

    def attachment_residual(self):
        """Current ``[dx, dy, dyaw]`` of the attached frame vs its target, in fixture axes."""
        from cable_identification.coupling import planar_residual
        frame = np.asarray(self.frames_mo.position.value[int(self.attachment.index.value)], dtype=float)
        return planar_residual(frame, self.attachment_target(),
                               np.asarray(self.attachment.planeOrientation.value, dtype=float))

    def attachment_reaction(self):
        """Force/moment ``[Fx, Fy, Mz]`` (fixture axes) the attachment applied on the cable in
        the last step: the solver's multipliers are impulses (x += h dx, v += dx), so lambda / h."""
        return np.asarray(self.attachment.findData("lambda").value, dtype=float) / float(
            self.solver_node.getRoot().dt.value)

    def nearest_frame(self, point):
        """Index of the centerline frame closest to a world point (the grasped material point)."""
        positions = np.asarray(self.frames_mo.position.value)[:, :3]
        return int(np.argmin(np.linalg.norm(positions - np.asarray(point, dtype=float)[:3], axis=1)))

    def frame_pose(self, index):
        return [float(v) for v in self.frames_mo.position.value[index]]

    def set_grasp_point(self, index):
        """Attach the grasp spring to frame ``index`` (the material point under the fingers).

        The cable beyond that frame stays free, so the effective free length is the rest
        arc length up to the grasped frame. Post-init only (the latch happens while
        stepping); ``points`` is tracked by the spring and re-indexed on the next update.
        """
        self.grasp_index = int(index)
        if self.grasp_spring is None:
            return
        self.grasp_spring.findData("points").value = [self.grasp_index]
        if self.grasp_spring.findLink("mstate").getLinkedBase() is not None:
            self.grasp_spring.reinit()

    def set_grasp_spring_enabled(self, enabled):
        """Engage/release the attachment spring.

        While no gripper holds the cable its end must be FREE (a detached cable end
        is free anyway): the spring only exists while the fingers are closed.
        """
        if self.grasp_spring is None:
            return
        k = float(self.cfg["grasp_stiffness"]) if enabled else 0.0
        ka = float(self.cfg["grasp_angular_stiffness"]) if enabled else 0.0
        self.grasp_spring.findData("stiffness").value = [k]
        self.grasp_spring.findData("angularStiffness").value = [ka]
        # reinit() dereferences the mstate link, which Simulation.init() resolves; before
        # that (runSofa builds the whole scene first) it segfaults, and init() will pick
        # the new values up anyway.
        if self.grasp_spring.findLink("mstate").getLinkedBase() is not None:
            self.grasp_spring.reinit()

    def save_state(self):
        """Deep copies of every independent mechanical field, so a probe or a what-if
        rollout can be undone exactly.

        Frames are mapped from the strains (or modal coordinates) and the base, so the
        independent positions, velocities and rest references fully determine them.
        """
        def copy(data):
            return np.array(data.value, dtype=float, copy=True)

        state = {"base": copy(self.base_mo.position), "base_velocity": copy(self.base_mo.velocity),
                 "base_rest": copy(self.base_mo.rest_position),
                 "strain_rest": copy(self.strain_mo.rest_position)}
        if self.modal_mo is not None:
            state.update(modal=copy(self.modal_mo.position), modal_velocity=copy(self.modal_mo.velocity))
        else:
            state.update(strain=copy(self.strain_mo.position),
                         strain_velocity=copy(self.strain_mo.velocity))
        return state

    def restore_state(self, state):
        """Write a ``save_state`` snapshot back directly and re-map the frames."""
        fields = [(self.base_mo.position, "base"), (self.base_mo.velocity, "base_velocity"),
                  (self.base_mo.rest_position, "base_rest"),
                  (self.strain_mo.rest_position, "strain_rest")]
        if self.modal_mo is not None:
            fields += [(self.modal_mo.position, "modal"), (self.modal_mo.velocity, "modal_velocity")]
        else:
            fields += [(self.strain_mo.position, "strain"),
                       (self.strain_mo.velocity, "strain_velocity")]
        for data, key in fields:
            with data.writeable() as values:
                values[:] = state[key]
        self.refresh_mapping()

    def frame_poses(self):
        return self.frames_mo.position.value.copy()

    def tip_pose(self):
        return [float(v) for v in self.frames_mo.position.value[-1]]

    def marker_positions(self):
        poses = self.frames_mo.position.value
        return [list(poses[i][:3]) for i in self.marker_indices]

    def get_parameter(self, name):
        """Read back a physical parameter (SI units)."""
        if name == "rayleigh_mass":
            return float(self.ode_solver.findData("rayleighMass").value)
        if name == "rayleigh_stiffness":
            return self.kelvin_voigt_s
        return float(self.force_field.findData(_FORCE_FIELD_DATA[name]).value)

    def set_parameter(self, name, value):
        """Update a physical parameter and reinit the affected components.

        The Kelvin-Voigt damping is proportional to the Hooke stiffness, so every
        stiffness update re-synchronises the damping coefficients as well.
        """
        value = float(value)
        if name == "rayleigh_mass":
            self.ode_solver.findData("rayleighMass").value = value
            return
        if name == "rayleigh_stiffness":
            self.kelvin_voigt_s = value
        else:
            self.force_field.findData(_FORCE_FIELD_DATA[name]).value = value
            self.force_field.reinit()
        self.sync_damping()

    def hooke_diagonal(self):
        """Current ``diag(GJ, EI, EI, EA, GA, GA)`` of the force field (follows Data links)."""
        ff = self.force_field
        return np.array([float(ff.GI.value), float(ff.EI.value), float(ff.EI.value),
                         float(ff.EA.value), float(ff.GA.value), float(ff.GA.value)])

    def sync_damping(self):
        """Kelvin-Voigt coefficients ``rayleigh_stiffness_s * Sigma * l`` per section."""
        if self.damping is None:
            return
        lengths = np.asarray(self.force_field.length.value, dtype=float)
        coefficients = self.kelvin_voigt_s * np.outer(lengths, self.hooke_diagonal())
        self.damping.findData("dampingCoefficient").value = coefficients.tolist()


_FORCE_FIELD_DATA = {"EI": "EI", "GJ": "GI", "GI": "GI", "EA": "EA", "GA": "GA"}


def build_cable(parent, cfg, name="cable", base_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                show=False, reduction=None, expose_matrices=False):
    """Build the cable under `parent`; returns CableHandles.

    ``expose_matrices`` routes the solver through an explicit ``MatrixLinearSystem``
    (same assembled A, readable from Python) and adds M/B/K observer systems.
    """
    if reduction is not None:
        from cable_identification.strain_basis import validate_reduction
        modes_path, _, _ = validate_reduction(cfg, reduction)
    strains, section_lengths, curv_in, frames, curv_out = build_geometry(cfg)
    show_flag = 1 if show else 0
    show_scale = 0.03 if show else 0.0

    solver = parent.addChild(f"{name}_solver")
    # All physical damping is the explicit Kelvin-Voigt force field below; the
    # solver's Rayleigh factors would also damp the clamp and grasp springs.
    ode_solver = solver.addObject(
        "EulerImplicitSolver",
        rayleighMass=float(cfg["rayleigh_mass_per_s"]), rayleighStiffness=0.0)
    linear_systems = {}
    if expose_matrices:
        # The typed binding (A()/b()/x()) is chosen at addObject time, so import first.
        import Sofa.SofaLinearSystem  # noqa: F401
        template = "CompressedRowSparseMatrixd"
        linear_systems["A"] = solver.addObject("MatrixLinearSystem", template=template, name="systemA")
        # One contribution each, never solved: the terms of the dynamics equation. Dense
        # on purpose: a sparse matrix only compresses when a solver factorises it, and
        # the Python binding reads the compressed arrays (an unsolved sparse observer
        # reads back as all zeros).
        for key, flags in (("M", (True, False, False)), ("B", (False, True, False)),
                           ("K", (False, False, True)), ("K_direct", (False, False, True))):
            linear_systems[key] = solver.addObject(
                "MatrixLinearSystem", template="FullMatrix", name=f"system{key}",
                assembleMass=flags[0], assembleDamping=flags[1], assembleStiffness=flags[2],
                assembleGeometricStiffness=False, applyProjectiveConstraints=False,
                applyMappedComponents=key != "K_direct")
        # The composite assembles every listed system each step and hands A to the solver.
        # SOFA 25.12 never raises the composite's own change flag (it only forwards), so
        # the solver would never factorise: link the flag to the solved system's.
        composite = solver.addObject(
            "CompositeLinearSystem", template=template, name="systemComposite",
            linearSystems=[s.getLinkPath() for s in linear_systems.values()],
            solverLinearSystem=linear_systems["A"].getLinkPath(),
            factorizationInvalidation="@systemA.factorizationInvalidation")
        linear_solver = solver.addObject("SparseLDLSolver", name="solver", template=template,
                                         linearSystem=composite.getLinkPath())
    else:
        linear_solver = solver.addObject("SparseLDLSolver", name="solver",
                                         template="CompressedRowSparseMatrixd")

    attached_scene = uses_planar_attachment(cfg)
    if attached_scene:
        # compliance of the coupled system (base + strains or modes) from the solver's own
        # factorisation: LinearSolverConstraintCorrection has no Vec6 template and only
        # covers its own mechanical state
        solver.addObject("GenericConstraintCorrection", name="constraintCorrection",
                         linearSolver=linear_solver.getLinkPath(),
                         ODESolver=ode_solver.getLinkPath())

    rigid_base = solver.addChild("rigidBase")
    base_mo = rigid_base.addObject(
        "MechanicalObject", template="Rigid3d", name="RigidBaseMO",
        position=[list(base_pose)], velocity=[[0.0] * 6], showObject=0)
    fixture = None
    if attached_scene:
        # Fixed fixture: response and velocity projected to zero. The projection never resets
        # positions, so the pose is only ever written by set_base_pose (GraspCoupling.on_fixture).
        fixture = rigid_base.addObject(
            "FixedProjectiveConstraint", name="fixture", template="Rigid3d", indices=[0],
            activate_projectVelocity=True)
    else:
        rigid_base.addObject(
            "RestShapeSpringsForceField", name="baseSpring", template="Rigid3d",
            stiffness=1e8, angularStiffness=1e8, points=0, mstate="@RigidBaseMO")

    modal_mo = None
    mor_mapping = None
    coord_parent = solver
    if reduction is not None:
        coord_parent = solver.addChild("modalCoordinate")
        modal_mo = coord_parent.addObject(
            "MechanicalObject", template="Vec1d", name="modalCoordinateMO",
            position=[0.] * reduction.n_modes)
    coord = coord_parent.addChild("cosseratCoordinate")
    strain_mo = coord.addObject(
        "MechanicalObject", template="Vec6d", name="cosseratCoordinateMO",
        position=strains, **({"rest_position": strains} if reduction is not None else {}))
    if reduction is not None:
        mor_mapping = coord.addObject(
            "ModelOrderReductionMapping", name="strainModalMapping",
            input=modal_mo.getLinkPath(), output=strain_mo.getLinkPath(),
            modesPath=modes_path)
    force_field = coord.addObject(
        "BeamHookeLawForceField", name="hooke", template="Vec6d", crossSectionShape="circular",
        length=section_lengths, radius=float(cfg["radius_m"]),
        useInertiaParams=True,
        EI=float(cfg["EI_Nm2"]), GI=float(cfg["GJ_Nm2"]),
        EA=float(cfg["EA_N"]), GA=float(cfg["GA_N"]),
        rayleighStiffness=0.0)
    # Kelvin-Voigt internal damping as a real force (implicit through its B matrix).
    damping = coord.addObject(
        "DiagonalVelocityDampingForceField", name="kelvinVoigt", template="Vec6d",
        dampingCoefficient=(float(cfg["rayleigh_stiffness_s"])
                            * np.outer(section_lengths, hooke_diagonal(cfg))).tolist())

    frames_node = rigid_base.addChild("frames")
    coord.addChild(frames_node)
    frames_mo = frames_node.addObject(
        "MechanicalObject", template="Rigid3d", name="FramesMO",
        position=frames, showObject=show_flag, showObjectScale=show_scale)
    frames_node.addObject("UniformMass", vertexMass=frame_rigid_mass(cfg),
                          showAxisSizeFactor=0.0)
    mapping = frames_node.addObject(
        "DiscreteCosseratMapping", name="cosseratMapping",
        curv_abs_input=curv_in, curv_abs_output=curv_out,
        input1=strain_mo.getLinkPath(), input2=base_mo.getLinkPath(),
        output=frames_mo.getLinkPath(), debug=0,
        radius=float(cfg["radius_m"]))

    grasp_target_mo = None
    grasp_spring = None
    attachment = None
    if attached_scene:
        # Ideal planar grip of the end section s = L: rows [x, y, yaw] in fixture axes, off
        # until the coupling latches. Target = the straight end until then.
        x, y, z, w = (float(v) for v in base_pose[3:7])
        length = float(cfg["length_m"])
        end = [float(base_pose[0]) + length * (1 - 2 * (y * y + z * z)),
               float(base_pose[1]) + length * 2 * (x * y + z * w),
               float(base_pose[2]) + length * 2 * (x * z - y * w), x, y, z, w]
        attachment = frames_node.addObject(
            "PlanarAttachmentConstraint", name="attachment", template="Rigid3d",
            index=len(frames) - 1, active=False, target=end, targetVelocity=[0.0] * 6,
            planeOrientation=[x, y, z, w])
    elif cfg.get("grasp_tip"):
        # Kinematic pose the grasped frame is attached to (the gripper), initialized
        # at the straight tip so the attachment starts at rest. Stiff external
        # rest-shape springs = the standard attachment for mapped frames; the
        # coupling moves ``points`` to the grasped frame at latch time.
        target = parent.addChild(f"{name}_graspTarget")
        grasp_target_mo = target.addObject(
            "MechanicalObject", template="Rigid3d", name="TargetMO",
            position=[list(frames[-1])], showObject=show_flag,
            showObjectScale=show_scale)
        grasp_spring = frames_node.addObject(
            "RestShapeSpringsForceField", name="graspSpring", template="Rigid3d",
            stiffness=float(cfg["grasp_stiffness"]),
            angularStiffness=float(cfg["grasp_angular_stiffness"]),
            points=[len(frames) - 1],
            external_rest_shape=grasp_target_mo.getLinkPath(),
            external_points=[0])

    if cfg.get("planar") and reduction is None:
        # Simplified table: the rod may only deform in the plane of its base.
        # The constraint must sit on the INDEPENDENT dofs. FramesMO is a mapped
        # state, and SOFA rejects projective constraints there ("only main
        # mechanical states have an associated submatrix"), which makes the
        # constraint a silent no-op. Torsion, bending about y and the shear
        # along z are locked; in-plane bending, extension and in-plane shear
        # stay free. The ROM gets its planarity from the basis instead.
        coord.addObject(
            "PartialFixedProjectiveConstraint", name="planarConstraint",
            indices=list(range(len(strains))),
            fixedDirections=[0 if k in PLANAR_ACTIVE_COMPONENTS else 1
                             for k in range(len(STRAIN_COMPONENTS))])

    cable = CableHandles(solver, base_mo, strain_mo, force_field, frames_mo, cfg,
                         grasp_target_mo, ode_solver, grasp_spring, mapping,
                         modal_mo, mor_mapping, linear_systems, damping, attachment, fixture)
    if cfg.get("convective_inertia", False):
        add_convective_inertia(cable, frames_node)
    return cable
