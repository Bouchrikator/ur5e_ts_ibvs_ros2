"""Parameterized SOFA Cosserat cable (reduced formulation: bending + torsion).

Canonical node structure (mirrors the Cosserat v25.12 plugin examples):

    <parent>
      cable_solver          EulerImplicitSolver + SparseLDLSolver
        rigidBase           Rigid3d base frame (driven boundary) + anchor spring
        cosseratCoordinate  Vec3d strains (torsion, bend-y, bend-z) + BeamHookeLawForceField
        frames              Rigid3d centerline frames + UniformMass + DiscreteCosseratMapping

The base frame is the DRIVEN end (robot gripper). The other end is free.
The Cosserat data field ``GI`` receives the torsional rigidity GJ [N.m^2].
Units: SI (m, kg, s, N).
"""

import os

import yaml

DEFAULT_CONFIG = {
    "length_m": 0.5,
    "radius_m": 0.004,
    "mass_kg": 0.05,
    "EI_Nm2": 0.01,
    "GJ_Nm2": 0.008,
    "EA_N": 5000.0,
    "GA_N": 2000.0,
    "rayleigh_mass_per_s": 0.1,
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
    "attach_distance_m": 0.15,   # xy proximity to the free end
    "attach_height_m": 0.01,     # gripper must be this close to the table plane
    "attach_ramp_s": 0.5,
    # "proximity" latches on its own; "explicit" waits for a supervisor request
    "attach_mode": "proximity",
    # Tip attachment compliance (see cable_common.yaml for the measured
    # trade-off): 2e3 N/m = 0.21 mm attached error, stable over 300 s holds;
    # 2e4 N/m destabilizes the solver through the Cosserat mapping (springs
    # on mapped frames have no geometric stiffness) and the rod coils up.
    "grasp_stiffness": 2000.0,
    "grasp_angular_stiffness": 0.05,
    # Constrain the cable to the z = base plane (table shaping without contact)
    "planar": False,
    "planar_stiffness": 1.0e4,
}

# Core SOFA component plugins the scene needs (plus Cosserat itself)
REQUIRED_PLUGINS = [
    "Sofa.Component.AnimationLoop",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.StateContainer",
    "Sofa.Component.SolidMechanics.Spring",
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.Mass",
    "Cosserat",
]

# Physical parameters the estimator is allowed to update online.
IDENTIFIABLE_PARAMETERS = ("EI", "GJ", "EA", "GA",
                           "rayleigh_stiffness", "rayleigh_mass")


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


def prepare_root(root, cfg, animation_loop="DefaultAnimationLoop"):
    """Plugins + gravity/dt + animation loop (None: the caller adds its own loop)."""
    root.gravity = [float(g) for g in cfg["gravity"]]
    root.dt = float(cfg["timestep_s"])
    root.addObject("RequiredPlugin", name="cable_plugins", pluginName=REQUIRED_PLUGINS)
    if animation_loop:
        root.addObject(animation_loop)


def build_geometry(cfg):
    """Sections/frames arrays following the plugin's BuildCosseratGeometry convention."""
    length = float(cfg["length_m"])
    ns = int(cfg["number_of_sections"])
    nf = int(cfg["number_of_frames"])

    section_len = length / ns
    strains = [[0.0, 0.0, 0.0] for _ in range(ns)]
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


class CableHandles:
    """Live handles into the built scene used by the adapter/identification code."""

    def __init__(self, solver_node, base_mo, strain_mo, force_field, frames_mo, cfg,
                 grasp_target_mo=None, ode_solver=None, grasp_spring=None, mapping=None):
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
        self.marker_indices = marker_frame_indices(cfg)

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
        if self.mapping is not None:
            self.mapping.init()

    def set_grasp_pose(self, pose7):
        """Move the kinematic target the cable tip is constrained to."""
        with self.grasp_target_mo.position.writeable() as p:
            p[0] = pose7

    def set_grasp_spring_enabled(self, enabled):
        """Engage/release the tip attachment spring.

        While no gripper holds the cable the tip must be FREE: the spring acts
        on the mapped Cosserat frames, whose geometric stiffness is not seen
        by the implicit solver, and holding the tip with it for minutes pumps
        energy into the rod until it coils up (headless A/B repro: chain
        length 0.31 m vs 0.70 m after 600 s). Physically a detached cable end
        is free anyway. Attached dynamics are unchanged.
        """
        if self.grasp_spring is None:
            return
        k = float(self.cfg["grasp_stiffness"]) if enabled else 0.0
        ka = float(self.cfg["grasp_angular_stiffness"]) if enabled else 0.0
        self.grasp_spring.findData("stiffness").value = [k]
        self.grasp_spring.findData("angularStiffness").value = [ka]
        self.grasp_spring.reinit()

    def save_state(self):
        """Snapshot the independent DOFs so a what-if rollout can be undone.

        Frames are mapped from the strains and the base, so those two plus the
        strain velocity fully determine the configuration. Sigma-point
        evaluation replays the cable under trial parameters and must leave the
        estimator exactly where it found it.
        """
        return {
            "strain": self.strain_mo.position.value.copy(),
            "strain_velocity": self.strain_mo.velocity.value.copy(),
            "base": self.base_mo.position.value.copy(),
        }

    def restore_state(self, state):
        with self.strain_mo.position.writeable() as p:
            p[:] = state["strain"]
        with self.strain_mo.velocity.writeable() as v:
            v[:] = state["strain_velocity"]
        with self.base_mo.position.writeable() as p:
            p[:] = state["base"]

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
        field = _FORCE_FIELD_DATA[name]
        return float(self.force_field.findData(field).value)

    def set_parameter(self, name, value):
        """Update a physical parameter and reinit the affected components.

        Rayleigh damping lives on BOTH the ODE solver and the force field, so
        `rayleigh_stiffness` updates the two of them to stay consistent.
        """
        value = float(value)
        if name == "rayleigh_mass":
            self.ode_solver.findData("rayleighMass").value = value
            return
        field = _FORCE_FIELD_DATA[name]
        self.force_field.findData(field).value = value
        if name == "rayleigh_stiffness" and self.ode_solver is not None:
            self.ode_solver.findData("rayleighStiffness").value = value
        self.force_field.reinit()


_FORCE_FIELD_DATA = {"EI": "EI", "GJ": "GI", "GI": "GI", "EA": "EA", "GA": "GA",
                     "rayleigh_stiffness": "rayleighStiffness"}


def build_cable(parent, cfg, name="cable", base_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                show=False):
    """Build the cable under `parent`; returns CableHandles."""
    strains, section_lengths, curv_in, frames, curv_out = build_geometry(cfg)
    show_flag = 1 if show else 0
    show_scale = 0.03 if show else 0.0

    solver = parent.addChild(f"{name}_solver")
    ode_solver = solver.addObject(
        "EulerImplicitSolver",
        rayleighMass=float(cfg["rayleigh_mass_per_s"]),
        rayleighStiffness=float(cfg["rayleigh_stiffness_s"]))
    solver.addObject("SparseLDLSolver", name="solver",
                     template="CompressedRowSparseMatrixd")

    rigid_base = solver.addChild("rigidBase")
    base_mo = rigid_base.addObject(
        "MechanicalObject", template="Rigid3d", name="RigidBaseMO",
        position=[list(base_pose)], showObject=0)
    rigid_base.addObject(
        "RestShapeSpringsForceField", name="baseSpring", template="Rigid3d",
        stiffness=1e8, angularStiffness=1e8, points=0, mstate="@RigidBaseMO")

    coord = solver.addChild("cosseratCoordinate")
    strain_mo = coord.addObject(
        "MechanicalObject", template="Vec3d", name="cosseratCoordinateMO",
        position=strains)
    force_field = coord.addObject(
        "BeamHookeLawForceField", name="hooke", crossSectionShape="circular",
        length=section_lengths, radius=float(cfg["radius_m"]),
        useInertiaParams=True,
        EI=float(cfg["EI_Nm2"]), GI=float(cfg["GJ_Nm2"]),
        EA=float(cfg["EA_N"]), GA=float(cfg["GA_N"]),
        rayleighStiffness=float(cfg["rayleigh_stiffness_s"]))

    frames_node = rigid_base.addChild("frames")
    coord.addChild(frames_node)
    frames_mo = frames_node.addObject(
        "MechanicalObject", template="Rigid3d", name="FramesMO",
        position=frames, showObject=show_flag, showObjectScale=show_scale)
    frames_node.addObject("UniformMass", totalMass=float(cfg["mass_kg"]),
                          showAxisSizeFactor=0.0)
    mapping = frames_node.addObject(
        "DiscreteCosseratMapping", name="cosseratMapping",
        curv_abs_input=curv_in, curv_abs_output=curv_out,
        input1=strain_mo.getLinkPath(), input2=base_mo.getLinkPath(),
        output=frames_mo.getLinkPath(), debug=0,
        radius=float(cfg["radius_m"]))

    grasp_target_mo = None
    grasp_spring = None
    if cfg.get("grasp_tip"):
        # Kinematic pose the tip is clamped to (the gripper), initialized at
        # the straight tip so the attachment starts at rest. Stiff external
        # rest-shape springs = the standard attachment for mapped frames.
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

    if cfg.get("planar"):
        # Simplified table: the rod may only bend in the plane of its base.
        # The constraint must sit on the INDEPENDENT dofs. FramesMO is a mapped
        # state, and SOFA rejects projective constraints there ("only main
        # mechanical states have an associated submatrix"), which makes the
        # constraint a silent no-op. Strain = (torsion, bend_y, bend_z), so
        # freeing bend_z alone keeps the centerline in the base xy-plane.
        coord.addObject(
            "PartialFixedProjectiveConstraint", name="planarConstraint",
            indices=list(range(len(strains))),
            fixedDirections=[1, 1, 0])

    return CableHandles(solver, base_mo, strain_mo, force_field, frames_mo, cfg,
                        grasp_target_mo, ode_solver, grasp_spring, mapping)
