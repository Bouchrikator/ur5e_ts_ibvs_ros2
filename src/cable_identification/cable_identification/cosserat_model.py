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
}

# Core SOFA component plugins the scene needs (plus Cosserat itself)
REQUIRED_PLUGINS = [
    "Sofa.Component.AnimationLoop",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.StateContainer",
    "Sofa.Component.SolidMechanics.Spring",
    "Sofa.Component.Mass",
    "Cosserat",
]


def load_config(path=None):
    cfg = dict(DEFAULT_CONFIG)
    if path:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        cfg.update(data.get("cable", data))
    return cfg


def add_required_plugins(root):
    root.addObject("RequiredPlugin", name="cable_plugins", pluginName=REQUIRED_PLUGINS)
    # SOFA v25.12 requires an explicit animation loop at the root
    root.addObject("DefaultAnimationLoop")


def prepare_root(root, cfg):
    """Plugins + gravity/dt + animation loop."""
    root.gravity = [float(g) for g in cfg["gravity"]]
    root.dt = float(cfg["timestep_s"])
    root.addObject("RequiredPlugin", name="cable_plugins", pluginName=REQUIRED_PLUGINS)
    root.addObject("DefaultAnimationLoop")


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
                 grasp_target_mo=None):
        self.solver_node = solver_node
        self.base_mo = base_mo
        self.strain_mo = strain_mo
        self.force_field = force_field
        self.frames_mo = frames_mo
        self.cfg = cfg
        self.grasp_target_mo = grasp_target_mo
        self.marker_indices = marker_frame_indices(cfg)

    def set_base_pose(self, pose7):
        """Kinematically drive/relocate the cable base: [x y z qx qy qz qw]."""
        with self.base_mo.position.writeable() as p:
            p[0] = pose7
        with self.base_mo.rest_position.writeable() as rp:
            rp[0] = pose7

    def set_grasp_pose(self, pose7):
        """Move the kinematic target the cable tip is constrained to."""
        with self.grasp_target_mo.position.writeable() as p:
            p[0] = pose7

    def frame_poses(self):
        return self.frames_mo.position.value.copy()

    def tip_pose(self):
        return [float(v) for v in self.frames_mo.position.value[-1]]

    def marker_positions(self):
        poses = self.frames_mo.position.value
        return [list(poses[i][:3]) for i in self.marker_indices]

    def set_parameter(self, name, value):
        """Update EI/GI(=GJ)/EA/GA (or rayleigh) and reinit the force field."""
        field = {"EI": "EI", "GJ": "GI", "GI": "GI", "EA": "EA", "GA": "GA",
                 "rayleigh_stiffness": "rayleighStiffness"}[name]
        self.force_field.findData(field).value = float(value)
        self.force_field.reinit()


def build_cable(parent, cfg, name="cable", base_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                show=False):
    """Build the cable under `parent`; returns CableHandles."""
    strains, section_lengths, curv_in, frames, curv_out = build_geometry(cfg)
    show_flag = 1 if show else 0
    show_scale = 0.03 if show else 0.0

    solver = parent.addChild(f"{name}_solver")
    solver.addObject("EulerImplicitSolver",
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
    frames_node.addObject(
        "DiscreteCosseratMapping", name="cosseratMapping",
        curv_abs_input=curv_in, curv_abs_output=curv_out,
        input1=strain_mo.getLinkPath(), input2=base_mo.getLinkPath(),
        output=frames_mo.getLinkPath(), debug=0,
        radius=float(cfg["radius_m"]))

    grasp_target_mo = None
    if cfg.get("grasp_tip"):
        # Kinematic pose the tip is clamped to (the gripper), initialized at
        # the straight tip so the attachment starts at rest. Stiff external
        # rest-shape springs = the standard attachment for mapped frames.
        target = parent.addChild(f"{name}_graspTarget")
        grasp_target_mo = target.addObject(
            "MechanicalObject", template="Rigid3d", name="TargetMO",
            position=[list(frames[-1])], showObject=show_flag,
            showObjectScale=show_scale)
        frames_node.addObject(
            "RestShapeSpringsForceField", name="graspSpring", template="Rigid3d",
            stiffness=2e3, angularStiffness=50.0,
            points=[len(frames) - 1],
            external_rest_shape=grasp_target_mo.getLinkPath(),
            external_points=[0])

    return CableHandles(solver, base_mo, strain_mo, force_field, frames_mo, cfg,
                        grasp_target_mo)
