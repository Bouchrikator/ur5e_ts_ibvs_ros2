"""Optimus (ROUKF) estimator around the Cosserat cable of `cosserat_model`.

Scene layout (protocol §12): the physical cable is built unchanged by
`cosserat_model.build_cable`; Optimus objects are only inserted around it.

    root                 FilteringAnimationLoop, ROUKFilter
      cable_solver       OptimParams, StochasticStateWrapper (strains in state,
                         parameter reduced), BoundaryController
        rigidBase        RigidBaseMO
        cosseratCoordinate  cosseratCoordinateMO, hooke (EI <- @estimated_EI.value)
          frames         FramesMO, DiscreteCosseratMapping
            markers      Vec3 points at the marker frames (RigidMapping),
                         SimulatedStateObservationSource (trackedObservations),
                         MappedStateObservationManager

Two contracts this module owns:

* one parameter per experiment (protocol §6.2), estimated in log space
  (`transformParams=exponential`) so it stays strictly positive;
* the boundary pose is written once per filter step by the caller and
  re-applied by the BoundaryController on every AnimateBeginEvent, i.e. once
  per sigma-point propagation, so every sigma point sees the same g[k] (§8.2).
"""

import math

# Physical parameter name -> BeamHookeLawForceField data field
_PARAM_FIELD = {"EI": "EI", "GJ": "GI"}


def lognormal_sigma(relative_std):
    """Log-space standard deviation of a log-normal law with coefficient of
    variation `relative_std` = sigma_theta / mu_theta."""
    c = float(relative_std)
    if not c > 0.0:
        raise ValueError("relative_std must be > 0")
    return math.sqrt(math.log1p(c * c))


def optimus_stdev(relative_std):
    """`stdev` to give OptimParams for `transformParams=exponential`.

    Optimus computes the initial log-space variance as log(stdev)^2, so the
    value is exp(sigma_q), NOT sigma_theta (protocol §6.3).
    """
    return math.exp(lognormal_sigma(relative_std))


def initial_log_variance(stdev_optimus):
    """What OptimParams::getInitVariance returns for the exponential transform."""
    return math.log(float(stdev_optimus)) ** 2


def make_boundary_controller(cable):
    """Controller re-applying the current gripper pose to the cable base on every
    AnimateBeginEvent, so each sigma-point propagation of a filter step starts
    from the identical boundary condition. `pose` is set by the driver between
    filter steps only."""
    import Sofa.Core

    class BoundaryController(Sofa.Core.Controller):
        def __init__(self):
            super().__init__()
            self.name = "BoundaryController"
            self.pose = [float(v) for v in cable.base_mo.position.value[0]]
            self.applications = 0

        def onAnimateBeginEvent(self, _event):
            cable.set_base_pose(self.pose)
            with cable.base_mo.velocity.writeable() as v:
                v[:] = 0.0
            self.applications += 1

    return BoundaryController()


class OptimusCable:
    """Handles of the estimator scene."""

    def __init__(self, cable, params, wrapper, roukf, obs_source, obs_manager,
                 markers_mo, controller):
        self.cable = cable
        self.params = params            # {name: OptimParams} in state order
        self.wrapper = wrapper
        self.roukf = roukf
        self.obs_source = obs_source
        self.obs_manager = obs_manager
        self.markers_mo = markers_mo
        self.controller = controller

    @property
    def parameters(self):
        return tuple(self.params)

    def set_boundary_pose(self, pose7):
        """Gripper pose for the NEXT filter step. Written to the base now, so the
        mapped frames the sigma-point propagations start from already reflect
        it, and re-applied by the controller at every propagation (§8.2)."""
        self.controller.pose = [float(v) for v in pose7]
        self.cable.set_base_pose(self.controller.pose)
        with self.cable.base_mo.velocity.writeable() as v:
            v[:] = 0.0

    def set_observation(self, points, valid=True):
        """Observed marker positions (M x 3, same frame and order as the model
        markers) consumed by the NEXT filter step. `valid=False` keeps the
        previous points but makes the step prediction-only (protocol §8.4)."""
        if valid:
            self.obs_source.trackedObservations.value = [[float(c) for c in p] for p in points]
        self.obs_source.trackedObservationsValid.value = bool(valid)

    def estimates(self):
        """Physical values of the estimated parameters (after the last correction)."""
        return {n: float(p.findData("value").value[0]) for n, p in self.params.items()}

    def estimate(self, name=None):
        values = self.estimates()
        return values[name] if name else next(iter(values.values()))

    def log_variances(self):
        v = [float(x) for x in self.roukf.reducedVariance.value]
        return {n: (v[i] if i < len(v) else float("nan")) for i, n in enumerate(self.params)}

    def log_variance(self, name=None):
        values = self.log_variances()
        return values[name] if name else next(iter(values.values()))

    def std_devs(self):
        """Physical standard deviations of the log-normal posteriors."""
        return {n: lognormal_std(v, s2) for (n, v), s2 in
                zip(self.estimates().items(), self.log_variances().values())}

    def innovation_rms(self):
        """RMS innovation per marker [m] of the last correction (nan if none yet)."""
        innov = [float(x) for x in self.roukf.reducedInnovation.value]
        if not innov:
            return float("nan")
        return math.sqrt(sum(x * x for x in innov) / (len(innov) // 3))


def lognormal_std(value, log_variance):
    """Physical std of a log-normal law with median `value` and log-variance s2."""
    if not (math.isfinite(log_variance) and log_variance >= 0.0):
        return float("nan")
    return float(value) * math.sqrt(math.expm1(log_variance))


def build_optimus_cable(root, cfg, parameters=("EI",), init_values=None, relative_std=0.30,
                        observation_std=0.002, name="cable"):
    """Build cable + Optimus estimator under `root`, which must not carry an
    animation loop yet (FilteringAnimationLoop is added here).

    `parameters`: names among EI, GJ (one OptimParams each; the ROUKF reduced
    state concatenates them in this order). `init_values`: {name: value} or a
    sequence aligned with `parameters`; defaults to the config values."""
    from cable_identification import cosserat_model as cm

    if isinstance(parameters, str):
        parameters = (parameters,)
    parameters = tuple(parameters)
    if not parameters or len(set(parameters)) != len(parameters):
        raise ValueError(f"parameters must be a non-empty set of names, got {parameters!r}")
    for p in parameters:
        if p not in _PARAM_FIELD:
            raise ValueError(f"unsupported parameter {p!r}; one of {sorted(_PARAM_FIELD)}")
    defaults = {"EI": cfg["EI_Nm2"], "GJ": cfg["GJ_Nm2"]}
    if init_values is None:
        init_values = {p: defaults[p] for p in parameters}
    elif not isinstance(init_values, dict):
        init_values = dict(zip(parameters, init_values))
    for p in parameters:
        v = float(init_values.get(p, defaults[p]))
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"initial value of {p} must be finite and > 0, got {v!r}")
        init_values[p] = v
    if isinstance(relative_std, dict):
        rel = {p: float(relative_std[p]) for p in parameters}
    elif isinstance(relative_std, (list, tuple)):
        rel = dict(zip(parameters, (float(v) for v in relative_std)))
    else:
        rel = {p: float(relative_std) for p in parameters}
    for p in parameters:
        if not (math.isfinite(rel.get(p, float("nan"))) and rel[p] > 0.0):
            raise ValueError(f"relative_std of {p} must be finite and > 0, got {rel.get(p)!r}")

    root.addObject("RequiredPlugin", name="optimus_plugin",
                   pluginName=["Optimus", "Sofa.Component.Mapping.NonLinear"])
    root.addObject("FilteringAnimationLoop", name="filterLoop", verbose=False)
    roukf = root.addObject(
        "ROUKFilter", name="roukf", sigmaTopology="Simplex", useBlasToMultiply=False,
        observationErrorVarianceType="inverse", verbose=False)

    cable = cm.build_cable(root, cfg, name=name)
    solver = cable.solver_node

    # Objects init in creation order and before child nodes, so the parameters'
    # init() (value <- initValue) precedes the force field's; the tracked Data
    # link then rebuilds m_K_section on every later change (Cosserat patch 0001).
    params = {}
    for p in parameters:
        params[p] = solver.addObject(
            "OptimParams", name=f"estimated_{p}", template="Vector", optimize=True,
            numParams=1, initValue=[init_values[p]], stdev=[optimus_stdev(rel[p])],
            transformParams="exponential")
        cable.force_field.findData(_PARAM_FIELD[p]).setParent(params[p].findData("value"))

    wrapper = solver.addObject(
        "StochasticStateWrapper", name="stateWrapper", template="Vec3d",
        mstate="cosseratCoordinate/cosseratCoordinateMO",
        mappedState="rigidBase/frames/markers/markersMO",
        estimatePosition=True, estimateVelocity=False, positionStdev=[0.0],
        verbose=False)
    controller = make_boundary_controller(cable)
    solver.addObject(controller)

    frames_node = cable.frames_mo.getContext()
    markers = frames_node.addChild("markers")
    n_markers = len(cable.marker_indices)
    markers_mo = markers.addObject(
        "MechanicalObject", template="Vec3d", name="markersMO",
        position=[[0.0, 0.0, 0.0]] * n_markers)
    markers.addObject(
        "RigidMapping", name="markerMapping", input=cable.frames_mo.getLinkPath(),
        output=markers_mo.getLinkPath(), rigidIndexPerPoint=list(cable.marker_indices),
        globalToLocalCoords=False)
    obs_source = markers.addObject(
        "SimulatedStateObservationSource", name="observations", template="Vec3d",
        trackedObservations=[[0.0, 0.0, 0.0]] * n_markers)
    obs_manager = markers.addObject(
        "MappedStateObservationManager", name="observationManager",
        observationStdev=float(observation_std), observationIndices=list(range(n_markers)),
        stateWrapper=wrapper.getLinkPath(), verbose=False)

    return OptimusCable(cable, params, wrapper, roukf, obs_source, obs_manager, markers_mo,
                        controller)
