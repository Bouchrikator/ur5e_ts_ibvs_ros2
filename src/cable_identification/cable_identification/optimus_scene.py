"""Optimus (ROUKF) estimator around the Cosserat cable of `cosserat_model`.

Scene layout (protocol §12): the physical cable is built unchanged by
`cosserat_model.build_cable`; Optimus objects are only inserted around it.

    root                 FilteringAnimationLoop, ROUKFilter
      cable_solver       OptimParams, StochasticStateWrapper (strains in state,
                         parameter reduced), BoundaryController
        rigidBase        RigidBaseMO
        cosseratCoordinate  cosseratCoordinateMO (Vec6d), hooke (EI <- @estimated_EI.value),
                            kelvinVoigt (coefficients re-synced to EI at every propagation)
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

Observations are per marker: a marker that is not observed (NaN position or
no finite positive variance) gets zero weight in R^-1, which is exactly the
dropped row; no observed marker at all is a prediction-only step (§8.4). The
innovation gate lives in the filter, after the sigma-point prediction:
NIS = nu' S^-1 nu against the chi-square quantile for the observed coordinates.
"""

import math

import numpy as np

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
            # the Kelvin-Voigt coefficients are proportional to the stiffness the sigma point
            # is propagated with (EI through the OptimParams Data link)
            cable.sync_damping()
            self.applications += 1

    return BoundaryController()


class OptimusCable:
    """Handles of the estimator scene."""

    def __init__(self, cable, params, wrapper, roukf, obs_source, obs_manager,
                 markers_mo, controller, observation_std):
        self.cable = cable
        self.params = params            # {name: OptimParams} in state order
        self.wrapper = wrapper
        self.roukf = roukf
        self.obs_source = obs_source
        self.obs_manager = obs_manager
        self.markers_mo = markers_mo
        self.controller = controller
        self.observation_std = float(observation_std)
        self.observed = np.zeros(len(cable.marker_indices), dtype=bool)   # markers of the pending observation

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

    def set_observation(self, points, variances=None, valid=True):
        """Observation consumed by the NEXT filter step: `points` (M x 3, model
        frame and marker order; a NaN row = marker not observed), `variances`
        (M x 3 [m^2], None = observation_std^2 everywhere; non-positive or
        non-finite = coordinate not observed). Unobserved coordinates get zero
        weight in R^-1. `valid=False` or nothing observed: prediction-only step."""
        n = len(self.observed)
        if not valid or points is None:
            self.observed[:] = False
            self.obs_source.trackedObservationsValid.value = False
            return
        pts = np.asarray(points, dtype=float).reshape(n, 3)
        var = (np.full((n, 3), self.observation_std ** 2) if variances is None
               else np.asarray(variances, dtype=float).reshape(n, 3))
        coord_ok = np.isfinite(pts) & np.isfinite(var) & (var > 0.0)
        self.observed[:] = coord_ok.all(axis=1)
        if not self.observed.any():
            self.obs_source.trackedObservationsValid.value = False
            return
        # a partially observed marker is dropped whole; the placeholder position is
        # never weighted, it only keeps the innovation vector finite
        var = np.where(self.observed[:, None], var, 0.0)
        pts = np.where(self.observed[:, None], pts, 0.0)
        self.obs_source.trackedObservations.value = pts.tolist()
        self.obs_manager.observationVariances.value = var.ravel().tolist()
        self.obs_source.trackedObservationsValid.value = True

    def correction_applied(self):
        """Whether the last filter step corrected the state (an observation was
        given and passed the innovation gate)."""
        return bool(self.roukf.correctionApplied.value)

    def nis(self):
        """(normalized innovation squared, observed coordinates) of the last step;
        (nan, 0) when the step had no observation."""
        return float(self.roukf.nis.value), int(3 * self.observed.sum())

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
        """RMS innovation per observed marker [m] of the last observation (nan if
        the last step had none)."""
        innov = np.asarray(self.roukf.reducedInnovation.value, dtype=float)
        if innov.size != 3 * len(self.observed) or not self.observed.any():
            return float("nan")
        return float(np.sqrt(np.mean(np.sum(innov.reshape(-1, 3)[self.observed] ** 2, axis=1))))


def lognormal_std(value, log_variance):
    """Physical std of a log-normal law with median `value` and log-variance s2."""
    if not (math.isfinite(log_variance) and log_variance >= 0.0):
        return float("nan")
    return float(value) * math.sqrt(math.expm1(log_variance))


def build_optimus_cable(root, cfg, parameters=("EI",), init_values=None, relative_std=0.30,
                        observation_std=0.002, innovation_gate_sigma=0.0, name="cable"):
    """Build cable + Optimus estimator under `root`, which must not carry an
    animation loop yet (FilteringAnimationLoop is added here).

    `parameters`: names among EI, GJ (one OptimParams each; the ROUKF reduced
    state concatenates them in this order). `init_values`: {name: value} or a
    sequence aligned with `parameters`; defaults to the config values.
    `innovation_gate_sigma`: chi-square gate on the NIS after prediction, in
    sigmas (0: off)."""
    from cable_identification import cosserat_model as cm

    if isinstance(parameters, str):
        parameters = (parameters,)
    parameters = tuple(parameters)
    if not parameters or len(set(parameters)) != len(parameters):
        raise ValueError(f"parameters must be a non-empty set of names, got {parameters!r}")
    for p in parameters:
        if p not in _PARAM_FIELD:
            raise ValueError(f"unsupported parameter {p!r}; one of {sorted(_PARAM_FIELD)}")
    if "GJ" in parameters and cfg.get("planar"):
        raise ValueError(
            "GJ is structurally unidentifiable with planar=true: the planar constraint fixes "
            "the torsional strain (kappa_x = 0), so dV/dGJ = 0 and the markers carry no GJ "
            "information; identify GJ with planar=false under a torsional excitation")
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
        observationErrorVarianceType="inverse", innovationGateSigma=float(innovation_gate_sigma),
        verbose=False)

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
        "StochasticStateWrapper", name="stateWrapper", template="Vec6d",
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
    # <filter, master (Vec6 strains), observed (Vec3 markers)>: the observations carry the
    # observed type (Optimus port, MappedStateObservationManager)
    obs_manager = markers.addObject(
        "MappedStateObservationManager", name="observationManager", template="double,Vec6d,Vec3d",
        observationStdev=float(observation_std), observationIndices=list(range(n_markers)),
        stateWrapper=wrapper.getLinkPath(), verbose=False)

    return OptimusCable(cable, params, wrapper, roukf, obs_source, obs_manager, markers_mo,
                        controller, observation_std)
