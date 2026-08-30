"""Behaviour tests for the log-space parameter UKF.

The observation model is an analytic cantilever whose deflection scales as
``1/EI``, so the true parameter is known exactly and convergence can be
asserted instead of mocked.
"""

import numpy as np
import pytest

from cable_identification.parameter_filter import LogParameterUKF

LENGTH = 0.5
S_OVER_L = np.array([0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00])
ARC = S_OVER_L * LENGTH
LOAD = 0.05          # N/m, keeps the tip deflection at a few centimetres
AXIAL_LOAD = 40.0    # N, large enough that EA moves the markers past the noise
NOISE = 0.002        # 2 mm marker noise, as in the launch defaults

TRUE_EI = 0.010      # cable_truth.yaml
START_EI = 0.006     # cable_estimator_initial.yaml


def cantilever(ei, arc=ARC, load=LOAD, length=LENGTH):
    """Planar deflection of a uniformly loaded cantilever, flattened [x, y]."""
    y = load / (24.0 * ei) * (arc ** 4 - 4.0 * length * arc ** 3
                              + 6.0 * length ** 2 * arc ** 2)
    return np.column_stack([arc, y]).ravel()


def stretched_cantilever(theta, arc=ARC, load=LOAD, length=LENGTH):
    """Bending from EI and axial stretch from EA, so both are identifiable."""
    ei, ea = float(theta[0]), float(theta[1])
    x = arc * (1.0 + AXIAL_LOAD / ea)
    y = load / (24.0 * ei) * (arc ** 4 - 4.0 * length * arc ** 3
                              + 6.0 * length ** 2 * arc ** 2)
    return np.column_stack([x, y]).ravel()


def make_filter(**overrides):
    kwargs = dict(
        names=("EI",),
        initial_values=(START_EI,),
        initial_relative_std=(0.6,),
        process_relative_std=(0.01,),
    )
    kwargs.update(overrides)
    return LogParameterUKF(**kwargs)


def run(ukf, observe, truth_fn, steps, rng, noise=NOISE, **update_kwargs):
    """Feed the filter repeated noisy observations of the same true shape."""
    result = None
    for _ in range(steps):
        ukf.predict()
        z = truth_fn + rng.normal(0.0, noise, size=truth_fn.size)
        result = ukf.update(observe, z, noise, point_dim=2, **update_kwargs)
    return result


# -- convergence ---------------------------------------------------------


def test_recovers_the_true_bending_stiffness():
    ukf = make_filter()
    rng = np.random.default_rng(7)
    run(ukf, lambda t: cantilever(t[0]), cantilever(TRUE_EI), 60, rng)

    relative_error = abs(ukf.value_of("EI") - TRUE_EI) / TRUE_EI
    assert relative_error < 0.05, f"EI={ukf.value_of('EI')}, want {TRUE_EI}"


def test_starts_biased_so_the_test_is_not_circular():
    ukf = make_filter()
    assert abs(ukf.value_of("EI") - TRUE_EI) / TRUE_EI > 0.3


def test_uncertainty_shrinks_as_evidence_accumulates():
    ukf = make_filter()
    rng = np.random.default_rng(11)
    before = ukf.relative_std[0]
    run(ukf, lambda t: cantilever(t[0]), cantilever(TRUE_EI), 40, rng)
    assert ukf.relative_std[0] < before


def test_innovation_decreases_as_the_estimate_improves():
    ukf = make_filter()
    rng = np.random.default_rng(3)
    truth = cantilever(TRUE_EI)
    first = run(ukf, lambda t: cantilever(t[0]), truth, 1, rng)
    last = run(ukf, lambda t: cantilever(t[0]), truth, 40, rng)
    assert last.innovation < first.innovation


def test_converges_from_an_overestimate_too():
    ukf = make_filter(initial_values=(0.020,))
    rng = np.random.default_rng(5)
    run(ukf, lambda t: cantilever(t[0]), cantilever(TRUE_EI), 60, rng)
    assert abs(ukf.value_of("EI") - TRUE_EI) / TRUE_EI < 0.05


def test_the_axial_effect_is_above_the_noise_floor():
    """Guards the test itself: an unobservable parameter proves nothing."""
    truth = np.array([TRUE_EI, 4000.0])
    shifted = np.array([TRUE_EI, 4000.0 * 1.3])
    separation = np.abs(stretched_cantilever(truth)
                        - stretched_cantilever(shifted)).max()
    assert separation > 5e-4


def test_identifies_bending_and_axial_stiffness_together():
    truth = np.array([TRUE_EI, 4000.0])
    ukf = LogParameterUKF(
        names=("EI", "EA"),
        initial_values=(START_EI, 8000.0),
        initial_relative_std=(0.6, 0.6),
        process_relative_std=(0.01, 0.01),
    )
    rng = np.random.default_rng(13)
    run(ukf, stretched_cantilever, stretched_cantilever(truth), 80, rng,
        noise=5e-4)

    estimate = ukf.values
    assert abs(estimate[0] - truth[0]) / truth[0] < 0.10
    assert abs(estimate[1] - truth[1]) / truth[1] < 0.30


# -- robustness ----------------------------------------------------------


def test_the_innovation_gate_rejects_outliers():
    """The configuration the launch actually uses: outliers never land."""
    ukf = make_filter()
    before = ukf.value_of("EI")
    rng = np.random.default_rng(17)
    for _ in range(30):
        ukf.predict()
        result = ukf.update(lambda t: cantilever(t[0]),
                            rng.normal(0.0, 10.0, size=ARC.size * 2), NOISE,
                            point_dim=2, max_innovation=0.05)
        assert not result.accepted
        assert result.reason == "innovation above bound"
    assert ukf.value_of("EI") == before


def test_parameters_stay_positive_under_absurd_observations():
    """Even ungated, log(theta) must not underflow a stiffness to zero."""
    ukf = make_filter()
    rng = np.random.default_rng(17)
    for _ in range(30):
        ukf.predict()
        ukf.update(lambda t: cantilever(t[0]),
                   rng.normal(0.0, 10.0, size=ARC.size * 2), NOISE, point_dim=2)
    assert np.all(ukf.values > 0.0)
    assert np.all(np.isfinite(ukf.values))


def test_a_plausible_observation_passes_the_innovation_gate():
    ukf = make_filter()
    result = ukf.update(lambda t: cantilever(t[0]), cantilever(TRUE_EI), NOISE,
                        point_dim=2, max_innovation=0.5)
    assert result.accepted


def test_bounds_clamp_the_estimate():
    bounds = [(0.008, 0.012)]
    ukf = make_filter(initial_values=(0.010,), bounds=bounds)
    rng = np.random.default_rng(19)
    run(ukf, lambda t: cantilever(t[0]), cantilever(0.030), 60, rng)
    assert 0.008 <= ukf.value_of("EI") <= 0.012


def test_rejects_the_update_when_uncertainty_exceeds_the_bound():
    ukf = make_filter()
    before = ukf.value_of("EI")
    result = ukf.update(lambda t: cantilever(t[0]), cantilever(TRUE_EI),
                        NOISE, point_dim=2, max_relative_std=1e-6)
    assert not result.accepted
    assert result.reason == "uncertainty above bound"
    assert ukf.value_of("EI") == before


def test_rejects_non_finite_model_output():
    ukf = make_filter()
    before = ukf.value_of("EI")
    result = ukf.update(lambda t: np.full(ARC.size * 2, np.nan),
                        cantilever(TRUE_EI), NOISE, point_dim=2)
    assert not result.accepted
    assert ukf.value_of("EI") == before


def test_dropped_markers_are_ignored_but_still_inform():
    ukf = make_filter()
    rng = np.random.default_rng(23)
    mask = np.ones(ARC.size * 2, dtype=bool)
    mask[:4] = False           # first two markers occluded
    run(ukf, lambda t: cantilever(t[0]), cantilever(TRUE_EI), 60, rng, mask=mask)
    assert abs(ukf.value_of("EI") - TRUE_EI) / TRUE_EI < 0.10


def test_all_markers_dropped_is_a_no_op():
    ukf = make_filter()
    before = ukf.value_of("EI")
    result = ukf.update(lambda t: cantilever(t[0]), cantilever(TRUE_EI), NOISE,
                        mask=np.zeros(ARC.size * 2, dtype=bool), point_dim=2)
    assert not result.accepted
    assert result.reason == "no valid markers"
    assert ukf.value_of("EI") == before


def test_predict_only_inflates_uncertainty():
    ukf = make_filter()
    value, spread = ukf.value_of("EI"), ukf.relative_std[0]
    ukf.predict()
    assert ukf.value_of("EI") == value
    assert ukf.relative_std[0] > spread


# -- contract ------------------------------------------------------------


def test_reports_physical_std_not_log_std():
    ukf = make_filter()
    assert ukf.std_dev[0] == pytest.approx(
        ukf.value_of("EI") * ukf.relative_std[0])


def test_as_dict_is_keyed_by_parameter_name():
    ukf = LogParameterUKF(("EI", "GJ"), (0.006, 0.012), (0.5, 0.5), (0.01, 0.01))
    assert sorted(ukf.as_dict()) == ["EI", "GJ"]
    assert ukf.as_dict()["GJ"] == pytest.approx(0.012)


def test_sigma_points_are_symmetric_about_the_mean():
    ukf = make_filter()
    points = ukf.sigma_points()
    assert points.shape == (3, 1)
    assert points[1] - points[0] == pytest.approx(points[0] - points[2])


def test_rejects_non_positive_initial_values():
    with pytest.raises(ValueError):
        make_filter(initial_values=(0.0,))


def test_rejects_mismatched_observation_length():
    ukf = make_filter()
    with pytest.raises(ValueError):
        ukf.update(lambda t: np.zeros(3), cantilever(TRUE_EI), NOISE, point_dim=2)
