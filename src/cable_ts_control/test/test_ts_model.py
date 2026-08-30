"""Unit tests for the Takagi-Sugeno cable model and the PDC law."""

import numpy as np
import pytest

from cable_ts_control.ts_model import (
    CableTsModel,
    rule_memberships,
    triangular_memberships,
)

BOUNDS = [(-0.1, 0.1), (-0.05, 0.05)]


def make_model(gains=None, lyapunov=None):
    """Four stable vertices of a 4-state / 2-input reduced cable."""
    a_vertices, b_vertices = [], []
    for index, decay in enumerate((0.90, 0.92, 0.94, 0.96)):
        a = np.array([[1.0, 0.0, 0.03, 0.0],
                      [0.0, 1.0, 0.0, 0.03],
                      [0.0, 0.0, decay, 0.0],
                      [0.0, 0.0, 0.0, decay]])
        b = np.array([[0.0, 0.0],
                      [0.0, 0.0],
                      [0.02 + 0.001 * index, 0.0],
                      [0.0, 0.02 + 0.001 * index]])
        a_vertices.append(a)
        b_vertices.append(b)
    return CableTsModel(a_vertices, b_vertices, BOUNDS, 1.0 / 30.0,
                        gains=gains, lyapunov=lyapunov)


# -- memberships -------------------------------------------------------
def test_membership_at_the_lower_bound_is_fully_low():
    assert triangular_memberships(-0.1, -0.1, 0.1) == pytest.approx((1.0, 0.0))


def test_membership_at_the_upper_bound_is_fully_high():
    assert triangular_memberships(0.1, -0.1, 0.1) == pytest.approx((0.0, 1.0))


def test_membership_at_the_midpoint_is_balanced():
    assert triangular_memberships(0.0, -0.1, 0.1) == pytest.approx((0.5, 0.5))


@pytest.mark.parametrize("value", [-5.0, 5.0])
def test_membership_saturates_outside_the_range(value):
    low, high = triangular_memberships(value, -0.1, 0.1)

    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert low + high == pytest.approx(1.0)


def test_degenerate_premise_range_is_rejected():
    with pytest.raises(ValueError):
        triangular_memberships(0.0, 0.1, 0.1)


def test_two_premises_give_four_rules_that_sum_to_one():
    weights = rule_memberships([0.02, -0.01], BOUNDS)

    assert weights.size == 4
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()


def test_a_premise_corner_activates_exactly_one_rule():
    weights = rule_memberships([0.1, 0.05], BOUNDS)  # both premises high

    assert weights == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_rule_ordering_is_row_major_over_the_premises():
    # first premise low, second high -> bits (0, 1) -> rule index 1
    weights = rule_memberships([-0.1, 0.05], BOUNDS)

    assert weights == pytest.approx([0.0, 1.0, 0.0, 0.0])


def test_mismatched_premises_and_bounds_are_rejected():
    with pytest.raises(ValueError):
        rule_memberships([0.0], BOUNDS)


# -- model -------------------------------------------------------------
def test_model_reports_its_dimensions():
    model = make_model()

    assert (model.state_dim, model.input_dim, model.n_rules) == (4, 2, 4)


def test_blending_at_a_corner_returns_that_vertex():
    model = make_model()

    a, b, _ = model.blend([0.1, 0.05])

    assert a == pytest.approx(model.a_vertices[3])
    assert b == pytest.approx(model.b_vertices[3])


def test_blend_is_a_convex_combination_of_the_vertices():
    model = make_model()

    a, _, weights = model.blend([0.0, 0.0])
    expected = sum(w * m for w, m in zip(weights, model.a_vertices))

    assert a == pytest.approx(expected)


def test_prediction_matches_the_blended_matrices():
    model = make_model()
    state = np.array([0.02, -0.01, 0.0, 0.0])
    command = np.array([0.01, 0.02])

    a, b, _ = model.blend(state[:2])

    assert model.predict(state, command) == pytest.approx(a @ state + b @ command)


def test_wrong_number_of_vertices_is_rejected():
    model = make_model()

    with pytest.raises(ValueError):
        CableTsModel(model.a_vertices[:3], model.b_vertices[:3], BOUNDS, 0.03)


def test_non_positive_sample_time_is_rejected():
    model = make_model()

    with pytest.raises(ValueError):
        CableTsModel(model.a_vertices, model.b_vertices, BOUNDS, 0.0)


# -- PDC law -----------------------------------------------------------
def test_control_without_gains_is_refused():
    with pytest.raises(RuntimeError):
        make_model().control(np.zeros(4), np.zeros(4))


def test_control_is_zero_on_the_target():
    gains = [np.full((2, 4), 0.5) for _ in range(4)]
    model = make_model(gains=gains)
    reference = np.array([0.02, -0.01, 0.0, 0.0])

    command, _ = model.control(reference, reference)

    assert command == pytest.approx(np.zeros(2))


def test_control_opposes_the_shape_error():
    gains = [np.hstack([np.eye(2), np.zeros((2, 2))]) for _ in range(4)]
    model = make_model(gains=gains)
    state = np.array([0.03, 0.0, 0.0, 0.0])

    command, _ = model.control(state, np.zeros(4))

    assert command[0] < 0.0


def test_control_weights_sum_to_one():
    gains = [np.zeros((2, 4)) for _ in range(4)]
    _, weights = make_model(gains=gains).control(
        np.array([0.01, 0.0, 0.0, 0.0]), np.zeros(4))

    assert weights.sum() == pytest.approx(1.0)


def test_lyapunov_value_is_zero_on_the_target_and_positive_elsewhere():
    model = make_model(lyapunov=np.eye(4))
    reference = np.zeros(4)

    assert model.lyapunov_value(reference, reference) == pytest.approx(0.0)
    assert model.lyapunov_value(np.array([0.1, 0.0, 0.0, 0.0]), reference) > 0.0


def test_uncontrolled_open_loop_is_already_marginally_stable():
    """The A vertices have a unit mode, so feedback is what must stabilise it."""
    model = make_model()
    radii = [max(abs(np.linalg.eigvals(a))) for a in model.a_vertices]

    assert all(radius >= 1.0 - 1e-9 for radius in radii)


# -- persistence -------------------------------------------------------
def test_yaml_round_trip_preserves_matrices_and_gains(tmp_path):
    gains = [np.full((2, 4), 0.1 * (i + 1)) for i in range(4)]
    model = make_model(gains=gains, lyapunov=np.eye(4) * 2.0)

    path = tmp_path / "model.yaml"
    model.save(path)
    loaded = CableTsModel.load(path)

    assert loaded.premise_bounds == model.premise_bounds
    assert loaded.sample_time == pytest.approx(model.sample_time)
    for original, restored in zip(model.a_vertices, loaded.a_vertices):
        assert restored == pytest.approx(original)
    for original, restored in zip(model.b_vertices, loaded.b_vertices):
        assert restored == pytest.approx(original)
    for original, restored in zip(model.gains, loaded.gains):
        assert restored == pytest.approx(original)
    assert loaded.lyapunov == pytest.approx(model.lyapunov)


def test_gains_can_be_loaded_from_a_separate_file(tmp_path):
    import yaml

    model = make_model()
    model_path = tmp_path / "model.yaml"
    model.save(model_path)

    gains = [np.full((2, 4), 0.25) for _ in range(4)]
    gains_path = tmp_path / "gains.yaml"
    with open(gains_path, "w") as f:
        yaml.safe_dump({"cable_ts_model": {
            "K": [g.flatten().tolist() for g in gains],
            "P": np.eye(4).flatten().tolist()}}, f)

    loaded = CableTsModel.load(model_path, gains_path)

    assert loaded.gains is not None
    assert loaded.gains[0] == pytest.approx(gains[0])
    assert loaded.lyapunov == pytest.approx(np.eye(4))


# -- the mode count is not the state dimension ---------------------------


def test_n_modes_defaults_to_half_the_state():
    """Backward compatible for the plain [q, qdot] state."""
    assert make_model().n_modes == 2


def test_n_modes_is_independent_of_the_state_dimension():
    """With the gripper appended the state is [q, qdot, p_g]: 6 dims, 2 modes.

    Inferring it as state_dim // 2 gives 3 and silently mis-sizes every target.
    """
    a = [np.eye(6) for _ in range(4)]
    b = [np.zeros((6, 2)) for _ in range(4)]
    model = CableTsModel(a, b, [(-1.0, 1.0), (-1.0, 1.0)], 0.04, n_modes=2)

    assert model.state_dim == 6
    assert model.n_modes == 2


def test_n_modes_survives_a_save_load_round_trip(tmp_path):
    a = [np.eye(6) for _ in range(4)]
    b = [np.zeros((6, 2)) for _ in range(4)]
    model = CableTsModel(a, b, [(-1.0, 1.0), (-1.0, 1.0)], 0.04, n_modes=2)

    path = tmp_path / "model.yaml"
    model.save(path)

    assert CableTsModel.load(path).n_modes == 2
