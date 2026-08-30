"""Tests for the corrected fuzzy identification and validation.

Covers the three things that were wrong before: the state signal used offline
did not match the runtime one, the model was fitted on crisp cells but executed
blended, and the "rollout" was evaluated over scattered samples.
"""

import numpy as np
import pytest

from cable_ts_control.local_identification import (
    blended_predict,
    blended_rollout,
    blended_rollout_rmse,
    fit_fuzzy_model,
    fit_local_model,
)
from cable_ts_control.state_filter import ModalVelocityFilter, velocity_series
from cable_ts_control.ts_model import rule_memberships

BOUNDS = [(-1.0, 1.0), (-1.0, 1.0)]


def reference_model(seed=0):
    """A known, contractive 4-rule blended model to identify back."""
    rng = np.random.default_rng(seed)
    a = [0.9 * np.eye(2) + 0.05 * rng.standard_normal((2, 2)) for _ in range(4)]
    b = [0.5 * rng.standard_normal((2, 1)) for _ in range(4)]
    return a, b


def sample_transitions(a_vertices, b_vertices, n, rng, offsets=None):
    """Independent transitions covering the whole premise box.

    Rolling a trajectory out to build fit data lets the state drift into one
    corner, where the memberships saturate and the other rules are never
    excited; sampling the box directly keeps every rule identifiable.
    """
    states = rng.uniform(-0.9, 0.9, size=(n, 2))
    commands = rng.standard_normal((n, 1)) * 0.2
    next_states = np.zeros_like(states)
    for k, (x, u) in enumerate(zip(states, commands)):
        h = rule_memberships(x[:2], BOUNDS)
        a = sum(w * m for w, m in zip(h, a_vertices))
        b = sum(w * m for w, m in zip(h, b_vertices))
        value = a @ x + b @ u
        if offsets is not None:
            value = value + sum(w * c for w, c in zip(h, offsets))
        next_states[k] = value
    return states, commands, next_states


def simulate(a_vertices, b_vertices, commands, x0, offsets=None):
    states, next_states = [], []
    x = np.asarray(x0, dtype=float)
    for u in commands:
        h = rule_memberships(x[:2], BOUNDS)
        a = sum(w * m for w, m in zip(h, a_vertices))
        b = sum(w * m for w, m in zip(h, b_vertices))
        nxt = a @ x + b @ u
        if offsets is not None:
            nxt = nxt + sum(w * c for w, c in zip(h, offsets))
        states.append(x)
        next_states.append(nxt)
        x = nxt
    return np.array(states), np.array(next_states)


# -- state signal consistency -------------------------------------------


def test_offline_velocity_matches_the_runtime_filter_sample_by_sample():
    """Identification and execution must see the same state signal."""
    rng = np.random.default_rng(1)
    modal = np.cumsum(rng.standard_normal((40, 3)) * 0.01, axis=0)
    dt = 0.04

    offline = velocity_series(modal, dt, alpha=0.4)

    online = ModalVelocityFilter(3, alpha=0.4)
    for index, q in enumerate(modal):
        assert online.update(q, dt) == pytest.approx(offline[index])


def test_the_first_sample_has_no_velocity():
    filt = ModalVelocityFilter(2, alpha=0.4)
    assert filt.update([0.3, 0.4], 0.04) == pytest.approx([0.0, 0.0])


def test_a_constant_slope_converges_to_that_slope():
    dt = 0.04
    modal = np.array([[k * dt] for k in range(400)])
    assert velocity_series(modal, dt, alpha=0.4)[-1][0] == pytest.approx(1.0, rel=1e-6)


def test_rejects_an_out_of_range_alpha():
    with pytest.raises(ValueError):
        ModalVelocityFilter(2, alpha=0.0)


# -- joint fuzzy identification -----------------------------------------


def test_the_fuzzy_regressor_is_rank_deficient_by_construction():
    """Complementary triangular memberships are affine in the premises.

    ``h_2 + h_3`` equals the normalised first premise exactly, so wherever a
    premise is also a state component the ``h_i x`` blocks are linearly
    dependent. Individual vertex matrices are therefore NOT identifiable, only
    the blended prediction is. This is why the fit takes the minimum-norm
    solution instead of solving a normal equation.
    """
    rng = np.random.default_rng(0)
    states = rng.uniform(-0.9, 0.9, size=(600, 2))
    commands = rng.standard_normal((600, 1)) * 0.2
    memberships = np.array([rule_memberships(x[:2], BOUNDS) for x in states])

    normalised = (states[:, 0] - BOUNDS[0][0]) / (BOUNDS[0][1] - BOUNDS[0][0])
    assert memberships[:, 2] + memberships[:, 3] == pytest.approx(normalised)

    features = np.hstack([states, commands])
    regressor = (memberships[:, :, None] * features[:, None, :]).reshape(
        len(states), 4 * 3)
    singular = np.linalg.svd(regressor, compute_uv=False)
    assert singular[-1] / singular[0] < 1e-12


def test_joint_fit_reproduces_the_generating_predictions():
    """The prediction is identifiable even though the parameters are not."""
    a_true, b_true = reference_model()
    rng = np.random.default_rng(2)
    states, commands, next_states = sample_transitions(a_true, b_true, 400, rng)

    memberships = np.array([rule_memberships(x[:2], BOUNDS) for x in states])
    a_fit, b_fit, offsets = fit_fuzzy_model(states, commands, next_states,
                                            memberships)

    assert offsets is None
    predicted = blended_predict(a_fit, b_fit, BOUNDS, states, commands)
    assert predicted == pytest.approx(next_states, abs=1e-9)


def test_joint_fit_reproduces_predictions_of_a_model_with_offsets():
    a_true, b_true = reference_model(3)
    offsets_true = [0.01 * np.array([1.0, -1.0]) * (i + 1) for i in range(4)]
    rng = np.random.default_rng(4)
    states, commands, next_states = sample_transitions(
        a_true, b_true, 600, rng, offsets_true)

    memberships = np.array([rule_memberships(x[:2], BOUNDS) for x in states])
    a_fit, b_fit, offsets = fit_fuzzy_model(states, commands, next_states,
                                            memberships, affine=True)

    assert offsets is not None and len(offsets) == 4
    predicted = blended_predict(a_fit, b_fit, BOUNDS, states, commands, offsets)
    assert predicted == pytest.approx(next_states, abs=1e-9)


def test_minimum_norm_keeps_the_vertex_matrices_bounded():
    """A null-space component could inflate the matrices handed to the LMIs."""
    a_true, b_true = reference_model(13)
    rng = np.random.default_rng(14)
    states, commands, next_states = sample_transitions(a_true, b_true, 500, rng)
    memberships = np.array([rule_memberships(x[:2], BOUNDS) for x in states])

    a_fit, b_fit, _ = fit_fuzzy_model(states, commands, next_states, memberships)
    largest = max(np.linalg.norm(m) for m in a_fit + b_fit)
    reference = max(np.linalg.norm(m) for m in a_true + b_true)
    assert largest < 10.0 * reference


def test_joint_fit_beats_independent_crisp_fits_on_held_out_data():
    """The blended fit matches the model that actually runs, so it should not
    be worse than fitting each cell in isolation."""
    a_true, b_true = reference_model(5)
    rng = np.random.default_rng(6)
    states, commands, next_states = sample_transitions(a_true, b_true, 600, rng)

    split = 400
    train = slice(0, split)
    memberships = np.array([rule_memberships(x[:2], BOUNDS)
                            for x in states[train]])
    a_joint, b_joint, _ = fit_fuzzy_model(
        states[train], commands[train], next_states[train], memberships)

    # Independent crisp fits: one model per hard-assigned cell.
    labels = np.argmax(memberships, axis=1)
    a_crisp, b_crisp = [], []
    for rule in range(4):
        pick = np.flatnonzero(labels == rule)
        a_r, b_r = fit_local_model(states[train][pick], commands[train][pick],
                                   next_states[train][pick], ridge=1e-10)
        a_crisp.append(a_r)
        b_crisp.append(b_r)

    held = slice(split, len(states))
    joint = blended_predict(a_joint, b_joint, BOUNDS, states[held], commands[held])
    crisp = blended_predict(a_crisp, b_crisp, BOUNDS, states[held], commands[held])

    joint_err = np.sqrt(np.mean((joint - next_states[held]) ** 2))
    crisp_err = np.sqrt(np.mean((crisp - next_states[held]) ** 2))
    assert joint_err <= crisp_err


# -- blended rollout -----------------------------------------------------


def test_blended_rollout_reproduces_the_generating_model_exactly():
    a_true, b_true = reference_model(7)
    rng = np.random.default_rng(8)
    commands = rng.standard_normal((60, 1)) * 0.1
    x0 = np.array([0.1, -0.05])
    _, next_states = simulate(a_true, b_true, commands, x0)

    predicted = blended_rollout(a_true, b_true, BOUNDS, x0, commands)
    assert predicted == pytest.approx(next_states, abs=1e-12)


def test_blended_rollout_recomputes_memberships_as_the_state_moves():
    """A frozen local model is not the model; the weights must follow."""
    a_true, b_true = reference_model(9)
    commands = np.zeros((40, 1))
    x0 = np.array([-0.9, 0.9])

    blended = blended_rollout(a_true, b_true, BOUNDS, x0, commands)

    frozen_weights = rule_memberships(x0[:2], BOUNDS)
    a_frozen = sum(w * m for w, m in zip(frozen_weights, a_true))
    b_frozen = sum(w * m for w, m in zip(frozen_weights, b_true))
    state = x0.copy()
    frozen = []
    for u in commands:
        state = a_frozen @ state + b_frozen @ u
        frozen.append(state)

    assert not np.allclose(blended, np.array(frozen))


def test_rollout_rmse_is_zero_for_the_generating_model():
    a_true, b_true = reference_model(10)
    rng = np.random.default_rng(11)
    commands = rng.standard_normal((50, 1)) * 0.1
    x0 = np.array([0.05, 0.05])
    _, next_states = simulate(a_true, b_true, commands, x0)

    assert blended_rollout_rmse(a_true, b_true, BOUNDS, x0, commands,
                                next_states) == pytest.approx(0.0, abs=1e-12)


def test_rollout_rejects_mismatched_lengths():
    a_true, b_true = reference_model(12)
    with pytest.raises(ValueError):
        blended_rollout_rmse(a_true, b_true, BOUNDS, np.zeros(2),
                             np.zeros((10, 1)), np.zeros((9, 2)))
