"""Unit tests for the local identification of the reduced cable dynamics."""

import numpy as np
import pytest

from cable_ts_control.local_identification import (
    fit_local_model,
    one_step_rmse,
    rollout_rmse,
    split_by_premise_vertex,
)

TRUE_A = np.array([[1.0, 0.0, 0.03, 0.0],
                   [0.0, 1.0, 0.0, 0.03],
                   [0.0, 0.0, 0.90, 0.0],
                   [0.0, 0.0, 0.0, 0.92]])
TRUE_B = np.array([[0.0, 0.0],
                   [0.0, 0.0],
                   [0.02, 0.0],
                   [0.0, 0.025]])


def simulate(n_samples=400, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    states = rng.normal(0.0, 0.05, size=(n_samples, 4))
    commands = rng.normal(0.0, 0.02, size=(n_samples, 2))
    next_states = states @ TRUE_A.T + commands @ TRUE_B.T
    if noise:
        next_states = next_states + rng.normal(0.0, noise, next_states.shape)
    return states, commands, next_states


def test_noise_free_fit_recovers_the_exact_matrices():
    a, b = fit_local_model(*simulate())

    assert a == pytest.approx(TRUE_A, abs=1e-9)
    assert b == pytest.approx(TRUE_B, abs=1e-9)


def test_fit_is_robust_to_moderate_measurement_noise():
    a, b = fit_local_model(*simulate(n_samples=4000, noise=1e-4, seed=1))

    assert a == pytest.approx(TRUE_A, abs=5e-3)
    assert b == pytest.approx(TRUE_B, abs=5e-3)


def test_ridge_regularisation_still_returns_a_usable_fit():
    a, b = fit_local_model(*simulate(), ridge=1e-9)

    assert a == pytest.approx(TRUE_A, abs=1e-4)
    assert b == pytest.approx(TRUE_B, abs=1e-4)


def test_one_step_error_is_zero_for_the_exact_model():
    states, commands, next_states = simulate()

    assert one_step_rmse(TRUE_A, TRUE_B, states, commands, next_states) == \
        pytest.approx(0.0, abs=1e-12)


def test_one_step_error_grows_for_a_wrong_model():
    states, commands, next_states = simulate()
    wrong_a = TRUE_A * 1.5

    assert one_step_rmse(wrong_a, TRUE_B, states, commands, next_states) > 1e-3


def test_rollout_error_is_zero_when_the_model_generated_the_data():
    rng = np.random.default_rng(3)
    commands = rng.normal(0.0, 0.01, size=(50, 2))
    state = np.array([0.02, -0.01, 0.0, 0.0])

    reference = []
    current = state.copy()
    for command in commands:
        current = TRUE_A @ current + TRUE_B @ command
        reference.append(current)

    assert rollout_rmse(TRUE_A, TRUE_B, state, commands, np.array(reference)) == \
        pytest.approx(0.0, abs=1e-12)


def test_rollout_error_exposes_a_slightly_wrong_model():
    """A model that looks fine one step ahead can still drift over a rollout."""
    rng = np.random.default_rng(4)
    commands = rng.normal(0.0, 0.01, size=(100, 2))
    state = np.array([0.05, 0.0, 0.0, 0.0])

    reference = []
    current = state.copy()
    for command in commands:
        current = TRUE_A @ current + TRUE_B @ command
        reference.append(current)
    reference = np.array(reference)

    drifted = TRUE_A.copy()
    drifted[2, 2] += 0.02

    exact_error = rollout_rmse(TRUE_A, TRUE_B, state, commands, reference)
    drifted_error = rollout_rmse(drifted, TRUE_B, state, commands, reference)

    assert exact_error < 1e-12
    assert drifted_error > 1e5 * max(exact_error, 1e-15)


def test_too_few_samples_are_rejected():
    states, commands, next_states = simulate(n_samples=3)

    with pytest.raises(ValueError):
        fit_local_model(states, commands, next_states)


def test_mismatched_sample_counts_are_rejected():
    states, commands, next_states = simulate()

    with pytest.raises(ValueError):
        fit_local_model(states, commands[:-1], next_states)


def test_mismatched_rollout_lengths_are_rejected():
    with pytest.raises(ValueError):
        rollout_rmse(TRUE_A, TRUE_B, np.zeros(4), np.zeros((5, 2)),
                     np.zeros((4, 4)))


# -- premise splitting -------------------------------------------------
def test_samples_are_assigned_to_the_nearest_premise_corner():
    bounds = [(-0.1, 0.1), (-0.05, 0.05)]
    premises = np.array([[-0.09, -0.04],   # low, low   -> rule 0
                         [-0.09, 0.04],    # low, high  -> rule 1
                         [0.09, -0.04],    # high, low  -> rule 2
                         [0.09, 0.04]])    # high, high -> rule 3

    groups = split_by_premise_vertex(premises, bounds)

    assert [group.tolist() for group in groups] == [[0], [1], [2], [3]]


def test_every_sample_lands_in_exactly_one_group():
    rng = np.random.default_rng(5)
    bounds = [(-0.1, 0.1), (-0.05, 0.05)]
    premises = rng.uniform(-0.1, 0.1, size=(500, 2))

    groups = split_by_premise_vertex(premises, bounds)

    assert sum(len(group) for group in groups) == 500
    assert len(np.unique(np.concatenate(groups))) == 500


def test_wrong_premise_width_is_rejected():
    with pytest.raises(ValueError):
        split_by_premise_vertex(np.zeros((10, 3)), [(-1.0, 1.0), (-1.0, 1.0)])
