"""Unit tests for the observation corruption model."""

import numpy as np
import pytest

from cable_perception.observation_model import DelayLine, ObservationCorruptor

EXACT = np.array([[0.1, 0.0, 0.0],
                  [0.2, 0.0, 0.0],
                  [0.3, 0.0, 0.0],
                  [0.4, 0.0, 0.0]])


def test_noise_is_zero_mean_and_has_the_requested_spread():
    corruptor = ObservationCorruptor(noise_std_m=0.005, seed=1)
    errors = np.concatenate(
        [corruptor.corrupt(EXACT)[0] - EXACT for _ in range(2000)])

    assert np.allclose(errors.mean(axis=0), 0.0, atol=2e-4)
    assert np.allclose(errors.std(axis=0), 0.005, rtol=0.1)


def test_zero_noise_reproduces_the_exact_positions():
    noisy, valid = ObservationCorruptor(noise_std_m=0.0, seed=3).corrupt(EXACT)

    assert np.array_equal(noisy, EXACT)
    assert valid.all()


def test_dropout_rate_matches_the_requested_probability():
    corruptor = ObservationCorruptor(dropout_probability=0.25, seed=7)
    valid = np.concatenate([corruptor.corrupt(EXACT)[1] for _ in range(2000)])

    assert valid.mean() == pytest.approx(0.75, abs=0.02)


def test_no_dropout_keeps_every_marker():
    _, valid = ObservationCorruptor(dropout_probability=0.0, seed=5).corrupt(EXACT)

    assert valid.all()


def test_same_seed_replays_the_same_experiment():
    a = ObservationCorruptor(noise_std_m=0.003, dropout_probability=0.2, seed=42)
    b = ObservationCorruptor(noise_std_m=0.003, dropout_probability=0.2, seed=42)

    noisy_a, valid_a = a.corrupt(EXACT)
    noisy_b, valid_b = b.corrupt(EXACT)

    assert np.array_equal(noisy_a, noisy_b)
    assert np.array_equal(valid_a, valid_b)


def test_covariance_is_isotropic_and_matches_the_noise():
    covariance = ObservationCorruptor(noise_std_m=0.002).covariance()

    assert covariance == pytest.approx(
        [4e-6, 0.0, 0.0, 0.0, 4e-6, 0.0, 0.0, 0.0, 4e-6])


@pytest.mark.parametrize("kwargs", [
    {"noise_std_m": -0.001},
    {"dropout_probability": 1.5},
    {"dropout_probability": -0.1},
])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ObservationCorruptor(**kwargs)


def test_wrong_shaped_input_is_rejected():
    with pytest.raises(ValueError):
        ObservationCorruptor().corrupt(np.zeros((4, 2)))


def test_delay_line_holds_samples_until_the_latency_elapsed():
    line = DelayLine(delay_s=0.1)
    line.push(1.0, "sample")

    assert line.pop_ready(1.05) == []
    assert len(line) == 1
    assert line.pop_ready(1.10) == [(1.0, "sample")]
    assert len(line) == 0


def test_delay_line_releases_in_chronological_order():
    line = DelayLine(delay_s=0.05)
    for stamp in (1.0, 1.01, 1.02):
        line.push(stamp, stamp)

    assert [payload for _, payload in line.pop_ready(1.10)] == [1.0, 1.01, 1.02]


def test_zero_delay_releases_immediately():
    line = DelayLine(delay_s=0.0)
    line.push(2.0, "now")

    assert line.pop_ready(2.0) == [(2.0, "now")]


def test_negative_delay_is_rejected():
    with pytest.raises(ValueError):
        DelayLine(delay_s=-0.01)
