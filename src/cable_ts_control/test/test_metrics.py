"""Tests for the validation metrics (plan step 12)."""

import math

import pytest

from cable_ts_control.metrics import (
    latency_s,
    marker_errors,
    relative_error,
    saturation_ratio,
)


def test_identical_marker_sets_have_zero_error():
    markers = {0: (0.0, 0.0, 0.0), 1: (0.1, 0.0, 0.0)}
    rmse, worst, matched = marker_errors(markers, markers)
    assert rmse == 0.0
    assert worst == 0.0
    assert matched == 2


def test_reports_rmse_and_the_worst_marker_separately():
    reference = {0: (0.0, 0.0, 0.0), 1: (0.0, 0.0, 0.0)}
    actual = {0: (0.003, 0.0, 0.0), 1: (0.005, 0.0, 0.0)}
    rmse, worst, _ = marker_errors(reference, actual)
    assert worst == pytest.approx(0.005)
    assert rmse == pytest.approx(math.sqrt((0.003 ** 2 + 0.005 ** 2) / 2))
    assert rmse < worst


def test_dropped_markers_are_excluded_not_scored_as_zero():
    """A missing marker is absent evidence; counting it as 0 flatters the run."""
    reference = {0: (0.0, 0.0, 0.0), 1: (0.0, 0.0, 0.0)}
    actual = {0: (0.004, 0.0, 0.0)}
    rmse, worst, matched = marker_errors(reference, actual)
    assert matched == 1
    assert rmse == pytest.approx(0.004)
    assert worst == pytest.approx(0.004)


def test_no_shared_markers_is_not_a_number():
    rmse, worst, matched = marker_errors({0: (0.0, 0.0, 0.0)}, {5: (0.0, 0.0, 0.0)})
    assert matched == 0
    assert math.isnan(rmse) and math.isnan(worst)


def test_saturation_is_one_at_the_limit():
    assert saturation_ratio((0.05, 0.0, 0.0), 0.05) == pytest.approx(1.0)


def test_saturation_uses_the_largest_component():
    assert saturation_ratio((0.01, -0.04, 0.0), 0.05) == pytest.approx(0.8)


def test_saturation_flags_commands_beyond_the_limit():
    assert saturation_ratio((0.075, 0.0, 0.0), 0.05) > 1.0


def test_saturation_rejects_a_non_positive_limit():
    with pytest.raises(ValueError):
        saturation_ratio((0.0,), 0.0)


def test_relative_error_is_signed():
    assert relative_error(0.006, 0.010) == pytest.approx(-0.4)
    assert relative_error(0.014, 0.010) == pytest.approx(0.4)


def test_relative_error_rejects_zero_truth():
    with pytest.raises(ValueError):
        relative_error(1.0, 0.0)


def test_latency_is_the_age_of_the_evidence():
    assert latency_s(10.0, 10.033) == pytest.approx(0.033)
