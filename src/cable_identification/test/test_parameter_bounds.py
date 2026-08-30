"""Tests for the shared parameter uncertainty envelope loader."""

import pytest
import yaml

from cable_identification.parameter_bounds import bounds_for, load_parameter_bounds

VALID = {"parameter_bounds": {
    "EI": {"min": 0.005, "max": 0.015, "units": "N.m^2"},
    "rayleigh_stiffness": {"min": 0.010, "max": 0.040, "units": "s"},
}}


def write(tmp_path, payload):
    path = tmp_path / "bounds.yaml"
    path.write_text(yaml.safe_dump(payload))
    return str(path)


def test_loads_every_declared_parameter(tmp_path):
    bounds = load_parameter_bounds(write(tmp_path, VALID))
    assert bounds == {"EI": (0.005, 0.015),
                      "rayleigh_stiffness": (0.010, 0.040)}


def test_bounds_for_preserves_the_requested_order(tmp_path):
    path = write(tmp_path, VALID)
    assert bounds_for(path, ["rayleigh_stiffness", "EI"]) == [
        (0.010, 0.040), (0.005, 0.015)]


def test_rejects_an_undeclared_parameter(tmp_path):
    with pytest.raises(ValueError, match="GJ"):
        bounds_for(write(tmp_path, VALID), ["EI", "GJ"])


def test_rejects_inverted_bounds(tmp_path):
    payload = {"parameter_bounds": {"EI": {"min": 0.015, "max": 0.005}}}
    with pytest.raises(ValueError, match="0 < min < max"):
        load_parameter_bounds(write(tmp_path, payload))


def test_rejects_non_positive_bounds(tmp_path):
    """Log-space identification cannot represent a non-positive stiffness."""
    payload = {"parameter_bounds": {"EI": {"min": 0.0, "max": 0.015}}}
    with pytest.raises(ValueError, match="0 < min < max"):
        load_parameter_bounds(write(tmp_path, payload))


def test_rejects_an_empty_envelope(tmp_path):
    with pytest.raises(ValueError, match="no parameter bounds"):
        load_parameter_bounds(write(tmp_path, {"parameter_bounds": {}}))


def test_the_shipped_envelope_contains_truth_and_estimator_start():
    """The envelope must not put the experiment on a rail before it starts."""
    from ament_index_python.packages import get_package_share_directory
    import os

    path = os.path.join(get_package_share_directory("cable_ts_control"),
                        "config", "cable_parameter_bounds.yaml")
    low, high = load_parameter_bounds(path)["EI"]
    assert low < 0.006 < high, "estimator start EI must be strictly inside"
    assert low < 0.010 < high, "truth EI must be strictly inside"
