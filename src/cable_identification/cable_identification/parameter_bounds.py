"""Loader for the cable parameter uncertainty envelope.

Kept in one place because three consumers must agree on the same box: the SOFA
dataset generator (which vertices to excite), the LMI synthesis (which vertices
to certify) and the online estimator (where to clamp). A disagreement between
them would silently invalidate the stability certificate.
"""

import yaml

__all__ = ["load_parameter_bounds", "bounds_for"]


def load_parameter_bounds(path):
    """Read ``cable_parameter_bounds.yaml`` into ``{name: (min, max)}``."""
    with open(path, "r") as handle:
        data = yaml.safe_load(handle) or {}
    section = data.get("parameter_bounds", data)

    bounds = {}
    for name, entry in section.items():
        low, high = float(entry["min"]), float(entry["max"])
        if not 0.0 < low < high:
            raise ValueError(
                f"{name}: bounds must satisfy 0 < min < max, got ({low}, {high})")
        bounds[name] = (low, high)
    if not bounds:
        raise ValueError(f"{path} declares no parameter bounds")
    return bounds


def bounds_for(path, names):
    """Bounds for ``names``, in that order; raises if any is undeclared."""
    bounds = load_parameter_bounds(path)
    missing = [n for n in names if n not in bounds]
    if missing:
        raise ValueError(f"{path} lacks bounds for {missing}")
    return [bounds[n] for n in names]
