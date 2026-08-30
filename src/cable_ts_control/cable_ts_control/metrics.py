"""Experiment metrics for the cable shaping validation runs (plan step 12).

Pure functions over plain numbers and dictionaries: no ROS types, so the
arithmetic that ends up in the results table can be unit-tested rather than
eyeballed in a CSV afterwards.
"""

import math

__all__ = [
    "marker_errors",
    "saturation_ratio",
    "relative_error",
    "latency_s",
]


def marker_errors(reference, actual):
    """Euclidean error between two marker sets matched by id.

    ``reference`` and ``actual`` map marker id to an ``(x, y, z)`` tuple. Only
    ids present in both are compared, because a dropped marker is a gap in the
    evidence, not an error of size zero.

    Returns ``(rmse, max_error, matched)`` in metres.
    """
    shared = sorted(set(reference) & set(actual))
    if not shared:
        return float("nan"), float("nan"), 0

    squared = []
    worst = 0.0
    for marker_id in shared:
        a, b = reference[marker_id], actual[marker_id]
        distance = math.dist(a, b)
        squared.append(distance * distance)
        worst = max(worst, distance)
    return math.sqrt(sum(squared) / len(squared)), worst, len(shared)


def saturation_ratio(values, limit):
    """Largest component as a fraction of the limit; >= 1 means saturated."""
    if limit <= 0.0:
        raise ValueError("limit must be positive")
    peak = max((abs(float(v)) for v in values), default=0.0)
    return peak / limit


def relative_error(estimate, truth):
    """Signed relative error of an estimate against a known truth value."""
    if truth == 0.0:
        raise ValueError("truth must be non-zero for a relative error")
    return (float(estimate) - float(truth)) / float(truth)


def latency_s(source_stamp_s, sink_stamp_s):
    """Age of the evidence a command was built from; negative means clock skew."""
    return float(sink_stamp_s) - float(source_stamp_s)
