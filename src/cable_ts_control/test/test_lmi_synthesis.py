"""Unit tests for the cable TS-PDC LMI synthesis.

These are the acceptance tests of the control design: the gains returned by the
solver must genuinely certify the discrete Lyapunov decrease for every rule AND
every physical-parameter vertex.
"""

import numpy as np
import pytest

from cable_ts_control.lmi_synthesis import (
    solve_cable_ts_pdc,
    verify_lyapunov_decrease,
    worst_spectral_radius,
)

cvxpy = pytest.importorskip("cvxpy", reason="cvxpy is required for LMI synthesis")


def make_vertices(stiffness_scale=1.0, damping=0.9, dt=1.0 / 30.0):
    """Four rule vertices of a 2-mode cable at one physical-parameter vertex.

    Double-integrator-like modal dynamics: position integrates velocity, the
    gripper velocity drives the modal velocity. Open loop is marginally stable,
    so the LMIs have real work to do.
    """
    a_vertices, b_vertices = [], []
    for rule in range(4):
        # Rule-dependent stiffness coupling = the geometric nonlinearity.
        coupling = 1.0 + 0.15 * rule
        a = np.array([[1.0, 0.0, dt, 0.0],
                      [0.0, 1.0, 0.0, dt],
                      [-0.02 * coupling * stiffness_scale, 0.0, damping, 0.0],
                      [0.0, -0.02 * coupling * stiffness_scale, 0.0, damping]])
        b = np.array([[0.0, 0.0],
                      [0.0, 0.0],
                      [0.6 * coupling, 0.0],
                      [0.0, 0.6 * coupling]])
        a_vertices.append(a)
        b_vertices.append(b)
    return a_vertices, b_vertices


def test_single_parameter_vertex_is_feasible_and_certified():
    vertex_sets = [make_vertices()]

    gains, lyapunov, feasible = solve_cable_ts_pdc(vertex_sets)

    assert feasible
    assert len(gains) == 4
    assert verify_lyapunov_decrease(vertex_sets, gains, lyapunov) < 0.0


def test_lyapunov_matrix_is_symmetric_positive_definite():
    _, lyapunov, feasible = solve_cable_ts_pdc([make_vertices()])

    assert feasible
    assert lyapunov == pytest.approx(lyapunov.T, abs=1e-6)
    assert np.min(np.linalg.eigvalsh(0.5 * (lyapunov + lyapunov.T))) > 0.0


def test_gains_have_the_right_shape():
    gains, _, feasible = solve_cable_ts_pdc([make_vertices()])

    assert feasible
    assert all(gain.shape == (2, 4) for gain in gains)


def test_closed_loop_is_inside_the_unit_circle():
    vertex_sets = [make_vertices()]
    gains, _, feasible = solve_cable_ts_pdc(vertex_sets)

    assert feasible
    assert worst_spectral_radius(vertex_sets, gains) < 1.0


def test_one_gain_set_stabilises_every_parameter_vertex():
    """EI and damping are uncertainties, so a single gain set must cover all."""
    vertex_sets = [
        make_vertices(stiffness_scale=0.6, damping=0.88),
        make_vertices(stiffness_scale=0.6, damping=0.96),
        make_vertices(stiffness_scale=1.4, damping=0.88),
        make_vertices(stiffness_scale=1.4, damping=0.96),
    ]

    gains, lyapunov, feasible = solve_cable_ts_pdc(vertex_sets)

    assert feasible
    assert verify_lyapunov_decrease(vertex_sets, gains, lyapunov) < 0.0
    assert worst_spectral_radius(vertex_sets, gains) < 1.0


def test_gains_from_one_vertex_are_verified_against_all_of_them():
    """Guards the robustness claim: verification must look at every vertex."""
    narrow = [make_vertices(stiffness_scale=1.0, damping=0.9)]
    gains, lyapunov, feasible = solve_cable_ts_pdc(narrow)
    assert feasible

    wide = narrow + [make_vertices(stiffness_scale=40.0, damping=0.999)]

    assert verify_lyapunov_decrease(narrow, gains, lyapunov) < 0.0
    assert verify_lyapunov_decrease(wide, gains, lyapunov) > \
        verify_lyapunov_decrease(narrow, gains, lyapunov)


def test_empty_vertex_set_is_rejected():
    with pytest.raises(ValueError):
        solve_cable_ts_pdc([])


def test_inconsistent_rule_counts_are_rejected():
    a_vertices, b_vertices = make_vertices()

    with pytest.raises(ValueError):
        solve_cable_ts_pdc([(a_vertices, b_vertices),
                            (a_vertices[:2], b_vertices[:2])])
