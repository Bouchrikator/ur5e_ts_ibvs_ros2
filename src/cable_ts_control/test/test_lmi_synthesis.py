"""Unit tests for the cable TS-PDC LMI synthesis.

These are the acceptance tests of the control design: the gains returned by the
solver must genuinely certify the discrete Lyapunov decrease for every rule AND
every physical-parameter vertex.
"""

import numpy as np
import pytest

from cable_ts_control.lmi_synthesis import (
    PdcCertificate,
    solve_cable_ts_pdc,
    verify_certificate,
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

    cert = solve_cable_ts_pdc(vertex_sets)

    assert cert.feasible
    assert cert.certificate_type == "basic"
    assert len(cert.gains) == 4
    assert verify_lyapunov_decrease(vertex_sets, cert.gains, cert.lyapunov) < 0.0
    assert verify_certificate(vertex_sets, cert)["satisfied"]


def test_lyapunov_matrix_is_symmetric_positive_definite():
    cert = solve_cable_ts_pdc([make_vertices()])

    assert cert.feasible
    lyapunov = cert.lyapunov
    assert lyapunov == pytest.approx(lyapunov.T, abs=1e-6)
    assert np.min(np.linalg.eigvalsh(0.5 * (lyapunov + lyapunov.T))) > 0.0


def test_gains_have_the_right_shape():
    cert = solve_cable_ts_pdc([make_vertices()])

    assert cert.feasible
    assert all(gain.shape == (2, 4) for gain in cert.gains)


def test_closed_loop_is_inside_the_unit_circle():
    vertex_sets = [make_vertices()]
    cert = solve_cable_ts_pdc(vertex_sets)

    assert cert.feasible
    assert worst_spectral_radius(vertex_sets, cert.gains) < 1.0


def test_one_gain_set_stabilises_every_parameter_vertex():
    """EI and damping are uncertainties, so a single gain set must cover all."""
    vertex_sets = [
        make_vertices(stiffness_scale=0.6, damping=0.88),
        make_vertices(stiffness_scale=0.6, damping=0.96),
        make_vertices(stiffness_scale=1.4, damping=0.88),
        make_vertices(stiffness_scale=1.4, damping=0.96),
    ]

    cert = solve_cable_ts_pdc(vertex_sets)

    assert cert.feasible
    assert verify_lyapunov_decrease(vertex_sets, cert.gains, cert.lyapunov) < 0.0
    assert worst_spectral_radius(vertex_sets, cert.gains) < 1.0


def test_gains_from_one_vertex_are_verified_against_all_of_them():
    """Guards the robustness claim: verification must look at every vertex."""
    narrow = [make_vertices(stiffness_scale=1.0, damping=0.9)]
    cert = solve_cable_ts_pdc(narrow)
    assert cert.feasible
    gains, lyapunov = cert.gains, cert.lyapunov

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


# -- what PDC actually certifies -----------------------------------------


def test_open_loop_unstable_but_stabilisable_vertices_are_accepted():
    """PDC certifies A_i - B_i K_j, not A_i.

    Rejecting a model because some open-loop rho(A_i) > 1 throws away
    perfectly stabilisable plants. The scalar case A=1.2, B=1, K=0.5 gives a
    closed loop of 0.7.
    """
    a_vertices = [np.array([[1.2]]) for _ in range(2)]
    b_vertices = [np.array([[1.0]]) for _ in range(2)]

    assert all(abs(np.linalg.eigvals(a)[0]) > 1.0 for a in a_vertices)

    cert = solve_cable_ts_pdc([(a_vertices, b_vertices)])
    assert cert.feasible
    assert verify_lyapunov_decrease([(a_vertices, b_vertices)], cert.gains,
                                    cert.lyapunov) < 0.0
    assert worst_spectral_radius([(a_vertices, b_vertices)], cert.gains) < 1.0


def test_spectral_check_covers_only_the_certified_matrices():
    """h_i h_j == h_j h_i, so only the AVERAGED cross terms are certified.

    Testing each unaveraged A_i - B_i K_j alone rejects valid certificates.
    """
    vertex_sets = [make_vertices()]
    cert = solve_cable_ts_pdc(vertex_sets)
    assert cert.feasible
    gains, lyapunov = cert.gains, cert.lyapunov

    a_vertices, b_vertices = vertex_sets[0]
    unaveraged = max(
        float(np.max(np.abs(np.linalg.eigvals(a - b @ k))))
        for a, b in zip(a_vertices, b_vertices) for k in gains)

    assert worst_spectral_radius(vertex_sets, gains) <= unaveraged + 1e-9
    assert verify_lyapunov_decrease(vertex_sets, gains, lyapunov) < 0.0


# -- relaxed Tanaka-Wang conditions --------------------------------------


def test_relaxed_conditions_certify_when_they_report_feasible():
    vertex_sets = [make_vertices(), make_vertices(stiffness_scale=1.4)]
    cert = solve_cable_ts_pdc(vertex_sets, relaxed=True)

    if not cert.feasible:
        pytest.skip("relaxed problem infeasible for this vertex set")
    # The certificate must be complete: Q = P Y P, beta and s all returned.
    assert cert.certificate_type == "relaxed"
    assert cert.slack is not None
    assert cert.max_active_rules == 4
    assert 0.0 <= cert.beta < 1.0
    assert np.min(np.linalg.eigvalsh(
        0.5 * (cert.slack + cert.slack.T))) >= -1e-7
    # Verified with the RELAXED residuals — not the basic ones.
    assert verify_certificate(vertex_sets, cert)["satisfied"]
    # Only the diagonal terms are individually contractive under the relaxed
    # theorem; the averaged cross terms are covered by Q, not by this gate.
    assert worst_spectral_radius(
        vertex_sets, cert.gains, diagonal_only=True) < 1.0


def test_relaxed_is_no_more_restrictive_than_basic():
    """Y = 0 is admissible in the relaxed problem, so it cannot lose."""
    vertex_sets = [make_vertices()]
    basic = solve_cable_ts_pdc(vertex_sets).feasible
    relaxed = solve_cable_ts_pdc(vertex_sets, relaxed=True).feasible

    assert relaxed or not basic


def test_relaxed_rejects_a_nonsensical_active_rule_count():
    with pytest.raises(ValueError):
        solve_cable_ts_pdc([make_vertices()], relaxed=True, max_active_rules=0)


# -- the margin must survive the change of variables ---------------------


@pytest.mark.parametrize("relaxed", [False, True])
def test_feasible_always_means_strictly_certified(relaxed):
    """Whatever the solver reports, the Lyapunov decrease must be negative.

    The synthesis works in ``X = P^-1`` but is verified in ``P``. An absolute
    margin on the block in X space comes back as ``P (margin) P``, which
    vanishes as X grows: the solver then reports success on gains whose
    certificate fails. The decay-rate form is relative and does survive.
    """
    vertex_sets = [make_vertices(), make_vertices(stiffness_scale=1.3)]
    cert = solve_cable_ts_pdc(vertex_sets, relaxed=relaxed)

    if not cert.feasible:
        pytest.skip("infeasible for this vertex set")
    assert verify_certificate(vertex_sets, cert)["satisfied"]
    if not relaxed:
        assert verify_lyapunov_decrease(
            vertex_sets, cert.gains, cert.lyapunov) < 0.0


def test_decay_bounds_the_lyapunov_decrease():
    """``decay`` is the real margin: dV <= (decay - 1) V."""
    vertex_sets = [make_vertices()]
    cert = solve_cable_ts_pdc(vertex_sets, decay=0.9)
    assert cert.feasible

    worst = verify_lyapunov_decrease(vertex_sets, cert.gains, cert.lyapunov)
    # A tenth of the smallest Lyapunov eigenvalue is a conservative bound on
    # the guaranteed decrease; the certificate must be at least that negative.
    assert worst <= -0.1 * float(np.min(np.linalg.eigvalsh(cert.lyapunov))) * 0.5


def test_relaxed_verifier_accepts_the_scalar_counterexample():
    """Regression: relaxed certificates must be checked with Q, not without.

    P=1, Q=0.5, s=2, beta=0.999, G_11=G_22=0, H_12=1.1 satisfies the relaxed
    Tanaka-Wang conditions (residuals -0.499 and -0.289), yet the BASIC gates
    reject it: H'PH - P = 0.21 > 0 and rho(H) = 1.1 > 1. The earlier verifier
    applied exactly those basic gates to relaxed solutions and therefore threw
    away valid certificates.
    """
    a_vertices = [np.array([[0.6]]), np.array([[0.5]])]
    b_vertices = [np.array([[1.0]]), np.array([[-1.0]])]
    gains = [np.array([[0.6]]), np.array([[-0.5]])]
    vertex_sets = [(a_vertices, b_vertices)]
    # G_11 = 0.6 - 1.0*0.6 = 0, G_22 = 0.5 - (-1)(-0.5) = 0,
    # H_12 = ((0.6 + 0.5) + (0.5 + 0.6)) / 2 = 1.1.
    cert = PdcCertificate(
        feasible=True, certificate_type="relaxed", gains=gains,
        lyapunov=np.array([[1.0]]), slack=np.array([[0.5]]),
        beta=0.999, max_active_rules=2)

    report = verify_certificate(vertex_sets, cert)

    assert report["satisfied"]
    # stability gate (beta' = 1): 0 - 1 + 0.5 and 1.21 - 1 - 0.5
    assert report["worst_diagonal_residual"] == pytest.approx(-0.5, abs=1e-9)
    assert report["worst_cross_residual"] == pytest.approx(-0.29, abs=1e-9)
    # solved decay residuals (beta = 0.999): the audit's numbers
    assert report["worst_diagonal_residual_at_beta"] == \
        pytest.approx(-0.499, abs=1e-9)
    assert report["worst_cross_residual_at_beta"] == \
        pytest.approx(-0.289, abs=1e-9)
    # The basic gates wrongly reject this exact point:
    assert verify_lyapunov_decrease(vertex_sets, gains, cert.lyapunov) > 0.0
    assert worst_spectral_radius(vertex_sets, gains) > 1.0
    # ... while the diagonal-only gate (all the relaxed theorem promises
    # about individual matrices) passes:
    assert worst_spectral_radius(vertex_sets, gains, diagonal_only=True) < 1.0


def test_verify_certificate_rejects_a_violated_relaxed_condition():
    """With too little slack the same point must be rejected."""
    a_vertices = [np.array([[0.6]]), np.array([[0.5]])]
    b_vertices = [np.array([[1.0]]), np.array([[-1.0]])]
    gains = [np.array([[0.6]]), np.array([[-0.5]])]
    cert = PdcCertificate(
        feasible=True, certificate_type="relaxed", gains=gains,
        lyapunov=np.array([[1.0]]), slack=np.array([[0.1]]),
        beta=0.999, max_active_rules=2)

    report = verify_certificate([(a_vertices, b_vertices)], cert)

    # cross residual at the stability gate: 1.21 - 1 - 0.1 = 0.11 > 0
    assert not report["satisfied"]
    assert report["worst_cross_residual"] == pytest.approx(0.11, abs=1e-9)


def test_verify_certificate_refuses_infeasible_certificates():
    cert = PdcCertificate(feasible=False, certificate_type="basic")
    with pytest.raises(ValueError):
        verify_certificate([make_vertices()], cert)


def test_rejects_an_out_of_range_decay():
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            solve_cable_ts_pdc([make_vertices()], decay=bad)
