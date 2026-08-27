"""System test for the offline discrete LMI synthesis (ROC convention: pytest).

Runs the actual gain-synthesis pipeline on the nominal camera/target model and
checks feasibility plus the closed-loop discrete Lyapunov conditions.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from solve_ts_pdc_lmi_discrete import (  # noqa: E402
    build_stacked_interaction_matrix,
    compute_vertex_matrices,
    solve_ts_pdc_lmi_discrete,
)

# Nominal setup used on the robot (cube 5 cm at Z* = 0.4 m).
HALF = (0.05 / 2.0) / 0.4
S_STAR = np.array([[-HALF, -HALF], [HALF, -HALF], [HALF, HALF], [-HALF, HALF]])
Z0, Z_MIN, Z_MAX, TS = 0.5, 0.2, 1.0, 0.033


def test_interaction_matrix_shape_and_depth_column():
    L = build_stacked_interaction_matrix(S_STAR, Z0)
    assert L.shape == (8, 6)
    # vx column must be -1/Z for every point.
    assert np.allclose(L[::2, 0], -1.0 / Z0)


def test_lmi_synthesis_feasible_and_stable():
    _, _, B1, B2 = compute_vertex_matrices(S_STAR, Z0, Z_MIN, Z_MAX)
    F1, F2, P, feasible = solve_ts_pdc_lmi_discrete(B1, B2, TS, verbose=False)
    assert feasible, "Discrete LMI synthesis must be feasible for the nominal model"

    # P must be symmetric positive definite.
    assert np.allclose(P, P.T, atol=1e-9)
    assert np.min(np.linalg.eigvalsh(P)) > 0

    # Closed-loop discrete Lyapunov decrease at both vertices and cross term.
    identity = np.eye(6)
    for G in (
        identity - TS * B1 @ F1,
        identity - TS * B2 @ F2,
        identity - TS * 0.5 * (B1 @ F2 + B2 @ F1),
    ):
        Q = G.T @ P @ G - P
        assert np.max(np.linalg.eigvalsh(0.5 * (Q + Q.T))) < 0


def test_memberships_reconstruct_interaction_matrix():
    # The TS model is exact in 1/Z: h_far*L(Zmax) + h_close*L(Zmin) == L(Z).
    Z = 0.55
    a, a_min, a_max = 1.0 / Z, 1.0 / Z_MAX, 1.0 / Z_MIN
    h_far = (a_max - a) / (a_max - a_min)
    h_close = 1.0 - h_far
    L_blend = h_far * build_stacked_interaction_matrix(S_STAR, Z_MAX) + \
        h_close * build_stacked_interaction_matrix(S_STAR, Z_MIN)
    # Only the depth-dependent columns (0-2) vary with 1/Z; rotation columns
    # are depth-independent, so the whole blend matches exactly.
    assert np.allclose(L_blend, build_stacked_interaction_matrix(S_STAR, Z))
