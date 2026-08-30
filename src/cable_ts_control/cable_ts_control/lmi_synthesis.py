"""Discrete TS-PDC synthesis for the reduced cable model.

Same mathematical structure as the IBVS discrete solver, but the cable model
has its own ``A_i`` (the IBVS reduced error model has ``A = I``), and the gains
must hold for EVERY physical-parameter vertex at once — EI and damping are
bounded uncertainties, not premises.

Closed loop with the PDC law ``u = -sum_j h_j K_j e``:

    e(k+1) = sum_i sum_j h_i h_j ( A_i^p - B_i^p K_j ) e(k) = G_ij^p e(k)

Quadratic stability with ``V(e) = e' P e``, ``P = X^-1`` and ``K_j = M_j X^-1``.
Applying the Schur complement to ``G' P G - P < 0`` gives, for every parameter
vertex ``p``:

    [ X                         (A_i^p X - B_i^p M_j)' ]
    [ (A_i^p X - B_i^p M_j)      X                     ] > 0

Relaxation (Tanaka & Wang): the diagonal terms ``i = j`` plus the averaged
cross terms ``(G_ij + G_ji)/2`` for ``i < j``.
"""

import numpy as np


def _cross_averaged(a_i, a_j, b_i, b_j, x, m_i, m_j):
    """``((A_i X - B_i M_j) + (A_j X - B_j M_i)) / 2``."""
    return 0.5 * ((a_i @ x - b_i @ m_j) + (a_j @ x - b_j @ m_i))


def _solver_preferences(cp):
    """Solvers to try, most accurate first, with tight tolerances.

    The Lyapunov margin is only a few 1e-4, so a first-order solver at its
    default accuracy can return a point that fails verification. Interior-point
    solvers are tried first and SCS is only a fallback.
    """
    candidates = [
        (cp.MOSEK, {}),
        (cp.CLARABEL, {"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10,
                       "tol_feas": 1e-10}),
        (cp.CVXOPT, {"abstol": 1e-10, "reltol": 1e-10}),
        (cp.SCS, {"eps": 1e-10, "max_iters": 200000}),
    ]
    installed = set(cp.installed_solvers())
    return [(solver, options) for solver, options in candidates
            if solver in installed]


def solve_cable_ts_pdc(vertex_sets, eps=1e-9, gain_penalty=1e-3, verbose=False,
                       relaxed=False, max_active_rules=None, decay=0.999):
    """Synthesise PDC gains valid for every parameter vertex.

    Parameters
    ----------
    vertex_sets
        Iterable of ``(a_vertices, b_vertices)`` pairs — one entry per physical
        parameter combination (EI min/max, damping min/max, ...). Every entry
        must use the same rule ordering.
    eps
        Small absolute floor keeping the blocks strictly definite for the
        solver. It is NOT the stability margin — see ``decay``.
    decay
        ``beta = alpha^2`` of the discrete decay-rate conditions (Tanaka & Wang
        3.36-3.37, relaxed 3.43-3.44), giving ``dV <= (beta - 1) V``. This is
        the real margin and it is RELATIVE, so it survives the change of
        variables. An absolute floor on the block in ``X`` space does not: the
        verification happens in ``P = X^-1`` space, where the margin comes back
        as ``P (margin) P`` and vanishes once ``X`` grows, which is how the
        solver can report success on gains that then fail verification.
        Must satisfy ``0 <= decay < 1``.
    gain_penalty
        Weight of the gain-norm regularisation; larger means softer gains.
    relaxed
        Use the Tanaka & Wang relaxed discrete conditions (DFS 3.27-3.28),
        which introduce a slack ``Y >= 0`` and only require the diagonal terms
        to dominate by ``(s - 1) Y``. The basic conditions are the special case
        ``Y = 0``, so this can only enlarge the feasible set.
    max_active_rules
        ``s``, the largest number of rules that can fire at once. With
        complementary triangular memberships over ``n`` premises every product
        rule is active in the interior, so the default is the rule count.

    Returns ``(gains, lyapunov, feasible)``.
    """
    import cvxpy as cp

    vertex_sets = [(list(a), list(b)) for a, b in vertex_sets]
    if not vertex_sets:
        raise ValueError("at least one parameter vertex set is required")

    n_rules = len(vertex_sets[0][0])
    n = vertex_sets[0][0][0].shape[0]
    m = vertex_sets[0][1][0].shape[1]
    for a_vertices, b_vertices in vertex_sets:
        if len(a_vertices) != n_rules or len(b_vertices) != n_rules:
            raise ValueError("every parameter vertex set needs the same rule count")

    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must satisfy 0 <= decay < 1, got {decay}")

    x = cp.Variable((n, n), symmetric=True)
    gains = [cp.Variable((m, n)) for _ in range(n_rules)]

    # The constraints are homogeneous in (X, M) and the recovered gains
    # K = M X^-1 do not depend on that scale, so X is normalised to X >= I.
    # Without it the objective drives X onto its own lower bound and the
    # recovered P = X^-1 becomes numerically meaningless.
    constraints = [x >> np.eye(n)]
    identity = np.eye(2 * n)

    slack = None
    if relaxed:
        s = n_rules if max_active_rules is None else int(max_active_rules)
        if s < 1:
            raise ValueError("max_active_rules must be >= 1")
        slack = cp.Variable((n, n), symmetric=True)
        constraints.append(slack >> 0)
        diagonal_left = decay * x - (s - 1) * slack
        cross_left = decay * x + slack
    else:
        diagonal_left = decay * x
        cross_left = decay * x

    for a_vertices, b_vertices in vertex_sets:
        for i in range(n_rules):
            g_ii = a_vertices[i] @ x - b_vertices[i] @ gains[i]
            constraints.append(
                cp.bmat([[diagonal_left, g_ii.T], [g_ii, x]]) >> eps * identity)

        for i in range(n_rules):
            for j in range(i + 1, n_rules):
                g_cross = _cross_averaged(
                    a_vertices[i], a_vertices[j], b_vertices[i], b_vertices[j],
                    x, gains[i], gains[j])
                constraints.append(
                    cp.bmat([[cross_left, g_cross.T], [g_cross, x]]) >> 0)

    objective = cp.Minimize(
        cp.trace(x) + gain_penalty * sum(cp.norm(g, "fro") for g in gains))
    problem = cp.Problem(objective, constraints)

    for solver, options in _solver_preferences(cp):
        try:
            problem.solve(solver=solver, verbose=verbose, **options)
        except Exception:
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            break

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return None, None, False

    x_value = 0.5 * (x.value + x.value.T)
    x_inv = np.linalg.inv(x_value)
    return [g.value @ x_inv for g in gains], 0.5 * (x_inv + x_inv.T), True


def verify_lyapunov_decrease(vertex_sets, gains, lyapunov):
    """Largest eigenvalue of ``G' P G - P`` over all rules and parameters.

    Strictly negative means the quadratic Lyapunov certificate really holds
    for the returned gains — the acceptance test of the synthesis.
    """
    lyapunov = np.asarray(lyapunov, dtype=float)
    worst = -np.inf
    for a_vertices, b_vertices in vertex_sets:
        n_rules = len(a_vertices)
        closed_loop = []
        for i in range(n_rules):
            closed_loop.append(a_vertices[i] - b_vertices[i] @ gains[i])
        for i in range(n_rules):
            for j in range(i + 1, n_rules):
                closed_loop.append(0.5 * (
                    (a_vertices[i] - b_vertices[i] @ gains[j])
                    + (a_vertices[j] - b_vertices[j] @ gains[i])))
        for g in closed_loop:
            residual = g.T @ lyapunov @ g - lyapunov
            worst = max(worst, float(np.max(np.linalg.eigvalsh(
                0.5 * (residual + residual.T)))))
    return worst


def worst_spectral_radius(vertex_sets, gains):
    """Worst spectral radius over the matrices the certificate actually covers.

    Only the diagonal terms ``A_i - B_i K_i`` and the AVERAGED cross terms
    ``(G_ij + G_ji)/2`` are certified: ``h_i h_j == h_j h_i``, so the two
    unaveraged cross terms only ever appear summed. Testing each of them alone
    rejects perfectly valid PDC certificates.
    """
    worst = 0.0
    for a_vertices, b_vertices in vertex_sets:
        n_rules = len(a_vertices)
        closed_loop = [a_vertices[i] - b_vertices[i] @ gains[i]
                       for i in range(n_rules)]
        for i in range(n_rules):
            for j in range(i + 1, n_rules):
                closed_loop.append(0.5 * (
                    (a_vertices[i] - b_vertices[i] @ gains[j])
                    + (a_vertices[j] - b_vertices[j] @ gains[i])))
        for g in closed_loop:
            worst = max(worst, float(np.max(np.abs(np.linalg.eigvals(g)))))
    return worst
