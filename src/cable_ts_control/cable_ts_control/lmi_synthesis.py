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
cross terms ``(G_ij + G_ji)/2`` for ``i < j``. The relaxed conditions add a
slack ``Y >= 0``; in ``P`` space it appears as ``Q = P Y P`` and the solved
inequalities become

    G_ii' P G_ii - beta P + (s - 1) Q < 0
    H_ij' P H_ij - beta P - Q        <= 0

so a relaxed certificate is NOT verifiable with the basic inequalities: an
averaged cross term may have spectral radius above one and still be covered,
because Q compensates it through the (s-1)Q surplus on the diagonal terms.
The verification must therefore receive the complete certificate
(K_i, P, Q, beta, s, type) and check exactly what was solved.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class PdcCertificate:
    """Complete, independently checkable output of the PDC synthesis.

    Carries everything ``verify_certificate`` needs; nothing about the
    certificate is implicit in the solver call any more.
    """

    feasible: bool
    certificate_type: str                      # "basic" | "relaxed"
    gains: Optional[List[np.ndarray]] = None   # K_j
    lyapunov: Optional[np.ndarray] = None      # P = X^-1
    slack: Optional[np.ndarray] = None         # Q = P Y P (relaxed only)
    beta: float = 1.0                          # decay rate of the conditions
    max_active_rules: Optional[int] = None     # s (relaxed only)
    solver_name: Optional[str] = None
    solver_status: Optional[str] = None


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

    Returns a :class:`PdcCertificate`. For a relaxed solve the slack is
    returned as ``Q = P Y P`` so the certificate can be re-verified without
    the solver.
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
    s = None
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

    used_solver = None
    for solver, options in _solver_preferences(cp):
        try:
            problem.solve(solver=solver, verbose=verbose, **options)
        except Exception:
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            used_solver = str(solver)
            break

    certificate_type = "relaxed" if relaxed else "basic"
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return PdcCertificate(
            feasible=False, certificate_type=certificate_type, beta=decay,
            max_active_rules=s, solver_name=used_solver,
            solver_status=str(problem.status))

    x_value = 0.5 * (x.value + x.value.T)
    x_inv = np.linalg.inv(x_value)
    p = 0.5 * (x_inv + x_inv.T)
    q = None
    if relaxed:
        # The slack lives in X space; the P-space certificate is Q = P Y P.
        y = 0.5 * (slack.value + slack.value.T)
        q = p @ y @ p
        q = 0.5 * (q + q.T)
    return PdcCertificate(
        feasible=True, certificate_type=certificate_type,
        gains=[g.value @ x_inv for g in gains], lyapunov=p, slack=q,
        beta=decay, max_active_rules=s, solver_name=used_solver,
        solver_status=str(problem.status))


class _SparseSdp:
    """Conic problem assembled directly in sparse form (Clarabel or SCS).

    cvxpy canonicalises the 2n x 2n blocks through dense intermediates and
    Clarabel's interior point keeps a dense Hessian per PSD cone; at n = 34
    (16 strain modes, 40 cones of size 68) both exceed the 7 GB host. SCS only
    needs the sparse constraint matrix and one 68 x 68 eigendecomposition per
    cone and iteration. The maps below are the same inequalities as
    ``solve_cable_ts_pdc`` written column by column. Variables: upper triangle
    of X (and Y), the M_j entries, one SOC bound per M_j.
    """

    def __init__(self, n, m, n_rules, relaxed):
        self.n, self.m, self.n_rules = n, m, n_rules
        self.tri = [(i, j) for j in range(n) for i in range(j + 1)]
        self.x_index = {pair: k for k, pair in enumerate(self.tri)}
        offset = len(self.tri)
        self.y_offset = offset if relaxed else None
        if relaxed:
            offset += len(self.tri)
        self.m_offset = offset
        offset += n_rules * m * n
        self.t_offset = offset
        self.size = offset + n_rules
        self.psd_blocks = []

    def x_var(self, a, b):
        return self.x_index[(min(a, b), max(a, b))]

    def m_var(self, rule, d, b):
        return self.m_offset + (rule * self.m + d) * self.n + b

    def add_psd(self, entries, size):
        """``entries``: {(row, col) upper: [(coefficient, var or None)]}; svec(F) in the PSD cone."""
        self.psd_blocks.append((size, entries))

    def block(self, a_i, b_i, a_j, b_j, rule_i, rule_j, left_x_scale, left_y_scale, eps):
        """Upper triangle of [[left, G'], [G, X]] - eps I with G the (averaged) closed loop."""
        n = self.n
        entries = {}
        for (r, c) in self.tri:
            top = [(left_x_scale, self.x_var(r, c))]
            if left_y_scale and self.y_offset is not None:
                top.append((left_y_scale, self.y_offset + self.x_index[(r, c)]))
            if r == c:
                top.append((-eps, None))
            entries[(r, c)] = top
            bottom = [(1.0, self.x_var(r, c))]
            if r == c:
                bottom.append((-eps, None))
            entries[(n + r, n + c)] = bottom
        pairs = [(a_i, b_i, rule_j, 0.5 if a_j is not None else 1.0)]
        if a_j is not None:
            pairs.append((a_j, b_j, rule_i, 0.5))
        for a in range(n):
            for b in range(n):
                terms = []
                for a_mat, b_mat, rule, weight in pairs:
                    # G[b, a] = sum_c A[b, c] X[c, a] - sum_d B[b, d] M_rule[d, a]  (entry F[a, n + b] = G[b, a])
                    for c in range(n):
                        if a_mat[b, c] != 0.0:
                            terms.append((weight * a_mat[b, c], self.x_var(c, a)))
                    for d in range(self.m):
                        if b_mat[b, d] != 0.0:
                            terms.append((-weight * b_mat[b, d], self.m_var(rule, d, a)))
                entries[(a, n + b)] = terms
        self.add_psd(entries, 2 * n)

    def assemble(self, gain_penalty, lower_triangular, soc_first):
        """Sparse ``A x + s = b``; PSD blocks vectorised upper-column-major (Clarabel)
        or lower-column-major (SCS), off-diagonals scaled by sqrt(2)."""
        import scipy.sparse as sp
        rows, cols, vals, rhs = [], [], [], []
        soc_dim = self.m * self.n + 1
        psd_rows = sum(size * (size + 1) // 2 for size, _ in self.psd_blocks)
        soc_start = 0 if soc_first else psd_rows
        psd_start = self.n_rules * soc_dim if soc_first else 0
        for rule in range(self.n_rules):
            base = soc_start + rule * soc_dim
            rows.append(base)
            cols.append(self.t_offset + rule)
            vals.append(-1.0)
            for d in range(self.m):
                for b in range(self.n):
                    rows.append(base + 1 + d * self.n + b)
                    cols.append(self.m_var(rule, d, b))
                    vals.append(-1.0)
        offset = psd_start
        for size, entries in self.psd_blocks:
            for (row, col), terms in entries.items():
                scale = 1.0 if row == col else np.sqrt(2.0)
                if lower_triangular:
                    index = row * size - row * (row - 1) // 2 + (col - row)
                else:
                    index = col * (col + 1) // 2 + row
                for coefficient, var in terms:
                    if var is None:
                        rhs.append((offset + index, scale * coefficient))
                    else:
                        rows.append(offset + index)
                        cols.append(var)
                        vals.append(-scale * coefficient)
            offset += size * (size + 1) // 2
        n_rows = psd_rows + self.n_rules * soc_dim
        b = np.zeros(n_rows)
        for row, value in rhs:
            b[row] += value
        q = np.zeros(self.size)
        for i in range(self.n):
            q[self.x_index[(i, i)]] = 1.0
        q[self.t_offset:] = gain_penalty
        a_matrix = sp.csc_matrix((vals, (rows, cols)), shape=(n_rows, self.size))
        return a_matrix, b, q, [size for size, _ in self.psd_blocks], soc_dim

    def solve(self, gain_penalty, verbose, solver, eps, max_iters):
        if solver == "clarabel":
            import clarabel
            import scipy.sparse as sp
            a_matrix, b, q, psd_sizes, soc_dim = self.assemble(gain_penalty, False, False)
            cones = [clarabel.PSDTriangleConeT(size) for size in psd_sizes]
            cones += [clarabel.SecondOrderConeT(soc_dim)] * self.n_rules
            settings = clarabel.DefaultSettings()
            settings.verbose = verbose
            settings.max_iter = max_iters
            solution = clarabel.DefaultSolver(sp.csc_matrix((self.size, self.size)), q, a_matrix, b,
                                              cones, settings).solve()
            return str(solution.status), np.asarray(solution.x), str(solution.status) in ("Solved", "AlmostSolved")
        import scs
        a_matrix, b, q, psd_sizes, soc_dim = self.assemble(gain_penalty, True, True)
        solution = scs.SCS({"A": a_matrix, "b": b, "c": q}, {"q": [soc_dim] * self.n_rules, "s": psd_sizes},
                           eps_abs=eps, eps_rel=eps, max_iters=max_iters, verbose=verbose).solve()
        status = str(solution["info"]["status"])
        self.last_info = {key: float(solution["info"][key]) for key in ("res_pri", "res_dual", "gap", "pobj")}
        self.last_info["iterations"] = int(solution["info"]["iter"])
        # "solved (inaccurate ...)" is accepted here: verify_certificate is the gate.
        return status, np.asarray(solution["x"]), status.startswith("solved")

    def matrix(self, solution, base):
        out = np.zeros((self.n, self.n))
        for (i, j), k in self.x_index.items():
            out[i, j] = out[j, i] = solution[base + k]
        return out


def solve_cable_ts_pdc_sparse(vertex_sets, eps=1e-9, gain_penalty=1e-3, verbose=False,
                              relaxed=False, max_active_rules=None, decay=0.999, solver="scs",
                              solver_eps=1e-7, max_iters=100000):
    """Same problem as :func:`solve_cable_ts_pdc`, assembled sparsely.

    ``solver`` is ``"scs"`` (first order, memory-lean, the only one that fits the
    34-state modal problem on this host) or ``"clarabel"`` (interior point,
    small problems). Returns the same :class:`PdcCertificate`; acceptance still
    goes through :func:`verify_certificate`, which checks the inequalities
    independently of how they were assembled or how accurately they were solved.
    """
    vertex_sets = [(list(a), list(b)) for a, b in vertex_sets]
    if not vertex_sets:
        raise ValueError("at least one parameter vertex set is required")
    n_rules = len(vertex_sets[0][0])
    n, m = vertex_sets[0][0][0].shape[0], vertex_sets[0][1][0].shape[1]
    for a_vertices, b_vertices in vertex_sets:
        if len(a_vertices) != n_rules or len(b_vertices) != n_rules:
            raise ValueError("every parameter vertex set needs the same rule count")
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must satisfy 0 <= decay < 1, got {decay}")
    s = None
    if relaxed:
        s = n_rules if max_active_rules is None else int(max_active_rules)
        if s < 1:
            raise ValueError("max_active_rules must be >= 1")

    sdp = _SparseSdp(n, m, n_rules, relaxed)
    # X - I >= 0 and, relaxed, Y >= 0.
    sdp.add_psd({(i, j): [(1.0, sdp.x_index[(i, j)])] + ([(-1.0, None)] if i == j else [])
                 for (i, j) in sdp.tri}, n)
    if relaxed:
        sdp.add_psd({(i, j): [(1.0, sdp.y_offset + sdp.x_index[(i, j)])] for (i, j) in sdp.tri}, n)
    for a_vertices, b_vertices in vertex_sets:
        for i in range(n_rules):
            sdp.block(a_vertices[i], b_vertices[i], None, None, i, i, decay,
                      -(s - 1) if relaxed else 0.0, eps)
        for i in range(n_rules):
            for j in range(i + 1, n_rules):
                sdp.block(a_vertices[i], b_vertices[i], a_vertices[j], b_vertices[j], i, j, decay,
                          1.0 if relaxed else 0.0, 0.0)
    solution_status, values, solved = sdp.solve(gain_penalty, verbose, solver, solver_eps, max_iters)
    certificate_type = "relaxed" if relaxed else "basic"
    name = f"{solver.upper()}(sparse)"
    status = solution_status + (f" {getattr(sdp, 'last_info', '')}" if hasattr(sdp, "last_info") else "")
    if not solved:
        return PdcCertificate(feasible=False, certificate_type=certificate_type, beta=decay,
                              max_active_rules=s, solver_name=name, solver_status=status)
    x_value = sdp.matrix(values, 0)
    x_inv = np.linalg.inv(x_value)
    p = 0.5 * (x_inv + x_inv.T)
    gains = []
    for rule in range(n_rules):
        m_value = values[sdp.m_offset + rule * m * n: sdp.m_offset + (rule + 1) * m * n].reshape(m, n)
        gains.append(m_value @ x_inv)
    q = None
    if relaxed:
        y = sdp.matrix(values, sdp.y_offset)
        # A first-order solution may leave Y marginally indefinite (~1e-6); project
        # it back onto the cone. verify_certificate re-checks every inequality.
        eigenvalues, vectors = np.linalg.eigh(y)
        y = vectors @ np.diag(np.clip(eigenvalues, 0.0, None)) @ vectors.T
        q = p @ y @ p
        q = 0.5 * (q + q.T)
    return PdcCertificate(feasible=True, certificate_type=certificate_type, gains=gains, lyapunov=p,
                          slack=q, beta=decay, max_active_rules=s, solver_name=name,
                          solver_status=status)


def certificate_block_residuals(vertex_sets, certificate):
    """Largest eigenvalue of the certificate inequality per (vertex, rule pair), worst first.

    Diagnostic companion of :func:`verify_certificate`: names WHICH block keeps a
    non-certified solution from being accepted.
    """
    p = np.asarray(certificate.lyapunov, dtype=float)
    relaxed = certificate.certificate_type == "relaxed"
    q = np.asarray(certificate.slack, dtype=float) if relaxed else np.zeros_like(p)
    s = int(certificate.max_active_rules) if relaxed else 1
    blocks = []
    for vertex, (a_vertices, b_vertices) in enumerate(vertex_sets):
        for i in range(len(a_vertices)):
            for j in range(i, len(a_vertices)):
                if i == j:
                    g = a_vertices[i] - b_vertices[i] @ certificate.gains[i]
                    residual = g.T @ p @ g - p + (s - 1) * q
                else:
                    g = 0.5 * ((a_vertices[i] - b_vertices[i] @ certificate.gains[j])
                               + (a_vertices[j] - b_vertices[j] @ certificate.gains[i]))
                    residual = g.T @ p @ g - p - q
                blocks.append({"vertex": vertex, "rule_i": i, "rule_j": j, "residual": float(
                    np.max(np.linalg.eigvalsh(0.5 * (residual + residual.T))))})
    return sorted(blocks, key=lambda block: -block["residual"])


def verify_lyapunov_decrease(vertex_sets, gains, lyapunov):
    """Largest eigenvalue of ``G' P G - P`` over all rules and parameters.

    This is the BASIC-theorem residual (Y = 0). It is a valid acceptance test
    only for basic certificates: a relaxed certificate may fail this check
    on the averaged cross terms and still be valid, because the slack Q
    compensates them. Use :func:`verify_certificate` for the general case.
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


def verify_certificate(vertex_sets, certificate, cross_tolerance=1e-9):
    """Check the certificate against the inequalities it actually claims.

    Two levels are reported, both in ``P`` space and with ``Q = P Y P``
    (``Q = 0``, ``s = 1`` for basic certificates):

    * stability gate (``beta' = 1``) — what acceptance is decided on:

          G_ii' P G_ii - P + (s - 1) Q  < 0     (strict)
          H_ij' P H_ij - P - Q          <= 0

      With ``P > 0`` and ``Q >= 0`` these prove asymptotic stability of every
      blended closed loop; the solved decay ``beta < 1`` provides the
      numerical headroom that makes the gate robust to solver round-off.

    * decay residuals (``beta = certificate.beta``) — the exact inequalities
      the solver imposed, reported as ``*_at_beta`` for the archive. An
      interior-point solution satisfies them only up to its own tolerance, so
      they are informational, not the gate.

    Applying the BASIC inequalities to a relaxed certificate wrongly rejects
    valid solutions (scalar counterexample: P=1, Q=0.5, s=2, beta=0.999,
    G_ii=0, H_ij=1.1 satisfies the relaxed conditions while H'PH - P > 0).
    That mistake is pinned by a regression test.

    ``cross_tolerance`` absorbs round-off on the NON-strict cross inequality,
    relative to ``||P||_2``. Returns a dict with the residuals, the extremal
    eigenvalues of P and Q, and the boolean ``satisfied``.
    """
    if not certificate.feasible:
        raise ValueError("cannot verify an infeasible certificate")
    p = np.asarray(certificate.lyapunov, dtype=float)
    p = 0.5 * (p + p.T)
    beta = float(certificate.beta)
    relaxed = certificate.certificate_type == "relaxed"
    if relaxed:
        if certificate.slack is None or certificate.max_active_rules is None:
            raise ValueError("a relaxed certificate requires Q and s")
        q = np.asarray(certificate.slack, dtype=float)
        q = 0.5 * (q + q.T)
        s = int(certificate.max_active_rules)
    else:
        q = np.zeros_like(p)
        s = 1

    p_norm = float(np.linalg.norm(p, 2))
    worst = {"diag": -np.inf, "cross": -np.inf,
             "diag_beta": -np.inf, "cross_beta": -np.inf}
    for a_vertices, b_vertices in vertex_sets:
        n_rules = len(a_vertices)
        for i in range(n_rules):
            g_ii = a_vertices[i] - b_vertices[i] @ certificate.gains[i]
            base = g_ii.T @ p @ g_ii + (s - 1) * q
            for key, b_check in (("diag", 1.0), ("diag_beta", beta)):
                residual = base - b_check * p
                worst[key] = max(worst[key], float(np.max(np.linalg.eigvalsh(
                    0.5 * (residual + residual.T)))))
        for i in range(n_rules):
            for j in range(i + 1, n_rules):
                h_ij = 0.5 * (
                    (a_vertices[i] - b_vertices[i] @ certificate.gains[j])
                    + (a_vertices[j] - b_vertices[j] @ certificate.gains[i]))
                base = h_ij.T @ p @ h_ij - q
                for key, b_check in (("cross", 1.0), ("cross_beta", beta)):
                    residual = base - b_check * p
                    worst[key] = max(worst[key], float(np.max(
                        np.linalg.eigvalsh(0.5 * (residual + residual.T)))))

    p_min = float(np.min(np.linalg.eigvalsh(p)))
    q_min = float(np.min(np.linalg.eigvalsh(q))) if relaxed else None
    q_norm = float(np.linalg.norm(q, 2)) if relaxed else 0.0
    satisfied = (
        p_min > 0.0
        # Q >= 0 up to round-off: it enters the stability argument through
        # (s-1)*sum h_i^2 >= 2*sum h_i h_j, so a tiny negative eigenvalue is
        # absorbed by the strict diagonal margin.
        and (q_min is None or q_min >= -1e-7 * (1.0 + q_norm))
        and worst["diag"] < 0.0
        and worst["cross"] <= cross_tolerance * p_norm)
    return {
        "satisfied": bool(satisfied),
        "worst_diagonal_residual": worst["diag"],
        "worst_cross_residual": worst["cross"],
        "worst_diagonal_residual_at_beta": worst["diag_beta"],
        "worst_cross_residual_at_beta": worst["cross_beta"],
        "worst_normalized_residual":
            max(worst["diag"], worst["cross"]) / p_norm,
        "p_min_eigenvalue": p_min,
        "q_min_eigenvalue": q_min,
    }


def worst_spectral_radius(vertex_sets, gains, diagonal_only=False):
    """Worst spectral radius over the matrices the certificate actually covers.

    Only the diagonal terms ``A_i - B_i K_i`` and the AVERAGED cross terms
    ``(G_ij + G_ji)/2`` are certified: ``h_i h_j == h_j h_i``, so the two
    unaveraged cross terms only ever appear summed. Testing each of them alone
    rejects perfectly valid PDC certificates.

    Under the RELAXED theorem even the averaged cross terms are not
    individually contractive — only the diagonal terms are (their condition
    implies ``G_ii' P G_ii < beta P``). Pass ``diagonal_only=True`` when
    gating a relaxed certificate; the cross terms are then covered solely by
    :func:`verify_certificate`.
    """
    worst = 0.0
    for a_vertices, b_vertices in vertex_sets:
        n_rules = len(a_vertices)
        closed_loop = [a_vertices[i] - b_vertices[i] @ gains[i]
                       for i in range(n_rules)]
        if not diagonal_only:
            for i in range(n_rules):
                for j in range(i + 1, n_rules):
                    closed_loop.append(0.5 * (
                        (a_vertices[i] - b_vertices[i] @ gains[j])
                        + (a_vertices[j] - b_vertices[j] @ gains[i])))
        for g in closed_loop:
            worst = max(worst, float(np.max(np.abs(np.linalg.eigvals(g)))))
    return worst
