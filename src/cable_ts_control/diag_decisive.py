"""Throwaway diagnostic: is the LMI refusal structural or numerical?

Three questions, one run:
  A. PBH controllability at the near-unit eigenvalues, per (vertex, rule).
  B. Failing subsets: solver-infeasible or point-returned-then-rejected?
  C. Full problem under (decay, state scaling) — scaling is a similarity
     transform x~ = S x with S = diag(1,1,1,Ts,Ts,Ts,1,1) (velocities as
     per-step displacements), certificates mapped back and re-verified in
     ORIGINAL coordinates, so a flip here proves pure conditioning.
"""

import sys
from dataclasses import replace

import numpy as np
import yaml

from cable_ts_control.lmi_synthesis import (
    solve_cable_ts_pdc, verify_certificate)
from cable_ts_control.ts_model import CableTsModel

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/ros2_ws/artifacts/cable_ts_model_400x5_nofeed.yaml"
raw = yaml.safe_load(open(path))
model = CableTsModel.from_dict(raw)
n, m = model.state_dim, model.input_dim
pb = raw["cable_ts_model"]["parameter_bounds"]
sets = [([np.array(a).reshape(n, n) for a in A],
         [np.array(b).reshape(n, m) for b in B])
        for A, B in zip(pb["A"], pb["B"])]
Ts = model.sample_time
nm = model.n_modes

print(f"model {path}: state {n}, input {m}, {model.n_rules} rules, "
      f"{len(sets)} parameter vertices, Ts={Ts}")

print("\n== A. PBH margin sigma_min([A - lam I, B]) at |lam| >= 0.98 ==")
for v, (A, B) in enumerate(sets):
    for r in range(model.n_rules):
        margins = []
        for lam in np.linalg.eigvals(A[r]):
            if abs(lam) >= 0.98:
                pbh = np.hstack([A[r] - lam * np.eye(n), B[r]])
                margins.append((abs(lam), np.linalg.svd(pbh, compute_uv=False)[-1]))
        worst = min(margins, key=lambda t: t[1]) if margins else None
        print(f"  vtx{v} rule{r}: near-unit eigs {len(margins)}, "
              f"worst PBH {worst[1]:.3e} at |lam|={worst[0]:.4f}" if worst
              else f"  vtx{v} rule{r}: no near-unit eigenvalues")


def probe(vs, decay=0.999, scale=None, label=""):
    solve_sets = vs
    if scale is not None:
        s_inv = np.linalg.inv(scale)
        solve_sets = [([scale @ a @ s_inv for a in A], [scale @ b for b in B])
                      for A, B in vs]
    cert = solve_cable_ts_pdc(solve_sets, relaxed=True, decay=decay)
    if not cert.feasible:
        print(f"  {label}: solver-INFEASIBLE ({cert.solver_status})")
        return None
    if scale is not None:
        cert = replace(
            cert,
            gains=[k @ scale for k in cert.gains],
            lyapunov=scale @ cert.lyapunov @ scale,
            slack=None if cert.slack is None else scale @ cert.slack @ scale)
    rep = verify_certificate(vs, cert)
    print(f"  {label}: solver point ({cert.solver_name}, {cert.solver_status}) -> "
          f"{'CERTIFIED' if rep['satisfied'] else 'REJECTED'} "
          f"diag={rep['worst_diagonal_residual']:.3e} "
          f"cross={rep['worst_cross_residual']:.3e} "
          f"p_min={rep['p_min_eigenvalue']:.3e} q_min={rep['q_min_eigenvalue']:.3e}")
    return cert if rep["satisfied"] else None


print("\n== B. failing subsets: solver status vs verification ==")
probe([sets[0]], label="vtx0, all 4 rules")
probe([([A[2]], [B[2]]) for A, B in sets], label="rule 2 across vertices")
probe([([A[3]], [B[3]]) for A, B in sets], label="rule 3 across vertices")

print("\n== C. full problem: decay x scaling grid ==")
S = np.diag([1.0] * nm + [Ts] * nm + [1.0] * (n - 2 * nm))
winner = None
for decay in (0.999, 0.9999, 0.99999):
    for scale, tag in ((None, "plain"), (S, "scaled")):
        cert = probe(sets, decay=decay, scale=scale,
                     label=f"decay={decay} {tag}")
        if cert is not None and winner is None:
            winner = (decay, tag, cert)

print("\n== verdict ==")
if winner is None:
    print("no variant certifies: obstruction is STRUCTURAL for this model, "
          "not solver conditioning -> quasi-static/narrow-box route")
else:
    decay, tag, cert = winner
    print(f"FEASIBLE at decay={decay} ({tag}); certificate verified in "
          f"original coordinates. ||K||_max = "
          f"{max(np.linalg.norm(k, 2) for k in cert.gains):.3f}")
