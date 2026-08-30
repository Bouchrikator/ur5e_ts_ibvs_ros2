"""Throwaway diagnostic: where is the common-Lyapunov obstruction now?"""

import itertools

import numpy as np
import yaml

from cable_ts_control.lmi_synthesis import (
    solve_cable_ts_pdc, verify_certificate)
from cable_ts_control.ts_model import CableTsModel

path = "/tmp/cm3.yaml"
raw = yaml.safe_load(open(path))
model = CableTsModel.from_dict(raw)
n, m = model.state_dim, model.input_dim
pb = raw["cable_ts_model"]["parameter_bounds"]
sets = [([np.array(a).reshape(n, n) for a in A],
         [np.array(b).reshape(n, m) for b in B])
        for A, B in zip(pb["A"], pb["B"])]

print(f"{path}: {model.n_rules} rules, state {n}, bounds "
      f"{[tuple(round(v, 4) for v in b) for b in model.premise_bounds]}")
for v, (A, B) in enumerate(sets):
    da = max(np.linalg.norm(A[i] - A[j]) for i, j in itertools.product(range(len(A)), repeat=2))
    db = max(np.linalg.norm(B[i] - B[j]) for i, j in itertools.product(range(len(B)), repeat=2))
    print(f"  vtx{v}: ||dA||/||A|| = {da / np.linalg.norm(A[0]):.3f}, "
          f"||dB||/||B|| = {db / np.linalg.norm(B[0]):.3f}")


def ok(vs):
    # NOTE: the earlier version of this diagnostic verified relaxed solves
    # with the BASIC residual G'PG - P, which wrongly rejects valid relaxed
    # certificates. Every minimal-infeasible-subset result produced by that
    # version is unsound and must be regenerated.
    cert = solve_cable_ts_pdc(vs, eps=1e-6, relaxed=True)
    return bool(cert.feasible and verify_certificate(vs, cert)["satisfied"])


print("\nsingle parameter vertex, all rules + cross terms:")
for v in range(len(sets)):
    print(f"  vertex {v}: {ok([sets[v]])}")
print("single rule, all parameter vertices:")
for r in range(model.n_rules):
    print(f"  rule {r}: {ok([([A[r]], [B[r]]) for A, B in sets])}")
print("rule pairs inside vertex 0:")
for i, j in itertools.combinations(range(model.n_rules), 2):
    A, B = sets[0]
    print(f"  rules {i}+{j}: {ok([([A[i], A[j]], [B[i], B[j]])])}")
