import time, numpy as np, yaml, sys
from cable_ts_control.ts_model import CableTsModel
from cable_ts_control.scripts.solve_cable_ts_lmi import load_vertex_sets
from cable_ts_control.lmi_synthesis import solve_cable_ts_pdc_sparse, verify_certificate, worst_spectral_radius
raw = yaml.safe_load(open("/ros2_ws/artifacts/cable_mor/cable_ts_model_modal_rom.yaml"))
model = CableTsModel.from_dict(raw); vs = load_vertex_sets(model, raw["cable_ts_model"])
print("state", model.state_dim, "rules", model.n_rules, "vertices", len(vs), flush=True)
for a, b in vs:
    print(" rho(A) per rule", [round(float(np.max(np.abs(np.linalg.eigvals(x)))), 5) for x in a], flush=True)
t = time.perf_counter()
c = solve_cable_ts_pdc_sparse(vs, eps=1e-6, solver="scs", solver_eps=1e-5, verbose=True)
print("basic", c.solver_status, c.feasible, f"{time.perf_counter()-t:.0f}s", flush=True)
if c.feasible:
    r = verify_certificate(vs, c); print(r, "rho", worst_spectral_radius(vs, c.gains), flush=True)
