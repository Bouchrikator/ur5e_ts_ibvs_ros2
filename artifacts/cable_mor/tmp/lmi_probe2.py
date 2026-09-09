import time, numpy as np, yaml, scipy.sparse as sp, scs
from cable_ts_control.ts_model import CableTsModel
from cable_ts_control.scripts.solve_cable_ts_lmi import load_vertex_sets
from cable_ts_control.lmi_synthesis import _SparseSdp
raw = yaml.safe_load(open("/ros2_ws/artifacts/cable_mor/cable_ts_model_modal_rom.yaml"))
model = CableTsModel.from_dict(raw); vs = load_vertex_sets(model, raw["cable_ts_model"])
n, m, R = model.state_dim, model.input_dim, model.n_rules
sdp = _SparseSdp(n, m, R, False)
sdp.add_psd({(i, j): [(1.0, sdp.x_index[(i, j)])] + ([(-1.0, None)] if i == j else []) for (i, j) in sdp.tri}, n)
for a, b in vs:
    for i in range(R): sdp.block(a[i], b[i], None, None, i, i, 0.999, 0.0, 1e-6)
    for i in range(R):
        for j in range(i + 1, R): sdp.block(a[i], b[i], a[j], b[j], i, j, 0.999, 0.0, 0.0)
A, bvec, q, psd, soc = sdp.assemble(1e-3, True, True)
print("A", A.shape, "nnz", A.nnz, "psd cones", len(psd), flush=True)
t = time.perf_counter()
sol = scs.SCS({"A": A, "b": bvec, "c": q}, {"q": [soc] * R, "s": psd}, eps_abs=1e-5, eps_rel=1e-5, max_iters=2000, verbose=True).solve()
info = sol["info"]; print({k: info[k] for k in ("status", "iter", "res_pri", "res_dual", "gap", "pobj", "solve_time")}, f"{time.perf_counter()-t:.0f}s", flush=True)
