import numpy as np, yaml
from cable_ts_control.ts_model import CableTsModel
from cable_ts_control.scripts.solve_cable_ts_lmi import load_vertex_sets
raw = yaml.safe_load(open("/ros2_ws/artifacts/cable_mor/cable_ts_model_modal_rom.yaml"))
model = CableTsModel.from_dict(raw); vs = load_vertex_sets(model, raw["cable_ts_model"])
n = model.state_dim
for v, (A, B) in enumerate(vs):
    for r, (a, b) in enumerate(zip(A, B)):
        eig = np.linalg.eigvals(a)
        unc = []
        for lam in eig:
            s = np.linalg.svd(np.hstack([a - lam * np.eye(n), b]), compute_uv=False)
            if s[-1] < 1e-9 * s[0]:
                unc.append(abs(lam))
        unc = np.array(sorted(unc))
        ctrb = np.hstack([np.linalg.matrix_power(a, k) @ b for k in range(n)])
        sv = np.linalg.svd(ctrb, compute_uv=False)
        print(f"v{v} r{r}: PBH-uncontrollable eigs {len(unc)} |lam| in [{unc.min():.6f},{unc.max():.6f}] "
              f"(all<1: {bool(unc.max() < 1)}), ctrb sv ratio sv[15]/sv[0]={sv[15]/sv[0]:.1e} sv[33]/sv[0]={sv[33]/sv[0]:.1e}")
