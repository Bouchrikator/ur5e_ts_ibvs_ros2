"""Throwaway diagnostic: conditioning of the identification, in scale-free form.

The state mixes metres with metres per second, so a raw SVD of it is
unit-dependent and says nothing. Every quantity here is standardised first.
"""

import numpy as np

from cable_ts_control.modal_basis import ModalBasis
from cable_ts_control.state_filter import velocity_series
from cable_ts_control.ts_model import rule_memberships

data = np.load("cable_dataset.npz")
basis = ModalBasis.load("cable_modal_basis.yaml")
dt = float(data["timestep_s"][0])
sel = data["vertex_ids"] == 0

q = np.array([basis.project(s) for s in data["shapes"][sel]])
qd = velocity_series(q, dt, 0.4)
g = data["gripper"][sel] - data["gripper"].mean(axis=0)
u = data["commands"][sel]
x = np.hstack([q, qd, g])

names = ["q1", "q2", "qd1", "qd2", "g1", "g2"]
print("column rms:", {n: f"{v:.4g}" for n, v in zip(names, x.std(axis=0))})

z = (x - x.mean(0)) / x.std(0)
s = np.linalg.svd(z, compute_uv=False)
print("standardised state singular values:", np.array2string(s, precision=3))
print("condition number:", f"{s[0] / s[-1]:.1f}")

print("\nvariance of each state component explained by the other five:")
for k, name in enumerate(names):
    other = np.delete(z, k, axis=1)
    reg = np.hstack([other, np.ones((len(other), 1))])
    theta, *_ = np.linalg.lstsq(reg, z[:, k], rcond=None)
    r2 = 1 - np.mean((z[:, k] - reg @ theta) ** 2)
    print(f"  {name:4s}: R2 = {r2:.4f}")

bounds = [(-0.4683, 0.3643), (-0.2368, 0.1022)]
h = np.array([rule_memberships(row[:2], bounds) for row in x])
feat = np.hstack([x, u])
feat = (feat - feat.mean(0)) / feat.std(0)
reg = (h[:, :, None] * feat[:, None, :]).reshape(len(x), -1)
sv = np.linalg.svd(reg, compute_uv=False)
print(f"\nfuzzy regressor {reg.shape}: sigma_max/sigma_min = {sv[0] / sv[-1]:.4g}")
print("smallest 6 singular values:", np.array2string(sv[-6:], precision=4))
print(f"numerical rank (tol 1e-8 * smax): "
      f"{int((sv > 1e-8 * sv[0]).sum())} of {reg.shape[1]}")
