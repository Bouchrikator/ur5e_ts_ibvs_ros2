"""Throwaway diagnostic: is the filtered velocity consistent with the model?

The structured fit imposes the exact kinematic row q' = qdot. If the causal
low-pass lags, (q[k+1] - q[k])/dt differs from qdot[k] systematically, the
kinematic row is violated by the data, and the fit distorts K, D, H and G to
compensate - differently in every rule.
"""

import numpy as np

from cable_ts_control.modal_basis import ModalBasis
from cable_ts_control.state_filter import velocity_series

data = np.load("cable_dataset.npz")
basis = ModalBasis.load("/tmp/cb3.yaml")
dt = float(data["timestep_s"][0])
ref = basis.boundary_reference
sel = (data["vertex_ids"] == 0) & (data["trajectory_ids"] == 0)
boundary = data["gripper"][sel] - ref
q = np.array([basis.project(s, boundary=b)
              for s, b in zip(data["shapes"][sel], boundary)])

exact = np.diff(q, axis=0) / dt
for alpha in (0.2, 0.4, 0.7, 1.0):
    v = velocity_series(q, dt, alpha)
    err = exact - v[:-1]
    print(f"alpha {alpha:.1f}: rms|(dq/dt) - qdot| = {np.sqrt(np.mean(err ** 2)):.5f}, "
          f"rms qdot = {np.sqrt(np.mean(v ** 2)):.5f}, "
          f"relative = {np.sqrt(np.mean(err ** 2)) / np.sqrt(np.mean(v ** 2)):.3f}")
