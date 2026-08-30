"""Throwaway prototype: does removing the boundary from the basis help?

The marker set includes the tip, which the gripper holds, so part of the modal
coordinate is algebraically slaved to the gripper position. A Craig-Bampton
style split writes

    y = y0 + Psi g + Phi q~

where Psi g is the quasi-static response to the boundary and q~ are the
fixed-interface modes, uncorrelated with g by construction. Compares the plain
PCA basis against that split on the same dataset.
"""

import numpy as np

from cable_ts_control.local_identification import (
    blended_rollout, fit_structured_fuzzy_model)
from cable_ts_control.state_filter import velocity_series
from cable_ts_control.ts_model import rule_memberships

data = np.load("cable_dataset.npz")
dt = float(data["timestep_s"][0])
shapes, gripper, commands = data["shapes"], data["gripper"], data["commands"]
vtx, traj = data["vertex_ids"], data["trajectory_ids"]
g_ref = gripper.mean(axis=0)
train = np.isin(traj, [0, 1, 2])
n_modes = 4


def plain_basis(y):
    mean = y.mean(axis=0)
    _, s, vt = np.linalg.svd(y - mean, full_matrices=False)
    return mean, None, vt[:n_modes].T, s


def boundary_basis(y, g):
    reg = np.hstack([g - g_ref, np.ones((len(g), 1))])
    theta, *_ = np.linalg.lstsq(reg, y, rcond=None)
    psi, mean = theta[:-1].T, theta[-1]
    residual = y - reg @ theta
    _, s, vt = np.linalg.svd(residual, full_matrices=False)
    return mean, psi, vt[:n_modes].T, s


def evaluate(name, mean, psi, phi, singular):
    boundary = gripper - g_ref
    base = np.broadcast_to(mean, shapes.shape).copy()
    if psi is not None:
        base = base + boundary @ psi.T
    q = (shapes - base) @ phi
    recon = base + q @ phi.T
    print(f"\n--- {name}: reconstruction rmse "
          f"{np.sqrt(np.mean((recon - shapes) ** 2)) * 1e3:.3f} mm, "
          f"energy {np.sum(singular[:n_modes] ** 2) / np.sum(singular ** 2) * 100:.2f} %")
    print(f"    corr(q, g) max = "
          f"{np.abs(np.corrcoef(np.hstack([q, boundary]).T)[:n_modes, n_modes:]).max():.4f}")

    spreads, rollouts = [], []
    for v in range(4):
        sel = train & (vtx == v)
        x = np.hstack([q[sel], velocity_series(q[sel], dt, 0.4), boundary[sel]])
        u = commands[sel]
        bounds = [(np.percentile(x[:, i], 5), np.percentile(x[:, i], 95))
                  for i in range(2)]
        h = np.array([rule_memberships(r[:2], bounds) for r in x])
        a, b = fit_structured_fuzzy_model(
            x[:-1], u[:-1], x[1:], h[:-1], dt, n_modes, ridge=1e-8)
        spreads.append((
            max(np.linalg.norm(a[i] - a[j]) for i in range(4) for j in range(4))
            / np.linalg.norm(a[0]),
            max(np.linalg.norm(b[i] - b[j]) for i in range(4) for j in range(4))
            / np.linalg.norm(b[0])))

        sv = train & (vtx == v) & (traj == 2)
        xv = np.hstack([q[sv], velocity_series(q[sv], dt, 0.4), boundary[sv]])
        rolled = blended_rollout(a, b, bounds, xv[0], commands[sv][:50])
        target = base[sv][1:51] + rolled[:, :n_modes] @ phi.T
        rollouts.append(np.sqrt(np.mean((target - shapes[sv][1:51]) ** 2)))
    for v, ((da, db), r) in enumerate(zip(spreads, rollouts)):
        print(f"    vtx{v}: ||dA||/||A|| = {da:.3f}, ||dB||/||B|| = {db:.3f}, "
              f"rollout {r * 1e3:.1f} mm")


evaluate("plain PCA", *plain_basis(shapes))
evaluate("boundary split", *boundary_basis(shapes, gripper))
