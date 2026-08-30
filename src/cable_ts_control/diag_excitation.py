"""Throwaway diagnostic: is the excitation persistently exciting the dynamics?

Compares the frequency content of the commanded gripper velocity against the
natural frequencies of the identified cable modes. A quasi-static sweep leaves
the cable at its static equilibrium for the current gripper position, so the
data only ever shows the equilibrium manifold and the dynamic directions of the
state are never probed.
"""

import numpy as np
import yaml

from cable_ts_control.ts_model import CableTsModel

data = np.load("cable_dataset.npz")
dt = float(data["timestep_s"][0])
commands = data["commands"]
traj = data["trajectory_ids"]
vtx = data["vertex_ids"]

raw = yaml.safe_load(open("/tmp/model_m2.yaml"))
model = CableTsModel.from_dict(raw)
n = model.state_dim
pb = raw["cable_ts_model"]["parameter_bounds"]

print("identified cable modes (continuous equivalent):")
for v, group in enumerate(pb["A"]):
    for i, a in enumerate(group):
        lam = np.linalg.eigvals(np.array(a).reshape(n, n))
        s = np.log(lam.astype(complex)) / dt
        keep = np.abs(s.imag) > 1e-6
        if keep.any():
            f = np.sort(np.abs(s.imag[keep]) / (2 * np.pi))
            z = -s.real[keep] / np.abs(s[keep])
            print(f"  vtx{v} rule{i}: f = {np.array2string(f, precision=3)} Hz, "
                  f"zeta = {np.array2string(np.sort(z), precision=3)}")
        break
    if v == 0:
        pass

seg = commands[(vtx == 0) & (traj == 0)]
n_s = len(seg)
freq = np.fft.rfftfreq(n_s, dt)
power = np.abs(np.fft.rfft(seg - seg.mean(axis=0), axis=0)) ** 2
power = power.sum(axis=1)
power /= power.sum()

print(f"\ncommanded gripper velocity, {n_s} samples at {1 / dt:.1f} Hz")
cumulative = np.cumsum(power)
for f_cut in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
    idx = np.searchsorted(freq, f_cut)
    print(f"  energy below {f_cut:4.2f} Hz: {cumulative[min(idx, len(cumulative) - 1)] * 100:6.2f} %")
print(f"  median frequency: {freq[np.searchsorted(cumulative, 0.5)]:.3f} Hz")
print(f"  95th percentile:  {freq[np.searchsorted(cumulative, 0.95)]:.3f} Hz")
