"""Throwaway diagnostic: is the recorded command aligned with its transition?

The model integrates g[k+1] = g[k] + dt * u[k]. The dataset must therefore
store, at index k, the velocity applied BETWEEN sample k and sample k+1.
"""

import numpy as np

data = np.load("cable_dataset.npz")
dt = float(data["timestep_s"][0])
sel = (data["vertex_ids"] == 0) & (data["trajectory_ids"] == 0)
g = data["gripper"][sel]
u = data["commands"][sel]

step = np.diff(g, axis=0) / dt          # velocity that took g[k] to g[k+1]
aligned = np.abs(step - u[:-1]).max()   # u[k] drives k -> k+1  (what the model assumes)
shifted = np.abs(step - u[1:]).max()    # u[k+1] drives k -> k+1

print(f"max |(g[k+1]-g[k])/dt - u[k]|   = {aligned:.6e}   <- model's assumption")
print(f"max |(g[k+1]-g[k])/dt - u[k+1]| = {shifted:.6e}   <- one-sample shift")
print(f"command rms = {np.sqrt(np.mean(u ** 2)):.6f} m/s")
