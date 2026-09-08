"""PDC closed loop executed INSIDE the SOFA scene.

``u_k = -sum_i h_i(rho_k) K_i (x_k - x*)`` is computed at sample ``k`` from the
ROM's own modal state (or an injected estimate) and drives the grasp target
over ``[k, k+1)`` through ``GraspCoupling`` -- the same input channel and the
same causal convention as the identification dataset (``u[k]`` moves ``k -> k+1``).
"""

import numpy as np
import Sofa.Core

from cable_ts_control.modal_contract import build_modal_state
from cable_ts_control.state_filter import ModalVelocityFilter


class CablePdcSofaController(Sofa.Core.Controller):
    def __init__(self, cable, coupling, model, target, control_dt, velocity_alpha,
                 gripper_reference, gripper_start, max_linear_vel, state_source=None, **kwargs):
        super().__init__(**kwargs)
        self.cable, self.coupling, self.model = cable, coupling, model
        self.target = np.asarray(target, dtype=float)
        self.control_dt = float(control_dt)
        self.sofa_dt = float(cable.cfg["timestep_s"])
        self.substeps = round(self.control_dt / self.sofa_dt)
        if self.substeps < 1 or abs(self.substeps * self.sofa_dt - self.control_dt) > 1e-12:
            raise ValueError("Control period must be an exact multiple of the SOFA timestep")
        self.filter = ModalVelocityFilter(model.n_modes, velocity_alpha)
        self.gripper_reference = np.asarray(gripper_reference, dtype=float)
        self.gripper = np.asarray(gripper_start, dtype=float).copy()
        self.max_linear_vel = float(max_linear_vel)
        # Default state source: the ROM's independent coordinates, read directly. An
        # injected source receives the controller (last state/command for its prior).
        self.state_source = state_source or (lambda _: self.cable.modal_mo.position.value.ravel())
        self.command = np.zeros(2)
        self.step = 0
        self.log = {key: [] for key in ("time", "state", "command", "gripper", "lyapunov",
                                        "error_norm", "weights")}

    def onAnimateBeginEvent(self, event):
        if self.step % self.substeps == 0:
            a = np.asarray(self.state_source(self), dtype=float)
            a_dot = self.filter.update(a, self.control_dt)
            x = build_modal_state(a, a_dot, self.gripper, self.gripper_reference)
            u, weights = self.model.control(x, self.target)
            norm = float(np.linalg.norm(u))
            if norm > self.max_linear_vel:
                u = u * (self.max_linear_vel / norm)
            self.command = u
            self.log["time"].append(self.step * self.sofa_dt)
            self.log["state"].append(x)
            self.log["command"].append(u.copy())
            self.log["gripper"].append(self.gripper.copy())
            self.log["lyapunov"].append(self.model.lyapunov_value(x, self.target))
            self.log["error_norm"].append(float(np.linalg.norm(x - self.target)))
            self.log["weights"].append(weights)
        self.gripper = self.gripper + self.command * self.sofa_dt
        self.coupling.update_grasp([*self.gripper, 0.0, 0.0, 0.0, 0.0, 1.0],
                                   1.0 + (self.step + 1) * self.sofa_dt)
        self.step += 1

    def arrays(self):
        return {key: np.asarray(values) for key, values in self.log.items()}
