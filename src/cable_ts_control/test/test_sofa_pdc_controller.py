"""Checks of the SOFA-side pieces that need no running scene: contract of the
training controller's causal convention and the PDC controller's time convention,
using a stub cable/coupling. Real SOFA gates are cable_sofa_* commands."""

import numpy as np
import pytest

Sofa = pytest.importorskip("Sofa.Core", reason="SofaPython3 needed for controller classes")

from cable_ts_control.sofa_pdc_controller import CablePdcSofaController  # noqa: E402
from cable_ts_control.ts_model import CableTsModel  # noqa: E402


class _Modal:
    def __init__(self, values):
        self.position = type("D", (), {"value": np.asarray(values, dtype=float).reshape(-1, 1)})()


class _Cable:
    cfg = {"timestep_s": 0.01}

    def __init__(self):
        self.modal_mo = _Modal([0.1, -0.2])


class _Coupling:
    def __init__(self):
        self.poses = []

    def update_grasp(self, pose, now_s):
        self.poses.append((now_s, np.asarray(pose[:2])))


def _model():
    a = np.eye(6)
    a[0, 2] = a[1, 3] = 0.04
    b = np.zeros((6, 2))
    b[4:, :] = np.eye(2)
    gains = [np.zeros((2, 6)) for _ in range(4)]
    for k in gains:
        k[:, 4:] = 0.5 * np.eye(2)  # pull the gripper block to its reference
    from cable_ts_control.premises import BoundaryPremises
    model = CableTsModel([a] * 4, [b] * 4, [(0.0, 0.1), (-0.5, 0.5)], 0.04, gains=gains,
                         lyapunov=np.eye(6), n_modes=2,
                         premise_map=BoundaryPremises(4, 0.7, (0.6, 0.0)))
    return model


def test_pdc_command_at_k_drives_the_gripper_over_k_to_k_plus_1():
    cable, coupling = _Cable(), _Coupling()
    controller = CablePdcSofaController(cable, coupling, _model(), np.zeros(6), 0.04, 0.4,
                                        gripper_reference=(0.6, 0.0), gripper_start=(0.66, 0.02),
                                        max_linear_vel=0.05)
    for _ in range(8 * 4):
        controller.onAnimateBeginEvent(None)
    log = controller.arrays()
    # u[k] = (g[k+1] - g[k]) / dt exactly, the dataset convention checked by check_input_alignment.
    np.testing.assert_allclose(np.diff(log["gripper"], axis=0) / 0.04, log["command"][:-1], atol=1e-12)
    assert np.all(np.linalg.norm(log["command"], axis=1) <= 0.05 + 1e-12)
    assert log["error_norm"][-1] < log["error_norm"][0]
    assert len(coupling.poses) == 32 and log["state"].shape == (8, 6)
