"""h_ROM(a): the ROM graph itself (ModelOrderReductionMapping -> DiscreteCosseratMapping).

Kinematic only: the scene is initialised once and never stepped, so this is not a
second physics cable. It is the observation model of the visual observer and the
marker-space metric of the modal TS identification.
"""

import numpy as np
import Sofa.Core
import Sofa.Simulation

from cable_identification import cosserat_model as cm
from cable_identification.strain_basis import validate_reduction


class ModalObservationModel:
    def __init__(self, cfg, reduction):
        _, self.modes, self.metadata = validate_reduction(cfg, reduction)
        self.root = Sofa.Core.Node("modal_observation_model")
        cm.prepare_root(self.root, cfg, extra_plugins=["ModelOrderReduction"])
        self.cable = cm.build_cable(self.root, cfg, reduction=reduction)
        Sofa.Simulation.init(self.root)
        self.n_modes = self.modes.shape[1]
        self.n_markers = len(self.cable.marker_indices)

    def markers(self, modal):
        """Planar marker vector (2 * n_markers,) of the shape with coordinates ``a``."""
        self._apply(modal)
        return np.asarray(self.cable.marker_positions())[:, :2].ravel()

    def frames(self, modal):
        """Flattened centerline frame positions (3 * n_frames,) at ``a``."""
        self._apply(modal)
        return np.asarray(self.cable.frame_poses())[:, :3].ravel()

    def _apply(self, modal):
        with self.cable.modal_mo.position.writeable() as position:
            position[:] = np.asarray(modal, dtype=float).reshape(-1, 1)
        self.cable.refresh_mapping()

    def project(self, strain):
        """Diagnostic oracle ``a = Phi^T kappa`` when the FOM strain is available."""
        return self.modes.T @ np.asarray(strain, dtype=float).ravel()

    def close(self):
        Sofa.Simulation.unload(self.root)
