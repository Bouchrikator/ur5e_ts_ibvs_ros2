"""Modal (PCA) reduction of the cable shape.

Step 6 of the plan: the raw marker vector is too large and too correlated to be
a Takagi-Sugeno state. A handful of SOFA shapes already spans almost all of the
observed deformation, so the shape is described by

    y ~= y_mean + Phi @ q

with ``Phi`` orthonormal and ``q`` the modal amplitudes used as the reduced
state. Pure numerics, no ROS and no SOFA: the basis can be built offline,
unit tested, and reused by both the reducer node and the identification
scripts.
"""

import numpy as np
import yaml


def build_modal_basis(shapes, n_modes):
    """Principal directions of a set of cable shapes.

    Parameters
    ----------
    shapes
        (N, D) array: one flattened shape vector per row.
    n_modes
        Number of modes to keep.

    Returns ``(mean, Phi, singular_values)`` with ``Phi`` of shape
    ``(D, n_modes)`` and orthonormal columns.
    """
    shapes = np.asarray(shapes, dtype=float)
    if shapes.ndim != 2:
        raise ValueError(f"expected a 2-D (N, D) array, got {shapes.shape}")
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    if n_modes > min(shapes.shape):
        raise ValueError(
            f"cannot extract {n_modes} modes from a {shapes.shape} dataset")

    mean = shapes.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(shapes - mean, full_matrices=False)
    return mean, vt[:n_modes].T, singular_values


def explained_variance_ratio(singular_values, n_modes):
    """Fraction of the dataset variance captured by the first ``n_modes``."""
    energy = np.asarray(singular_values, dtype=float) ** 2
    total = energy.sum()
    if total == 0.0:
        return 0.0
    return float(energy[:n_modes].sum() / total)


class ModalBasis:
    """Projection between the marker space and the reduced modal space."""

    def __init__(self, mean, phi, marker_s_over_l=None, singular_values=None):
        self.mean = np.asarray(mean, dtype=float)
        self.phi = np.asarray(phi, dtype=float)
        if self.phi.ndim != 2:
            raise ValueError("phi must be 2-D")
        if self.phi.shape[0] != self.mean.shape[0]:
            raise ValueError(
                f"phi has {self.phi.shape[0]} rows but the mean has "
                f"{self.mean.shape[0]} entries")
        self.marker_s_over_l = list(marker_s_over_l or [])
        self.singular_values = (None if singular_values is None
                                else np.asarray(singular_values, dtype=float))

    @property
    def n_modes(self):
        return self.phi.shape[1]

    @property
    def dimension(self):
        return self.phi.shape[0]

    def project(self, y, valid=None):
        """Modal amplitudes of a shape vector.

        ``valid`` optionally masks entries of ``y`` that were not observed; the
        amplitudes are then recovered by least squares on the remaining rows,
        which is what keeps the loop alive through marker dropout.
        """
        y = np.asarray(y, dtype=float)
        if y.shape != self.mean.shape:
            raise ValueError(
                f"expected a shape vector of size {self.mean.shape[0]}, "
                f"got {y.shape[0]}")

        residual = y - self.mean
        if valid is None:
            return self.phi.T @ residual

        valid = np.asarray(valid, dtype=bool)
        if valid.shape != y.shape:
            raise ValueError("valid must have the same shape as y")
        if valid.sum() < self.n_modes:
            raise ValueError(
                f"need at least {self.n_modes} observed entries to identify "
                f"{self.n_modes} modes, got {int(valid.sum())}")
        q, *_ = np.linalg.lstsq(self.phi[valid], residual[valid], rcond=None)
        return q

    def reconstruct(self, q):
        """Shape vector of a set of modal amplitudes."""
        q = np.asarray(q, dtype=float)
        if q.shape != (self.n_modes,):
            raise ValueError(f"expected {self.n_modes} amplitudes, got {q.shape}")
        return self.mean + self.phi @ q

    def reconstruction_rmse(self, y, valid=None):
        """Per-entry RMS residual left by the reduced description [m]."""
        y = np.asarray(y, dtype=float)
        residual = y - self.reconstruct(self.project(y, valid))
        if valid is not None:
            residual = residual[np.asarray(valid, dtype=bool)]
        if residual.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(residual ** 2)))

    # -- persistence ---------------------------------------------------
    def to_dict(self):
        data = {
            "n_modes": int(self.n_modes),
            "dimension": int(self.dimension),
            "mean": self.mean.tolist(),
            "phi": self.phi.flatten(order="C").tolist(),
            "marker_s_over_l": [float(s) for s in self.marker_s_over_l],
        }
        if self.singular_values is not None:
            data["singular_values"] = self.singular_values.tolist()
        return {"cable_modal_basis": data}

    def save(self, path):
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, width=200)

    @classmethod
    def from_dict(cls, data):
        section = data["cable_modal_basis"]
        dimension = int(section["dimension"])
        n_modes = int(section["n_modes"])
        phi = np.asarray(section["phi"], dtype=float).reshape(
            (dimension, n_modes), order="C")
        return cls(section["mean"], phi,
                   section.get("marker_s_over_l"),
                   section.get("singular_values"))

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))
