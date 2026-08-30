"""Modal reduction of the cable shape with an explicit boundary block.

Step 6 of the plan: the raw marker vector is too large and too correlated to be
a Takagi-Sugeno state. The shape is therefore written as

    y ~= y0 + Psi b + Phi q

with ``b`` the gripper displacement (the prescribed boundary), ``Psi`` its
quasi-static response, ``Phi`` orthonormal and ``q`` the modal amplitudes used
as the reduced state.

The boundary block matters. The marker set includes the tip, which the gripper
holds, so a plain PCA of the shape produces modal coordinates that are
partly an algebraic function of the gripper position: measured correlation 0.76
against 0.00 with the split. The reduced state is then not minimal, part of the
modal acceleration is really the gripper acceleration, and the identification
has to absorb that into the input matrix, differently in every fuzzy rule.

Naming, precisely: this is a BOUNDARY-CONDITIONED POD, not Craig-Bampton.
The structure (boundary block + interior modes) mirrors the Craig-Bampton
partition, but ``Psi`` is a least-squares regression on data — not the static
constraint modes ``-K_ii^-1 K_ib`` — and ``Phi`` is a PCA of the residual, not
the fixed-interface eigenmodes of a mass/stiffness pair. Two consequences:
the zero correlation above is guaranteed on the FITTING data by least squares
(residuals are orthogonal to regressors), so it is evidence of a better basis,
not proof of dynamic or physical decoupling; and no mass-orthogonality or
boundary-coupling property of a true Craig-Bampton reduction may be assumed.

``Psi = None`` recovers the plain PCA of the original pipeline.

Pure numerics, no ROS and no SOFA: the basis can be built offline, unit tested,
and reused by both the reducer node and the identification scripts.
"""

import numpy as np
import yaml


def build_modal_basis(shapes, n_modes, boundary=None):
    """Principal directions of a set of cable shapes.

    Parameters
    ----------
    shapes
        (N, D) array: one flattened shape vector per row.
    n_modes
        Number of modes to keep.
    boundary
        Optional (N, P) array of the prescribed boundary displacement. When
        given, its quasi-static response is regressed out first and the modes
        describe only what the boundary does not explain.

    Returns ``(mean, Phi, singular_values, Psi)`` with ``Phi`` of shape
    ``(D, n_modes)`` and orthonormal columns, and ``Psi`` of shape ``(D, P)``
    or ``None``.
    """
    shapes = np.asarray(shapes, dtype=float)
    if shapes.ndim != 2:
        raise ValueError(f"expected a 2-D (N, D) array, got {shapes.shape}")
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    if n_modes > min(shapes.shape):
        raise ValueError(
            f"cannot extract {n_modes} modes from a {shapes.shape} dataset")

    if boundary is None:
        mean = shapes.mean(axis=0)
        residual, psi = shapes - mean, None
    else:
        boundary = np.asarray(boundary, dtype=float)
        if len(boundary) != len(shapes):
            raise ValueError("boundary must have one row per shape")
        regressor = np.hstack([boundary, np.ones((len(boundary), 1))])
        theta, *_ = np.linalg.lstsq(regressor, shapes, rcond=None)
        psi, mean = theta[:-1].T, theta[-1]
        residual = shapes - regressor @ theta

    _, singular_values, vt = np.linalg.svd(residual, full_matrices=False)
    return mean, vt[:n_modes].T, singular_values, psi


def explained_variance_ratio(singular_values, n_modes):
    """Fraction of the dataset variance captured by the first ``n_modes``."""
    energy = np.asarray(singular_values, dtype=float) ** 2
    total = energy.sum()
    if total == 0.0:
        return 0.0
    return float(energy[:n_modes].sum() / total)


class ModalBasis:
    """Projection between the marker space and the reduced modal space."""

    def __init__(self, mean, phi, marker_s_over_l=None, singular_values=None,
                 psi=None, boundary_reference=None):
        self.mean = np.asarray(mean, dtype=float)
        self.phi = np.asarray(phi, dtype=float)
        if self.phi.ndim != 2:
            raise ValueError("phi must be 2-D")
        if self.phi.shape[0] != self.mean.shape[0]:
            raise ValueError(
                f"phi has {self.phi.shape[0]} rows but the mean has "
                f"{self.mean.shape[0]} entries")
        self.psi = None if psi is None else np.asarray(psi, dtype=float)
        if self.psi is not None and self.psi.shape[0] != self.mean.shape[0]:
            raise ValueError(
                f"psi has {self.psi.shape[0]} rows but the mean has "
                f"{self.mean.shape[0]} entries")
        self.marker_s_over_l = list(marker_s_over_l or [])
        self.singular_values = (None if singular_values is None
                                else np.asarray(singular_values, dtype=float))
        # Origin the boundary displacement is measured from, so the reducer
        # online and the identification offline centre it identically.
        self.boundary_reference = (
            None if boundary_reference is None
            else np.asarray(boundary_reference, dtype=float))

    @property
    def n_modes(self):
        return self.phi.shape[1]

    @property
    def boundary_dim(self):
        return 0 if self.psi is None else self.psi.shape[1]

    @property
    def dimension(self):
        return self.phi.shape[0]

    def offset(self, boundary=None):
        """Shape the boundary alone accounts for."""
        if self.psi is None:
            return self.mean
        if boundary is None:
            raise ValueError(
                "this basis has a boundary block; the gripper displacement is "
                "required to project or reconstruct a shape")
        return self.mean + np.asarray(boundary, dtype=float) @ self.psi.T

    def project(self, y, valid=None, boundary=None):
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

        residual = y - self.offset(boundary)
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

    def reconstruct(self, q, boundary=None):
        """Shape vector of a set of modal amplitudes."""
        q = np.asarray(q, dtype=float)
        if q.shape != (self.n_modes,):
            raise ValueError(f"expected {self.n_modes} amplitudes, got {q.shape}")
        return self.offset(boundary) + self.phi @ q

    def reconstruction_rmse(self, y, valid=None, boundary=None):
        """Per-entry RMS residual left by the reduced description [m]."""
        y = np.asarray(y, dtype=float)
        residual = y - self.reconstruct(
            self.project(y, valid, boundary), boundary)
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
        if self.psi is not None:
            data["boundary_dim"] = int(self.boundary_dim)
            data["psi"] = self.psi.flatten(order="C").tolist()
        if self.boundary_reference is not None:
            data["boundary_reference"] = self.boundary_reference.tolist()
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
        psi = None
        if "psi" in section:
            psi = np.asarray(section["psi"], dtype=float).reshape(
                (dimension, int(section["boundary_dim"])), order="C")
        return cls(section["mean"], phi,
                   section.get("marker_s_over_l"),
                   section.get("singular_values"), psi,
                   section.get("boundary_reference"))

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f))
