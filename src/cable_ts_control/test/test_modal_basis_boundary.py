"""Tests of the boundary block (boundary-conditioned POD) of the modal basis."""

import numpy as np
import pytest

from cable_ts_control.modal_basis import ModalBasis, build_modal_basis


def dataset(n=200, seed=0):
    """Shapes whose tip is exactly the gripper, as in the real marker set."""
    rng = np.random.default_rng(seed)
    boundary = rng.normal(0.0, 0.05, size=(n, 2))
    bend = rng.normal(0.0, 0.02, size=(n, 1))
    s = np.linspace(0.2, 1.0, 5)[:, None]
    x = 0.7 * s.T * np.ones((n, 1))
    y = bend * np.sin(np.pi * s.T)
    shapes = np.stack([x, y], axis=2)
    shapes[:, :, 0] += boundary[:, 0:1] * s.T
    shapes[:, :, 1] += boundary[:, 1:2] * s.T
    return shapes.reshape(n, -1), boundary


class TestBoundaryBlock:
    def test_a_plain_basis_still_returns_no_psi(self):
        shapes, _ = dataset()
        _, _, _, psi = build_modal_basis(shapes, 2)
        assert psi is None

    def test_the_boundary_removes_its_own_contribution(self):
        shapes, boundary = dataset()
        mean, phi, _, psi = build_modal_basis(shapes, 1, boundary)
        assert psi.shape == (shapes.shape[1], 2)
        q = np.array([ModalBasis(mean, phi, psi=psi).project(y, boundary=b)
                      for y, b in zip(shapes, boundary)])
        correlation = np.corrcoef(np.hstack([q, boundary]).T)[0, 1:]
        assert np.abs(correlation).max() < 1e-8

    def test_it_reconstructs_better_than_plain_pca_at_equal_order(self):
        shapes, boundary = dataset()
        plain = ModalBasis(*build_modal_basis(shapes, 1)[:2])
        split = build_modal_basis(shapes, 1, boundary)
        with_boundary = ModalBasis(split[0], split[1], psi=split[3])
        plain_error = np.mean([plain.reconstruction_rmse(y) for y in shapes])
        split_error = np.mean([with_boundary.reconstruction_rmse(y, boundary=b)
                               for y, b in zip(shapes, boundary)])
        assert split_error < plain_error

    def test_projecting_without_the_boundary_is_refused(self):
        shapes, boundary = dataset()
        mean, phi, _, psi = build_modal_basis(shapes, 1, boundary)
        with pytest.raises(ValueError):
            ModalBasis(mean, phi, psi=psi).project(shapes[0])

    def test_the_boundary_survives_a_round_trip(self, tmp_path):
        shapes, boundary = dataset()
        mean, phi, singular, psi = build_modal_basis(shapes, 2, boundary)
        basis = ModalBasis(mean, phi, [0.5, 1.0], singular, psi,
                           boundary_reference=[0.1, -0.2])
        path = tmp_path / "basis.yaml"
        basis.save(str(path))
        loaded = ModalBasis.load(str(path))
        assert loaded.boundary_dim == 2
        assert loaded.psi == pytest.approx(psi)
        assert loaded.boundary_reference == pytest.approx([0.1, -0.2])
        assert loaded.project(shapes[3], boundary=boundary[3]) == pytest.approx(
            basis.project(shapes[3], boundary=boundary[3]))

    def test_a_mismatched_psi_is_rejected(self):
        with pytest.raises(ValueError):
            ModalBasis(np.zeros(10), np.zeros((10, 2)), psi=np.zeros((8, 2)))
