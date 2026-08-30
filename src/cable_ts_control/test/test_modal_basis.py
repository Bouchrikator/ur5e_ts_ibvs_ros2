"""Unit tests for the modal reduction of the cable shape."""

import numpy as np
import pytest

from cable_ts_control.modal_basis import (
    ModalBasis,
    build_modal_basis,
    explained_variance_ratio,
)


def make_dataset(n_samples=200, seed=0):
    """Shapes built from exactly two known modes plus a tiny third one."""
    rng = np.random.default_rng(seed)
    s = np.linspace(0.0, 1.0, 6)
    mode_a = np.concatenate([s, np.sin(np.pi * s)])
    mode_b = np.concatenate([s ** 2, np.sin(2 * np.pi * s)])
    mode_c = np.concatenate([s ** 3, np.sin(3 * np.pi * s)])

    amplitudes = rng.normal(0.0, [0.05, 0.02, 1e-5], size=(n_samples, 3))
    offset = np.full(mode_a.shape, 0.3)
    shapes = (offset
              + np.outer(amplitudes[:, 0], mode_a)
              + np.outer(amplitudes[:, 1], mode_b)
              + np.outer(amplitudes[:, 2], mode_c))
    return shapes


def test_basis_columns_are_orthonormal():
    _, phi, _, _ = build_modal_basis(make_dataset(), n_modes=2)

    assert phi.T @ phi == pytest.approx(np.eye(2), abs=1e-9)


def test_mean_matches_the_dataset_mean():
    shapes = make_dataset()
    mean, _, _, _ = build_modal_basis(shapes, n_modes=2)

    assert mean == pytest.approx(shapes.mean(axis=0))


def test_two_modes_capture_almost_all_the_energy():
    _, _, singular_values, _ = build_modal_basis(make_dataset(), n_modes=2)

    assert explained_variance_ratio(singular_values, 2) > 0.999


def test_explained_variance_is_monotonic():
    _, _, singular_values, _ = build_modal_basis(make_dataset(), n_modes=3)
    ratios = [explained_variance_ratio(singular_values, k) for k in (1, 2, 3)]

    assert ratios[0] <= ratios[1] <= ratios[2]


def test_project_and_reconstruct_round_trip_a_training_shape():
    shapes = make_dataset()
    mean, phi, _, _ = build_modal_basis(shapes, n_modes=2)
    basis = ModalBasis(mean, phi)

    recovered = basis.reconstruct(basis.project(shapes[0]))

    assert recovered == pytest.approx(shapes[0], abs=1e-4)


def test_reconstruction_rmse_is_small_on_training_data():
    shapes = make_dataset()
    mean, phi, _, _ = build_modal_basis(shapes, n_modes=2)
    basis = ModalBasis(mean, phi)

    assert basis.reconstruction_rmse(shapes[5]) < 1e-4


def test_projection_recovers_amplitudes_despite_missing_markers():
    """Dropout must not stop the reduced state from being identified."""
    shapes = make_dataset()
    mean, phi, _, _ = build_modal_basis(shapes, n_modes=2)
    basis = ModalBasis(mean, phi)

    full = basis.project(shapes[3])
    valid = np.ones(shapes.shape[1], dtype=bool)
    valid[[0, 7]] = False  # two entries unobserved

    partial = basis.project(shapes[3], valid)

    assert partial == pytest.approx(full, abs=1e-3)


def test_projection_fails_when_too_few_entries_remain():
    shapes = make_dataset()
    mean, phi, _, _ = build_modal_basis(shapes, n_modes=2)
    basis = ModalBasis(mean, phi)

    valid = np.zeros(shapes.shape[1], dtype=bool)
    valid[0] = True

    with pytest.raises(ValueError):
        basis.project(shapes[0], valid)


def test_wrong_sized_shape_vector_is_rejected():
    mean, phi, _, _ = build_modal_basis(make_dataset(), n_modes=2)
    basis = ModalBasis(mean, phi)

    with pytest.raises(ValueError):
        basis.project(np.zeros(3))


def test_wrong_sized_amplitudes_are_rejected():
    mean, phi, _, _ = build_modal_basis(make_dataset(), n_modes=2)
    basis = ModalBasis(mean, phi)

    with pytest.raises(ValueError):
        basis.reconstruct(np.zeros(5))


def test_requesting_more_modes_than_the_data_supports_is_rejected():
    with pytest.raises(ValueError):
        build_modal_basis(np.zeros((3, 12)), n_modes=8)


def test_yaml_round_trip_preserves_the_basis(tmp_path):
    shapes = make_dataset()
    mean, phi, singular_values, _ = build_modal_basis(shapes, n_modes=2)
    basis = ModalBasis(mean, phi, [0.1, 0.5, 1.0], singular_values)

    path = tmp_path / "basis.yaml"
    basis.save(path)
    loaded = ModalBasis.load(path)

    assert loaded.n_modes == 2
    assert loaded.dimension == basis.dimension
    assert loaded.mean == pytest.approx(basis.mean)
    assert loaded.phi == pytest.approx(basis.phi)
    assert loaded.marker_s_over_l == [0.1, 0.5, 1.0]
    assert loaded.project(shapes[1]) == pytest.approx(basis.project(shapes[1]))


def test_mismatched_mean_and_phi_are_rejected():
    with pytest.raises(ValueError):
        ModalBasis(np.zeros(10), np.zeros((8, 2)))
