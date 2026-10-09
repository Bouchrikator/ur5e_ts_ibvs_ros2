"""Small file-boundary checks; no SOFA runtime needed."""

from pathlib import Path
import tempfile

import numpy as np

from cable_identification.strain_basis import read_modes, read_state_file


def test_native_state_and_exact_mode_format():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.txt"
        path.write_text("T= 0\n  X0= 0 0 0\n  X= 0 0 1\n  V= 0 0 2\n"
                        "T= 0.04\n  X= 0 0 1.08\n  V= 0 0 2\n")
        state = read_state_file(path, 3)
        np.testing.assert_array_equal(state["strain"], [[0, 0, 1], [0, 0, 1.08]])
        np.testing.assert_array_equal(state["times"], [0, 0.04])
        path.write_text("3 1\n0\n0\n1\n")
        np.testing.assert_array_equal(read_modes(path), [[0], [0], [1]])
        for invalid in ("3 1\n0\n1\n", "3 1\n0 1\n0\n1\n", "3 1\n0\n0\nnan\n"):
            path.write_text(invalid)
            try:
                read_modes(path)
            except ValueError:
                continue
            raise AssertionError("Malformed MatrixLoader input was accepted")

def test_weighted_modes_round_trip_and_projection():
    from cable_identification.strain_basis import (
        project_strain, scale_state_file, weighted_norm, write_modes,
    )

    rng = np.random.default_rng(3)
    weights = np.array([4.4e-4, 4.4e-4, 4.4e-4, 219.0, 87.5, 87.5])
    snapshots = rng.normal(size=(50, 6)) * np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3])
    with tempfile.TemporaryDirectory() as directory:
        state = Path(directory) / "training.state"
        state.write_text("".join(f"T= {0.04 * k}\n  X0= 0 0 0 0 0 0\n  X= {' '.join(map(repr, row))}\n"
                                 f"  V= {' '.join(map(repr, row))}\n" for k, row in enumerate(snapshots)))
        scaled = Path(directory) / "weighted.state"
        scale_state_file(state, scaled, np.sqrt(weights))
        scaled_state = read_state_file(scaled, 6)
        np.testing.assert_allclose(scaled_state["strain"], snapshots * np.sqrt(weights), rtol=1e-15)
        # Euclidean POD of the scaled snapshots, unscaled: W-orthonormal, exact text round trip
        u = np.linalg.svd(scaled_state["strain"].T, full_matrices=False)[0][:, :3]
        modes = u / np.sqrt(weights)[:, None]
        modes_path = Path(directory) / "modes.txt"
        write_modes(modes_path, modes)
        np.testing.assert_array_equal(read_modes(modes_path), modes)
    np.testing.assert_allclose(modes.T @ (weights[:, None] * modes), np.eye(3), atol=1e-12)
    # a = Phi^T W (q - q0) reproduces every state inside the span, and the W-norm residual of
    # a state outside the span is orthogonal to the span
    inside = snapshots[:5] @ (weights[:, None] * modes) @ modes.T
    np.testing.assert_allclose(project_strain(modes, weights, inside) @ modes.T, inside, atol=1e-12)
    residual = snapshots[:5] - project_strain(modes, weights, snapshots[:5]) @ modes.T
    assert np.all(np.abs(residual @ (weights[:, None] * modes)) < 1e-9)
    assert abs(weighted_norm(weights, snapshots[0]) - np.sqrt(snapshots[0] @ (weights * snapshots[0]))) < 1e-15
