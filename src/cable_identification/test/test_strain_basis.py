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