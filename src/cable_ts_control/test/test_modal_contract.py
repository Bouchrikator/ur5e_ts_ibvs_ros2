"""Pure-numpy checks of the Cosserat-modal contract, observer and sparse LMI backend."""

import numpy as np
import pytest

from cable_ts_control.modal_contract import (
    COORDINATES, build_modal_state, check_dataset_contract, check_model_contract, target_state)
from cable_ts_control.modal_observer import ModalObserver

BASIS = {"state_coordinates": COORDINATES, "n_modes": 3, "strain_basis_sha256": "abc"}
MODEL = {"state_coordinates": COORDINATES, "n_modes": 3, "state_dim": 8, "strain_basis_sha256": "abc"}


def test_state_is_a_adot_gripper_without_any_marker_projection():
    x = build_modal_state([[1.0, 2.0, 3.0]], [[0.1, 0.2, 0.3]], [[0.7, 0.1]], [0.6, 0.0])
    np.testing.assert_allclose(x, [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.1, 0.1]])


def test_legacy_marker_dataset_is_refused():
    with pytest.raises(ValueError, match="Cosserat-modal fields"):
        check_dataset_contract({"shapes": np.zeros((2, 14)), "gripper": np.zeros((2, 2))}, BASIS)


def test_model_contract_rejects_wrong_dimension_hash_and_coordinates():
    assert check_model_contract(MODEL, BASIS) == COORDINATES
    with pytest.raises(ValueError, match="state_dim"):
        check_model_contract({**MODEL, "state_dim": 6}, BASIS)
    with pytest.raises(ValueError, match="hashes differ"):
        check_model_contract(MODEL, {**BASIS, "strain_basis_sha256": "other"})
    with pytest.raises(ValueError, match="coordinates"):
        check_model_contract({"n_modes": 3, "state_dim": 8}, expected_coordinates=COORDINATES)
    with pytest.raises(ValueError, match="gains"):
        check_model_contract(MODEL, BASIS, gains={"strain_basis_sha256": "other"})


def test_target_accepts_a_star_and_a_star_with_gripper():
    np.testing.assert_allclose(target_state([1, 2, 3], 3, 8), [1, 2, 3, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(target_state([1, 2, 3, 4, 5], 3, 8), [1, 2, 3, 0, 0, 0, 4, 5])
    with pytest.raises(ValueError):
        target_state([1, 2], 3, 8)


def test_observer_recovers_coordinates_with_occlusion_and_prior():
    basis = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])

    def observe(a):
        return basis @ a + 0.05 * np.square(basis @ a)  # mildly nonlinear map
    truth = np.array([0.3, -0.2])
    observer = ModalObserver(observe, 2, marker_std_m=1e-3, prior_std=10.0)
    a_hat, info = observer.update(observe(truth), [True, True], prior=np.zeros(2))
    np.testing.assert_allclose(a_hat, truth, atol=1e-6)
    assert info["rank"] == 2
    # Two coordinates left (one planar marker): still solvable thanks to the prior.
    a_hat, info = observer.update(observe(truth), [True, False], prior=truth + 0.01)
    assert info["observed"] == 2 and np.linalg.norm(a_hat - truth) < 0.01


def test_sparse_lmi_backend_matches_cvxpy_and_is_certified():
    pytest.importorskip("clarabel")
    pytest.importorskip("scs")
    pytest.importorskip("cvxpy")
    from cable_ts_control.lmi_synthesis import (
        solve_cable_ts_pdc, solve_cable_ts_pdc_sparse, verify_certificate)
    from test_lmi_synthesis import make_vertices
    vertex_sets = [make_vertices(), make_vertices(stiffness_scale=1.4)]
    for relaxed in (False, True):
        dense = solve_cable_ts_pdc(vertex_sets, relaxed=relaxed)
        assert dense.feasible
        for solver in ("clarabel", "scs"):
            sparse = solve_cable_ts_pdc_sparse(vertex_sets, relaxed=relaxed, solver=solver)
            assert sparse.feasible, solver
            assert verify_certificate(vertex_sets, sparse)["satisfied"], solver
            assert np.trace(np.linalg.inv(sparse.lyapunov)) == pytest.approx(
                np.trace(np.linalg.inv(dense.lyapunov)), rel=2e-2 if solver == "clarabel" else 0.2), solver
