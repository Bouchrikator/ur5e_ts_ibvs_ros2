"""Tests of the Takagi-Sugeno premise maps."""

import numpy as np
import pytest

from cable_ts_control.premises import (
    BoundaryPremises, ModalPremises, make_premise_map, premise_map_from_dict)


class TestModalPremises:
    def test_takes_the_leading_state_components(self):
        assert ModalPremises(2)(np.arange(6.0)) == pytest.approx([0.0, 1.0])

    def test_is_vectorised(self):
        states = np.arange(12.0).reshape(2, 6)
        assert ModalPremises(2)(states).shape == (2, 2)

    def test_rejects_a_zero_premise_count(self):
        with pytest.raises(ValueError):
            ModalPremises(0)


class TestBoundaryPremises:
    def test_foreshortening_and_bearing_of_a_straight_rod(self):
        premises = BoundaryPremises(gripper_index=4, cable_length_m=1.0)
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0])
        assert premises(state) == pytest.approx([0.5, 0.0])

    def test_the_reference_is_added_back(self):
        premises = BoundaryPremises(4, 1.0, gripper_reference=[0.5, 0.0])
        state = np.zeros(6)
        assert premises(state) == pytest.approx([0.5, 0.0])

    def test_bearing_follows_the_gripper(self):
        premises = BoundaryPremises(4, 1.0)
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        assert premises(state) == pytest.approx([0.5, np.pi / 2])

    def test_is_vectorised(self):
        premises = BoundaryPremises(4, 1.0)
        states = np.zeros((3, 6))
        states[:, 4] = [0.2, 0.5, 0.8]
        assert premises(states)[:, 0] == pytest.approx([0.8, 0.5, 0.2])

    def test_is_not_a_state_component(self):
        """The whole point: no premise repeats a column of the state."""
        premises = BoundaryPremises(4, 0.7)
        rng = np.random.default_rng(0)
        states = rng.normal(size=(200, 6))
        values = premises(states)
        for column in range(6):
            for premise in range(2):
                correlation = np.corrcoef(states[:, column], values[:, premise])[0, 1]
                assert abs(correlation) < 0.999

    def test_rejects_a_state_without_a_gripper_block(self):
        with pytest.raises(ValueError):
            BoundaryPremises(4, 1.0)(np.zeros(4))

    def test_rejects_a_non_positive_length(self):
        with pytest.raises(ValueError):
            BoundaryPremises(4, 0.0)


class TestSerialisation:
    def test_round_trip_of_each_kind(self):
        for premises in (ModalPremises(3), BoundaryPremises(4, 0.7, [0.1, 0.2])):
            restored = premise_map_from_dict(premises.to_dict())
            assert restored.n_premises == premises.n_premises
            state = np.arange(8.0)
            assert restored(state) == pytest.approx(premises(state))

    def test_none_stays_none(self):
        assert premise_map_from_dict(None) is None

    def test_an_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError):
            premise_map_from_dict({"kind": "nonsense"})
        with pytest.raises(ValueError):
            make_premise_map("nonsense")

    def test_boundary_needs_its_geometry(self):
        with pytest.raises(ValueError):
            make_premise_map("boundary")
