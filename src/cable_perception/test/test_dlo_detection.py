"""Unit tests of the marker-free DOO detection pipeline (arXiv:2201.06775)."""

import numpy as np
import pytest

from cable_perception.dlo_detection import (
    detect_centreline, fit_chain, merge_chains, merge_cost, orient_chain,
    resample_chain, skeleton_branches, zhang_suen_thin)


def thick_curve_mask(shape=(60, 120), thickness=3):
    """A sinusoidal ribbon: a cable-like blob with a known centreline."""
    mask = np.zeros(shape, dtype=bool)
    cols = np.arange(shape[1])
    rows = (shape[0] / 2 + 8 * np.sin(2 * np.pi * cols / shape[1])).astype(int)
    for offset in range(-(thickness // 2), thickness // 2 + 1):
        mask[np.clip(rows + offset, 0, shape[0] - 1), cols] = True
    return mask


class TestThinning:
    def test_reduces_a_ribbon_to_one_pixel_per_column(self):
        thin = zhang_suen_thin(thick_curve_mask())
        counts = thin.sum(axis=0)[1:-1]  # thinning erodes the two end columns
        assert counts.max() <= 2
        assert (counts >= 1).all()

    def test_keeps_the_component_connected(self):
        thin = zhang_suen_thin(thick_curve_mask())
        branches = skeleton_branches(thin)
        assert len(branches) == 1
        assert len(branches[0]) >= 100

    def test_empty_mask_gives_no_branches(self):
        assert skeleton_branches(zhang_suen_thin(np.zeros((10, 10), bool))) == []

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError):
            zhang_suen_thin(np.zeros((4, 4, 3), bool))


class TestChainFitting:
    def test_segments_have_the_requested_length(self):
        line = np.stack([np.linspace(0, 1, 200), np.zeros(200)], axis=1)
        chain = fit_chain(line, 0.1)[0]
        lengths = np.linalg.norm(np.diff(chain, axis=0), axis=1)
        assert np.allclose(lengths, 0.1, atol=1e-9)
        assert len(chain) == 11

    def test_follows_a_curve(self):
        t = np.linspace(0.0, np.pi, 400)
        arc = np.stack([np.cos(t), np.sin(t)], axis=1)
        chain = fit_chain(arc, 0.2)[0]
        # every vertex stays on the unit circle to within the chord error
        assert np.max(np.abs(np.linalg.norm(chain, axis=1) - 1.0)) < 0.02

    def test_a_sharp_reversal_starts_a_new_chain(self):
        out = np.stack([np.linspace(0, 1, 100), np.zeros(100)], axis=1)
        back = np.stack([np.linspace(1, 0, 100), np.full(100, 0.001)], axis=1)
        assert len(fit_chain(np.vstack([out, back]), 0.1)) == 2

    def test_rejects_a_non_positive_segment_length(self):
        with pytest.raises(ValueError):
            fit_chain(np.zeros((5, 2)), 0.0)


class TestMerging:
    def test_facing_ends_are_cheaper_than_aligned_but_distant_ones(self):
        near = ((np.array([0.0, 0.0]), np.array([1.0, 0.0])),
                (np.array([0.1, 0.02]), np.array([-1.0, 0.0])))
        far = ((np.array([0.0, 0.0]), np.array([1.0, 0.0])),
               (np.array([1.0, 0.0]), np.array([-1.0, 0.0])))
        assert merge_cost(*near) < merge_cost(*far)

    def test_back_to_back_ends_cost_more_than_facing_ones(self):
        facing = ((np.array([0.0, 0.0]), np.array([1.0, 0.0])),
                  (np.array([0.5, 0.0]), np.array([-1.0, 0.0])))
        turned = ((np.array([0.0, 0.0]), np.array([1.0, 0.0])),
                  (np.array([0.5, 0.0]), np.array([0.0, 1.0])))
        assert merge_cost(*facing) < merge_cost(*turned)

    def test_a_gap_is_bridged_into_one_ordered_chain(self):
        left = np.stack([np.linspace(0.0, 0.3, 7), np.zeros(7)], axis=1)
        right = np.stack([np.linspace(0.5, 0.8, 7), np.zeros(7)], axis=1)
        merged = merge_chains([left, right], 0.05)
        assert merged[0][0] == pytest.approx(0.0)
        assert merged[-1][0] == pytest.approx(0.8)
        assert np.all(np.diff(merged[:, 0]) > 0.0)

    def test_merging_preserves_total_span_of_three_pieces(self):
        pieces = [np.stack([np.linspace(a, a + 0.2, 5), np.zeros(5)], axis=1)
                  for a in (0.0, 0.3, 0.6)]
        merged = merge_chains(pieces, 0.05)
        assert merged[:, 0].min() == pytest.approx(0.0)
        assert merged[:, 0].max() == pytest.approx(0.8)

    def test_no_chains_gives_an_empty_result(self):
        assert len(merge_chains([], 0.05)) == 0


class TestResampling:
    def test_arc_length_fractions_of_a_straight_chain(self):
        chain = np.stack([np.linspace(0.0, 1.0, 11), np.zeros(11)], axis=1)
        points = resample_chain(chain, [0.0, 0.25, 0.5, 1.0])
        assert points[:, 0] == pytest.approx([0.0, 0.25, 0.5, 1.0])

    def test_fractions_are_arc_length_not_chord(self):
        chain = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        midpoint = resample_chain(chain, [0.5])[0]
        assert midpoint == pytest.approx([0.0, 1.0])

    def test_out_of_range_fractions_clamp_to_the_ends(self):
        chain = np.stack([np.linspace(0.0, 1.0, 5), np.zeros(5)], axis=1)
        points = resample_chain(chain, [-1.0, 2.0])
        assert points[:, 0] == pytest.approx([0.0, 1.0])

    def test_rejects_a_degenerate_chain(self):
        with pytest.raises(ValueError):
            resample_chain(np.zeros((1, 2)), [0.5])
        with pytest.raises(ValueError):
            resample_chain(np.zeros((4, 2)), [0.5])


class TestOrientation:
    def test_index_zero_ends_up_nearest_the_anchor(self):
        chain = np.stack([np.linspace(1.0, 0.0, 5), np.zeros(5)], axis=1)
        oriented = orient_chain(chain, np.zeros(2))
        assert oriented[0][0] == pytest.approx(0.0)

    def test_an_already_oriented_chain_is_untouched(self):
        chain = np.stack([np.linspace(0.0, 1.0, 5), np.zeros(5)], axis=1)
        assert np.allclose(orient_chain(chain, np.zeros(2)), chain)


class TestEndToEnd:
    def test_a_masked_cable_becomes_one_chain_of_the_right_length(self):
        mask = thick_curve_mask()
        chain = detect_centreline(mask, segment_length_px=5.0)
        assert len(chain) > 10
        length = np.sum(np.linalg.norm(np.diff(chain, axis=0), axis=1))
        # the ribbon spans 120 columns and undulates, so it is a little longer
        assert 115.0 < length < 145.0

    def test_an_occluded_cable_still_yields_a_single_chain(self):
        mask = thick_curve_mask()
        mask[:, 50:62] = False  # the gripper hides a slice of the cable
        chain = detect_centreline(mask, segment_length_px=5.0)
        assert len(chain) > 10
        gaps = np.linalg.norm(np.diff(chain, axis=0), axis=1)
        assert gaps.max() < 12.0  # the occlusion was filled, not left open

    def test_sampling_the_detected_chain_is_monotonic_in_arc_length(self):
        chain = detect_centreline(thick_curve_mask(), segment_length_px=5.0)
        chain = orient_chain(chain, np.zeros(2))
        points = resample_chain(chain, [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0])
        assert np.all(np.diff(points[:, 1]) > 0.0)  # columns increase
