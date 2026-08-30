"""Corruption of ideal cable marker observations.

Pure numerical logic, no ROS: the sensor imperfections we want to be robust to
(noise, dropout, latency) are a modelling concern, kept separate from the node
that carries the messages so both can be unit tested and reused by the camera
tracker.
"""

import numpy as np


class ObservationCorruptor:
    """Turns exact marker positions into realistic noisy observations.

    Parameters
    ----------
    noise_std_m
        Standard deviation of the isotropic Gaussian position noise [m].
    dropout_probability
        Per-marker, per-sample probability of being reported invalid.
    seed
        Fixes the random stream so an experiment can be replayed exactly.
    """

    def __init__(self, noise_std_m=0.002, dropout_probability=0.0, seed=None):
        if noise_std_m < 0.0:
            raise ValueError("noise_std_m must be >= 0")
        if not 0.0 <= dropout_probability <= 1.0:
            raise ValueError("dropout_probability must be in [0, 1]")
        self.noise_std_m = float(noise_std_m)
        self.dropout_probability = float(dropout_probability)
        self._rng = np.random.default_rng(seed)

    def covariance(self):
        """Row-major 3x3 isotropic position covariance [m^2]."""
        var = self.noise_std_m ** 2
        return [var, 0.0, 0.0,
                0.0, var, 0.0,
                0.0, 0.0, var]

    def corrupt(self, positions):
        """Apply noise and dropout to an (M, 3) array of exact positions.

        Returns ``(noisy_positions, valid_mask)``. Dropped markers keep their
        noisy position so downstream code can still log them, but the mask
        marks them unusable.
        """
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"expected an (M, 3) array, got {positions.shape}")

        noisy = positions + self._rng.normal(
            0.0, self.noise_std_m, size=positions.shape)
        valid = self._rng.random(positions.shape[0]) >= self.dropout_probability
        return noisy, valid


class DelayLine:
    """Holds samples back by a fixed latency, using their own timestamps.

    Models the end-to-end sensing delay (exposure, transport, processing) that
    the outer control loop has to tolerate.
    """

    def __init__(self, delay_s=0.0):
        if delay_s < 0.0:
            raise ValueError("delay_s must be >= 0")
        self.delay_s = float(delay_s)
        self._queue = []

    def push(self, stamp_s, payload):
        self._queue.append((float(stamp_s), payload))

    def pop_ready(self, now_s):
        """Return every payload whose delay has elapsed, oldest first."""
        ready, pending = [], []
        for stamp_s, payload in self._queue:
            if now_s - stamp_s >= self.delay_s:
                ready.append((stamp_s, payload))
            else:
                pending.append((stamp_s, payload))
        self._queue = pending
        return ready

    def __len__(self):
        return len(self._queue)
