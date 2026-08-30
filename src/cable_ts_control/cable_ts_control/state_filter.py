"""Causal modal velocity estimation.

The identification and the running controller must see the SAME state signal.
Fitting on ``np.gradient`` (centred, non-causal, unsmoothed) and then executing
on a causal low-pass of a backward difference means the model was never
identified for the state it is given at runtime.

Pure numerics, no ROS: the reducer node and the offline pipeline both import
this so they cannot drift apart.
"""

import numpy as np

__all__ = ["ModalVelocityFilter", "velocity_series"]


class ModalVelocityFilter:
    """First-order low-pass on the backward difference of the modal state.

    ``alpha`` is the weight of the newest raw difference: 1.0 is an unfiltered
    backward difference, small values are heavily smoothed but lag more.
    """

    def __init__(self, n_modes, alpha=0.4):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.n_modes = int(n_modes)
        self.alpha = float(alpha)
        self.reset()

    def reset(self):
        self._previous_q = None
        self._velocity = np.zeros(self.n_modes)

    @property
    def velocity(self):
        return self._velocity.copy()

    def update(self, q, dt):
        """Feed one modal sample and return the filtered velocity."""
        q = np.asarray(q, dtype=float).ravel()
        if q.shape != (self.n_modes,):
            raise ValueError(
                f"expected {self.n_modes} modal coordinates, got {q.shape}")
        if self._previous_q is not None and dt > 1e-6:
            raw = (q - self._previous_q) / dt
            self._velocity = self.alpha * raw + (1.0 - self.alpha) * self._velocity
        self._previous_q = q
        return self.velocity


def velocity_series(modal, dt, alpha=0.4):
    """Filtered velocities for a whole modal trajectory, sample by sample.

    Uses exactly the same recursion as the runtime filter, so an offline state
    trajectory matches what the reducer node would have produced from the same
    samples.
    """
    modal = np.asarray(modal, dtype=float)
    if modal.ndim != 2:
        raise ValueError(f"expected a (T, n_modes) array, got {modal.shape}")
    filt = ModalVelocityFilter(modal.shape[1], alpha)
    return np.array([filt.update(q, dt) for q in modal])
