"""Unscented Kalman filter for cable parameter identification.

Pure math: no SOFA and no ROS here. The observation model is injected as a
callable, so the same filter serves the analytic cable used by the unit tests,
the in-process SOFA prediction, and (once the compatibility gate passes) an
Optimus-backed one.

Parameters are filtered in LOG space, ``q = log(theta)``, as the identification
plan requires. Two reasons:

* stiffnesses and damping stay strictly positive whatever the correction does;
* uncertainty becomes relative, which is the honest prior for quantities known
  only to within a factor.

The sigma-point scheme is the classic scaled UKF, which is the same estimator
family as Optimus' ``UKFilterClassic``; results therefore carry over when the
SOFA-side filter replaces this one.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["LogParameterUKF", "UpdateResult"]

# Hard numerical rail on log(theta): exp() underflows to 0.0 near -745, which
# would silently turn a stiffness into zero and poison every later prediction.
_LOG_LIMIT = 50.0


@dataclass(frozen=True)
class UpdateResult:
    """Outcome of a single correction, in SI units."""

    innovation: float      # RMS Euclidean prediction error per point [m]
    marker_rmse: float     # RMS Euclidean error after the correction [m]
    accepted: bool
    reason: str = ""


def _symmetrize(matrix):
    return 0.5 * (matrix + matrix.T)


def _cholesky(matrix):
    """Lower Cholesky factor, adding scaled jitter until the matrix is PD."""
    m = _symmetrize(np.asarray(matrix, dtype=float))
    n = m.shape[0]
    scale = float(np.trace(m)) / n if n else 0.0
    jitter = max(scale, 1.0) * 1e-12
    for _ in range(8):
        try:
            return np.linalg.cholesky(m + jitter * np.eye(n))
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError("parameter covariance is not positive definite")


def _rms_per_point(residual, point_dim):
    """RMS Euclidean norm per observed point (marker), not per component."""
    r = np.asarray(residual, dtype=float).ravel()
    if r.size == 0:
        return 0.0
    if point_dim > 1 and r.size % point_dim == 0:
        r = r.reshape(-1, point_dim)
        return float(np.sqrt(np.mean(np.sum(r * r, axis=1))))
    return float(np.sqrt(np.mean(r * r)))


class LogParameterUKF:
    """Joint UKF over ``log`` of the identified cable parameters.

    The state is small (EI first, then damping per the plan), so the
    ``2n + 1`` sigma points cost only a handful of model evaluations.
    """

    def __init__(self, names, initial_values, initial_relative_std,
                 process_relative_std, bounds=None,
                 alpha=1.0, beta=2.0, kappa=None):
        self.names = tuple(names)
        n = len(self.names)
        if n == 0:
            raise ValueError("at least one parameter must be identified")

        values = np.asarray(initial_values, dtype=float).ravel()
        if values.shape != (n,):
            raise ValueError("initial_values must match names")
        if np.any(values <= 0.0):
            raise ValueError("log-space filtering needs strictly positive values")

        self._q = np.log(values)
        self._p = np.diag(np.asarray(
            initial_relative_std, dtype=float).ravel() ** 2)
        self._process = np.diag(np.asarray(
            process_relative_std, dtype=float).ravel() ** 2)
        if self._p.shape != (n, n) or self._process.shape != (n, n):
            raise ValueError("std vectors must match names")

        self._log_bounds = None
        if bounds is not None:
            b = np.asarray(bounds, dtype=float).reshape(n, 2)
            if np.any(b <= 0.0):
                raise ValueError("parameter bounds must be strictly positive")
            self._log_bounds = np.log(b)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(3 - n) if kappa is None else float(kappa)

        lam = self.alpha ** 2 * (n + self.kappa) - n
        if n + lam <= 0.0:
            raise ValueError("degenerate sigma-point scaling (alpha/kappa)")
        self._lambda = lam
        self._spread = np.sqrt(n + lam)

        self._w_mean = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        self._w_cov = self._w_mean.copy()
        self._w_mean[0] = lam / (n + lam)
        self._w_cov[0] = self._w_mean[0] + (1.0 - self.alpha ** 2 + self.beta)

    # -- state -------------------------------------------------------------
    @property
    def values(self):
        """Current physical parameter values."""
        return np.exp(self._q)

    @property
    def relative_std(self):
        """Std of ``log(theta)``, i.e. the relative uncertainty."""
        return np.sqrt(np.clip(np.diag(self._p), 0.0, None))

    @property
    def std_dev(self):
        """Physical std, delta-method transported out of log space."""
        return self.values * self.relative_std

    @property
    def covariance(self):
        return self._p.copy()

    def value_of(self, name):
        return float(self.values[self.names.index(name)])

    def as_dict(self):
        return {n: float(v) for n, v in zip(self.names, self.values)}

    # -- filtering ---------------------------------------------------------
    def sigma_points(self):
        """``2n + 1`` points in log space; row 0 is the mean."""
        root = _cholesky(self._p) * self._spread
        points = np.repeat(self._q[None, :], 2 * len(self._q) + 1, axis=0)
        for i in range(len(self._q)):
            points[i + 1] += root[:, i]
            points[len(self._q) + i + 1] -= root[:, i]
        return points

    def predict(self):
        """Random-walk process step: the parameters may drift slowly."""
        self._p = _symmetrize(self._p + self._process)

    def update(self, observe, observation, measurement_std,
               mask=None, point_dim=3, max_relative_std=None,
               max_innovation=None):
        """Correct the parameters from one marker observation.

        ``observe(theta) -> array`` must return the predicted observation for a
        physical parameter vector. It is called once per sigma point.
        """
        z = np.asarray(observation, dtype=float).ravel()
        if mask is not None:
            mask = np.asarray(mask, dtype=bool).ravel()
            if mask.shape != z.shape:
                raise ValueError("mask must match the observation length")
            if not mask.any():
                return UpdateResult(0.0, 0.0, False, "no valid markers")

        points = self.sigma_points()
        predictions = []
        for q_i in points:
            pred = np.asarray(observe(np.exp(q_i)), dtype=float).ravel()
            if pred.shape != z.shape:
                raise ValueError(
                    f"observation model returned {pred.shape}, expected {z.shape}")
            predictions.append(pred)
        predictions = np.asarray(predictions)

        if not np.all(np.isfinite(predictions)):
            return UpdateResult(0.0, 0.0, False, "model returned non-finite values")

        if mask is not None:
            z = z[mask]
            predictions = predictions[:, mask]

        z_hat = self._w_mean @ predictions
        d_z = predictions - z_hat
        noise = np.eye(z.size) * float(measurement_std) ** 2
        innovation_cov = _symmetrize((self._w_cov * d_z.T) @ d_z + noise)

        d_q = points - self._q
        cross = (self._w_cov * d_q.T) @ d_z

        nu = z - z_hat
        innovation = _rms_per_point(nu, point_dim)

        # An observation the model cannot explain is an outlier, not evidence:
        # accepting it would drag the parameters towards the rails.
        if max_innovation is not None and innovation > float(max_innovation):
            return UpdateResult(innovation, 0.0, False, "innovation above bound")

        try:
            gain = np.linalg.solve(innovation_cov.T, cross.T).T
        except np.linalg.LinAlgError:
            return UpdateResult(innovation, 0.0, False,
                                "singular innovation covariance")

        q_new = self._q + gain @ nu
        p_new = _symmetrize(self._p - gain @ innovation_cov @ gain.T)

        if not (np.all(np.isfinite(q_new)) and np.all(np.isfinite(p_new))):
            return UpdateResult(innovation, 0.0, False, "correction diverged")

        variance = np.diag(p_new)
        if np.any(variance < 0.0):
            bump = abs(variance.min()) + 1e-15
            p_new = _symmetrize(p_new + bump * np.eye(p_new.shape[0]))
            variance = np.diag(p_new)

        if max_relative_std is not None and np.any(
                np.sqrt(variance) > float(max_relative_std)):
            return UpdateResult(innovation, 0.0, False, "uncertainty above bound")

        q_new = np.clip(q_new, -_LOG_LIMIT, _LOG_LIMIT)
        if self._log_bounds is not None:
            q_new = np.clip(q_new, self._log_bounds[:, 0], self._log_bounds[:, 1])

        # Post-correction residual without a further model evaluation: the
        # cross-covariance carries the linearisation the sigma points already
        # paid for, J ~= C^T P^-1.
        try:
            jacobian = np.linalg.solve(_symmetrize(self._p), cross).T
            residual = nu - jacobian @ (q_new - self._q)
        except np.linalg.LinAlgError:
            residual = nu

        self._q = q_new
        self._p = p_new
        return UpdateResult(innovation, _rms_per_point(residual, point_dim), True)
