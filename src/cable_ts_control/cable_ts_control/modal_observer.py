"""Estimate the ONE strain-POD coordinate ``a`` from planar markers.

    a_hat = argmin ||W^1/2 (y - h(a))||^2 + lambda ||a - a_prior||^2

``h`` is injected (the SOFA ``Phi a -> DiscreteCosseratMapping -> markers``
chain in production, anything callable in the tests), so the estimator stays
pure numpy. It is a regularised Gauss-Newton step, not a POD: with 7 planar
markers ``rank(J) <= 14``, so for ``r > 14`` the prior (TS prediction or the
previous estimate) is what makes the problem well posed. Occluded markers get
zero weight.
"""

import numpy as np


class ModalObserver:
    def __init__(self, observe, n_modes, marker_std_m, prior_std, max_iterations=3,
                 jacobian_step=1e-3, step_tolerance=1e-7):
        if marker_std_m <= 0.0 or prior_std <= 0.0 or max_iterations < 1:
            raise ValueError("marker_std_m, prior_std must be > 0 and max_iterations >= 1")
        self.observe = observe
        self.n_modes = int(n_modes)
        self.marker_weight = 1.0 / marker_std_m ** 2
        self.prior_weight = 1.0 / prior_std ** 2
        self.max_iterations = int(max_iterations)
        self.jacobian_step = float(jacobian_step)
        self.step_tolerance = float(step_tolerance)

    def jacobian(self, a, h=None):
        """Forward differences of ``h`` about ``a`` (r + 1 evaluations)."""
        h = self.observe(a) if h is None else h
        columns = []
        for index in range(self.n_modes):
            shifted = np.array(a, dtype=float)
            shifted[index] += self.jacobian_step
            columns.append((self.observe(shifted) - h) / self.jacobian_step)
        return np.column_stack(columns)

    def update(self, markers, valid, prior):
        """One measurement update. Returns ``(a_hat, info)``.

        ``markers`` is the flattened planar marker vector, ``valid`` a boolean
        per marker (or per coordinate), ``prior`` the predicted ``a``.
        """
        y = np.asarray(markers, dtype=float).ravel()
        valid = np.asarray(valid, dtype=bool).ravel()
        if valid.size * 2 == y.size:
            valid = np.repeat(valid, 2)
        if valid.size != y.size:
            raise ValueError("valid mask does not match the marker vector")
        prior = np.asarray(prior, dtype=float).ravel()
        if prior.size != self.n_modes:
            raise ValueError(f"prior has {prior.size} entries, expected {self.n_modes}")
        weights = np.where(valid, self.marker_weight, 0.0)
        a = prior.copy()
        info = {"iterations": 0, "rank": 0, "condition": np.inf, "residual_rmse_m": np.nan,
                "observed": int(valid.sum())}
        for iteration in range(self.max_iterations):
            h = self.observe(a)
            jac = self.jacobian(a, h)
            residual = np.where(valid, y - h, 0.0)
            normal = jac.T @ (weights[:, None] * jac) + self.prior_weight * np.eye(self.n_modes)
            rhs = jac.T @ (weights * residual) - self.prior_weight * (a - prior)
            delta = np.linalg.solve(normal, rhs)
            a = a + delta
            info["iterations"] = iteration + 1
            if iteration == 0:
                info["rank"] = int(np.linalg.matrix_rank(jac[valid])) if valid.any() else 0
                info["condition"] = float(np.linalg.cond(normal))
            if np.linalg.norm(delta) < self.step_tolerance:
                break
        final = np.where(valid, y - self.observe(a), 0.0)
        info["residual_rmse_m"] = (float(np.sqrt(np.mean(final[valid] ** 2)))
                                   if valid.any() else float("nan"))
        return a, info
