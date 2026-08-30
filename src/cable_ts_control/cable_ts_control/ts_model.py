"""Takagi-Sugeno model and PDC control law for the cable shape.

Step 7/8 of the plan. The reduced cable obeys

    x(k+1) = sum_i h_i(rho) ( A_i x(k) + B_i u(k) )
    u(k)   = -sum_i h_i(rho) K_i ( x(k) - x_star )

with ``x = [q; qdot]`` the modal state, ``u`` the planar gripper velocity and
the premises ``rho = (q1, q2)`` — bounded, observable from the markers, and
carrying the dominant geometric nonlinearity.

EI and damping are handled as bounded UNCERTAINTIES rather than extra premises:
the same gains must stabilise every parameter vertex. Two premises therefore
give 4 online rules instead of an exploding rule base.

Pure numerics: no ROS, no SOFA, no solver. Shared by the identification
scripts, the LMI solver and the controller node.
"""

import numpy as np
import yaml


def triangular_memberships(value, lower, upper):
    """Two complementary memberships of a scalar premise on [lower, upper].

    Returns ``(w_low, w_high)``, both in [0, 1] and summing to 1. Values
    outside the range saturate, which keeps the controller defined even when
    the cable leaves the region the model was identified on.
    """
    if upper <= lower:
        raise ValueError(f"premise range must satisfy lower < upper, got [{lower}, {upper}]")
    high = (float(value) - lower) / (upper - lower)
    high = min(1.0, max(0.0, high))
    return 1.0 - high, high


def rule_memberships(premises, bounds):
    """Membership of every rule of the full premise grid.

    ``premises`` are the scalar premise values, ``bounds`` the matching
    ``(lower, upper)`` pairs. Rule ordering is the row-major product of the
    per-premise (low, high) pairs: rule index ``i = sum_j b_j * 2**(n-1-j)``
    with ``b_j = 0`` for low and ``1`` for high. There are ``2**n`` rules.
    """
    premises = list(premises)
    bounds = list(bounds)
    if len(premises) != len(bounds):
        raise ValueError(
            f"got {len(premises)} premises but {len(bounds)} bound pairs")
    if not premises:
        raise ValueError("at least one premise is required")

    per_premise = [triangular_memberships(v, *b) for v, b in zip(premises, bounds)]

    weights = np.ones(2 ** len(premises))
    for index in range(weights.size):
        for position, (low, high) in enumerate(per_premise):
            bit = (index >> (len(premises) - 1 - position)) & 1
            weights[index] *= high if bit else low
    return weights


class CableTsModel:
    """Vertex matrices of the reduced cable plus the PDC gains."""

    def __init__(self, a_vertices, b_vertices, premise_bounds, sample_time,
                 gains=None, lyapunov=None, parameter_bounds=None,
                 n_modes=None):
        self.a_vertices = [np.asarray(a, dtype=float) for a in a_vertices]
        self.b_vertices = [np.asarray(b, dtype=float) for b in b_vertices]
        self.premise_bounds = [tuple(float(v) for v in pair) for pair in premise_bounds]
        self.sample_time = float(sample_time)
        self.gains = None if gains is None else [np.asarray(k, dtype=float) for k in gains]
        self.lyapunov = None if lyapunov is None else np.asarray(lyapunov, dtype=float)
        self.parameter_bounds = parameter_bounds or {}
        # Explicit, because the state may carry more than [q, qdot]: with the
        # gripper position appended, state_dim // 2 is simply wrong.
        self.n_modes = (self.state_dim // 2 if n_modes is None else int(n_modes))
        self._validate()

    def _validate(self):
        expected = 2 ** len(self.premise_bounds)
        if len(self.a_vertices) != expected or len(self.b_vertices) != expected:
            raise ValueError(
                f"{len(self.premise_bounds)} premises need {expected} vertex "
                f"matrices, got {len(self.a_vertices)} A and "
                f"{len(self.b_vertices)} B")
        n = self.state_dim
        for a in self.a_vertices:
            if a.shape != (n, n):
                raise ValueError(f"every A vertex must be {n}x{n}, got {a.shape}")
        m = self.input_dim
        for b in self.b_vertices:
            if b.shape != (n, m):
                raise ValueError(f"every B vertex must be {n}x{m}, got {b.shape}")
        if self.gains is not None:
            for k in self.gains:
                if k.shape != (m, n):
                    raise ValueError(f"every gain must be {m}x{n}, got {k.shape}")
        if self.sample_time <= 0.0:
            raise ValueError("sample_time must be > 0")

    @property
    def state_dim(self):
        return self.a_vertices[0].shape[0]

    @property
    def input_dim(self):
        return self.b_vertices[0].shape[1]

    @property
    def n_rules(self):
        return len(self.a_vertices)

    def memberships(self, premises):
        return rule_memberships(premises, self.premise_bounds)

    def blend(self, premises):
        """Interpolated ``(A, B)`` at the current operating point."""
        h = self.memberships(premises)
        a = sum(w * m for w, m in zip(h, self.a_vertices))
        b = sum(w * m for w, m in zip(h, self.b_vertices))
        return a, b, h

    def predict(self, state, command, premises=None):
        """One-step prediction of the reduced model."""
        state = np.asarray(state, dtype=float)
        command = np.asarray(command, dtype=float)
        if premises is None:
            premises = state[:len(self.premise_bounds)]
        a, b, _ = self.blend(premises)
        return a @ state + b @ command

    def control(self, state, reference):
        """PDC command ``u = -sum_i h_i K_i (x - x_star)``."""
        if self.gains is None:
            raise RuntimeError("no PDC gains loaded in this model")
        state = np.asarray(state, dtype=float)
        reference = np.asarray(reference, dtype=float)
        error = state - reference
        h = self.memberships(state[:len(self.premise_bounds)])
        gain = sum(w * k for w, k in zip(h, self.gains))
        return -gain @ error, h

    def lyapunov_value(self, state, reference):
        """``V = e' P e`` of the stored Lyapunov certificate."""
        if self.lyapunov is None:
            raise RuntimeError("no Lyapunov matrix loaded in this model")
        error = np.asarray(state, dtype=float) - np.asarray(reference, dtype=float)
        return float(error @ self.lyapunov @ error)

    def closed_loop_spectral_radius(self):
        """Worst ``max |eig(A_i - B_i K_j)|`` over every rule combination.

        Below 1 is necessary (not sufficient) for the fuzzy closed loop; the
        LMI certificate is what actually proves stability.
        """
        if self.gains is None:
            raise RuntimeError("no PDC gains loaded in this model")
        worst = 0.0
        for a, b in zip(self.a_vertices, self.b_vertices):
            for k in self.gains:
                eigenvalues = np.linalg.eigvals(a - b @ k)
                worst = max(worst, float(np.max(np.abs(eigenvalues))))
        return worst

    # -- persistence ---------------------------------------------------
    def to_dict(self):
        data = {
            "state_dim": int(self.state_dim),
            "input_dim": int(self.input_dim),
            "n_rules": int(self.n_rules),
            "n_modes": int(self.n_modes),
            "sample_time": self.sample_time,
            "premise_bounds": [list(pair) for pair in self.premise_bounds],
            "A": [a.flatten(order="C").tolist() for a in self.a_vertices],
            "B": [b.flatten(order="C").tolist() for b in self.b_vertices],
        }
        if self.parameter_bounds:
            data["parameter_bounds"] = self.parameter_bounds
        if self.gains is not None:
            data["K"] = [k.flatten(order="C").tolist() for k in self.gains]
        if self.lyapunov is not None:
            data["P"] = self.lyapunov.flatten(order="C").tolist()
        return {"cable_ts_model": data}

    def save(self, path):
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, width=200)

    @classmethod
    def from_dict(cls, data):
        section = data["cable_ts_model"]
        n = int(section["state_dim"])
        m = int(section["input_dim"])
        a = [np.asarray(v, dtype=float).reshape((n, n), order="C")
             for v in section["A"]]
        b = [np.asarray(v, dtype=float).reshape((n, m), order="C")
             for v in section["B"]]
        gains = ([np.asarray(v, dtype=float).reshape((m, n), order="C")
                  for v in section["K"]] if "K" in section else None)
        lyapunov = (np.asarray(section["P"], dtype=float).reshape((n, n), order="C")
                    if "P" in section else None)
        return cls(a, b, section["premise_bounds"], section["sample_time"],
                   gains, lyapunov, section.get("parameter_bounds"),
                   section.get("n_modes"))

    @classmethod
    def load(cls, path, gains_path=None):
        """Load a model, optionally merging gains stored in a separate file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if gains_path:
            with open(gains_path, "r") as f:
                gains = yaml.safe_load(f)["cable_ts_model"]
            data["cable_ts_model"].update(
                {key: gains[key] for key in ("K", "P") if key in gains})
        return cls.from_dict(data)
