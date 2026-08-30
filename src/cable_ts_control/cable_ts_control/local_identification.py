"""Local linear identification of the reduced cable dynamics.

Step 7 of the plan: for each physical-parameter vertex and each shape region,
SOFA rollouts give triplets ``(x_k, u_k, x_k+1)``. A least-squares fit gives
the local matrices of that region,

    x_k+1 ~= A x_k + B u_k

which become the ``A_i, B_i`` vertices of the Takagi-Sugeno model.

Pure numerics so the fit and its acceptance criteria can be unit tested without
SOFA in the loop.
"""

import numpy as np


def fit_local_model(states, commands, next_states, ridge=0.0):
    """Least-squares fit of ``x_next = A x + B u``.

    ``ridge`` adds Tikhonov regularisation, which matters when an excitation
    direction is poorly covered by the dataset.
    """
    states = np.asarray(states, dtype=float)
    commands = np.asarray(commands, dtype=float)
    next_states = np.asarray(next_states, dtype=float)

    if states.ndim != 2 or commands.ndim != 2 or next_states.ndim != 2:
        raise ValueError("states, commands and next_states must all be 2-D")
    if not (len(states) == len(commands) == len(next_states)):
        raise ValueError(
            f"sample count mismatch: {len(states)} states, {len(commands)} "
            f"commands, {len(next_states)} next states")

    n, m = states.shape[1], commands.shape[1]
    if len(states) < n + m:
        raise ValueError(
            f"need at least {n + m} samples to identify a {n}x{n} A and a "
            f"{n}x{m} B, got {len(states)}")

    regressor = np.hstack([states, commands])
    if ridge > 0.0:
        gram = regressor.T @ regressor + ridge * np.eye(n + m)
        theta = np.linalg.solve(gram, regressor.T @ next_states)
    else:
        theta, *_ = np.linalg.lstsq(regressor, next_states, rcond=None)

    return theta[:n].T, theta[n:].T


def one_step_rmse(a, b, states, commands, next_states, position_dim=None):
    """RMS one-step prediction error of a local model.

    ``position_dim`` restricts the error to the leading position components.
    The state is ``[q, q_dot]``, so averaging over all of it mixes metres with
    metres per second: the result is dimensionless nonsense dominated by the
    velocity term, and cannot be compared against a tolerance in millimetres.
    """
    states = np.asarray(states, dtype=float)
    commands = np.asarray(commands, dtype=float)
    next_states = np.asarray(next_states, dtype=float)
    predicted = states @ np.asarray(a, dtype=float).T + commands @ np.asarray(b, dtype=float).T
    residual = predicted - next_states
    if position_dim is not None:
        residual = residual[:, :position_dim]
    return float(np.sqrt(np.mean(residual ** 2)))


def rollout_rmse(a, b, initial_state, commands, reference_states):
    """RMS error of a free-running rollout driven only by the commands.

    Much harsher than the one-step error and the honest way to check that a
    local model is usable for prediction.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    commands = np.asarray(commands, dtype=float)
    reference_states = np.asarray(reference_states, dtype=float)
    if len(commands) != len(reference_states):
        raise ValueError("commands and reference_states must have the same length")

    state = np.asarray(initial_state, dtype=float)
    errors = []
    for command, reference in zip(commands, reference_states):
        state = a @ state + b @ command
        errors.append(state - reference)
    return float(np.sqrt(np.mean(np.asarray(errors) ** 2)))


def split_by_premise_vertex(premises, bounds):
    """Assign each sample to the premise vertex it is closest to.

    Returns a list of index arrays, one per rule, using the same rule ordering
    as ``ts_model.rule_memberships``.
    """
    premises = np.asarray(premises, dtype=float)
    if premises.ndim != 2:
        raise ValueError(f"expected a 2-D (N, n_premises) array, got {premises.shape}")
    if premises.shape[1] != len(bounds):
        raise ValueError(
            f"got {premises.shape[1]} premise columns but {len(bounds)} bounds")

    bits = np.zeros(len(premises), dtype=int)
    for position, (lower, upper) in enumerate(bounds):
        midpoint = 0.5 * (lower + upper)
        is_high = (premises[:, position] >= midpoint).astype(int)
        bits |= is_high << (len(bounds) - 1 - position)

    return [np.flatnonzero(bits == rule) for rule in range(2 ** len(bounds))]
