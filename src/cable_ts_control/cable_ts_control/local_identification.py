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

from cable_ts_control.ts_model import rule_memberships


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


def fit_fuzzy_model(states, commands, next_states, memberships, ridge=0.0,
                    affine=False):
    """Jointly fit every rule of the blended model.

    The model that runs online is

        x(k+1) = sum_i h_i(rho_k) ( A_i x_k + B_i u_k )

    so that is what must be fitted. Fitting each rule separately on a crisp
    cell assignment fits a different model from the one that executes: the
    vertices never have to agree in the overlap regions, which is exactly where
    the blended dynamics live and where the cross-rule LMIs are evaluated.

    The equation is linear in the entries of every ``A_i, B_i``, so all rules
    are estimated in one least-squares problem with regressor rows

        [ h_1 x' , h_1 u' , ... , h_r x' , h_r u' ].

    ``affine`` adds a per-rule offset, which absorbs the static component
    instead of letting it bias the dynamic matrices. Returns
    ``(a_vertices, b_vertices, offsets)``; ``offsets`` is ``None`` unless
    ``affine`` is set.

    The regressor is RANK DEFICIENT by construction: complementary triangular
    memberships are exactly affine in the premises, so wherever the premises
    are state components the ``h_i x`` blocks are linearly dependent. The
    blended prediction is still unique; the individual vertex matrices are not.
    The minimum-norm solution is therefore the right pick — it is well defined,
    stable, and does not let an arbitrary null-space component inflate the
    matrices that are later handed to the LMIs.
    """
    states = np.asarray(states, dtype=float)
    commands = np.asarray(commands, dtype=float)
    next_states = np.asarray(next_states, dtype=float)
    memberships = np.asarray(memberships, dtype=float)

    if states.ndim != 2 or commands.ndim != 2 or next_states.ndim != 2:
        raise ValueError("states, commands and next_states must all be 2-D")
    if memberships.ndim != 2:
        raise ValueError(f"memberships must be (N, n_rules), got {memberships.shape}")
    if not (len(states) == len(commands) == len(next_states) == len(memberships)):
        raise ValueError("sample count mismatch between inputs and memberships")

    n, m = states.shape[1], commands.shape[1]
    n_rules = memberships.shape[1]
    block = n + m + (1 if affine else 0)
    if len(states) < n_rules * block:
        raise ValueError(
            f"need at least {n_rules * block} samples to fit {n_rules} rules, "
            f"got {len(states)}")

    columns = [states, commands] + ([np.ones((len(states), 1))] if affine else [])
    features = np.hstack(columns)
    regressor = (memberships[:, :, None] * features[:, None, :]).reshape(
        len(states), n_rules * block)

    if ridge > 0.0:
        # Tikhonov as an augmented least-squares system rather than a normal
        # equation: forming the Gram squares an already huge condition number.
        width = regressor.shape[1]
        regressor = np.vstack([regressor, np.sqrt(ridge) * np.eye(width)])
        targets = np.vstack([next_states, np.zeros((width, next_states.shape[1]))])
    else:
        targets = next_states
    theta, *_ = np.linalg.lstsq(regressor, targets, rcond=None)

    a_vertices, b_vertices, offsets = [], [], ([] if affine else None)
    for rule in range(n_rules):
        chunk = theta[rule * block:(rule + 1) * block]
        a_vertices.append(chunk[:n].T)
        b_vertices.append(chunk[n:n + m].T)
        if affine:
            offsets.append(chunk[n + m])
    return a_vertices, b_vertices, offsets


def fit_structured_fuzzy_model(states, commands, next_states, memberships, dt,
                               n_modes, ridge=0.0, enforce_passive=True,
                               floor_ratio=0.05, coherence=0.0):
    """Second-order structured fit of the reduced cable.

    An unstructured fit has to discover from data that a cable is a mechanical
    system. It never quite does: the position and boundary rows come out only
    approximately right, the rank-deficient regressor spreads the rest
    arbitrarily across rules, and neighbouring rules end up with spectral radii
    from 1.0 to 1.6 — far too different for the rules to share one Lyapunov
    function. Imposing the structure instead leaves only the physics to fit.

    With ``x = [q, qdot, g]`` and ``u`` the gripper velocity:

        qdot' = -K q - D qdot + H g + G u
        q'    = qdot                       (exact)
        g'    = u                          (exact)

    Only ``K, D, H, G`` are estimated, per rule, with the same fuzzy weighting
    as the blended model. ``enforce_passive`` projects the symmetric parts of
    ``K`` and ``D`` onto the positive semidefinite cone, which is what makes the
    continuous model dissipative and the discretisation Schur-stable.

    The continuous model is discretised exactly, not by forward Euler: at a
    40 ms period an Euler step of a stiff rod is not a stability guarantee.

    ``coherence`` penalises each rule's deviation from the mean rule. The
    plant is smooth, so local models at neighbouring corners of the premise box
    must be close; without the penalty the fit lets them drift far apart, and a
    single Lyapunov matrix then cannot cover all of them even though each rule
    is individually stabilisable.

    Returns ``(a_vertices, b_vertices)`` already in discrete time.
    """
    from scipy.linalg import expm

    states = np.asarray(states, dtype=float)
    commands = np.asarray(commands, dtype=float)
    next_states = np.asarray(next_states, dtype=float)
    memberships = np.asarray(memberships, dtype=float)

    n_state, n_input = states.shape[1], commands.shape[1]
    n_boundary = n_state - 2 * n_modes
    if n_boundary < 0:
        raise ValueError(
            f"state of {n_state} is too small for {n_modes} modes")
    n_rules = memberships.shape[1]

    # Only the velocity row is unknown; regress its continuous derivative.
    acceleration = (next_states[:, n_modes:2 * n_modes]
                    - states[:, n_modes:2 * n_modes]) / dt
    features = states
    block = n_state + n_input
    regressor = (memberships[:, :, None]
                 * np.hstack([features, commands])[:, None, :]).reshape(
        len(states), n_rules * block)

    if ridge > 0.0:
        width = regressor.shape[1]
        regressor = np.vstack([regressor, np.sqrt(ridge) * np.eye(width)])
        acceleration = np.vstack(
            [acceleration, np.zeros((width, acceleration.shape[1]))])
    if coherence > 0.0:
        # Shrink towards the mean rule, not towards zero.
        centring = np.kron(np.eye(n_rules) - np.ones((n_rules, n_rules)) / n_rules,
                           np.eye(block))
        regressor = np.vstack([regressor, np.sqrt(coherence) * centring])
        acceleration = np.vstack(
            [acceleration, np.zeros((len(centring), acceleration.shape[1]))])
    theta, *_ = np.linalg.lstsq(regressor, acceleration, rcond=None)

    a_vertices, b_vertices = [], []
    for rule in range(n_rules):
        chunk = theta[rule * block:(rule + 1) * block]
        stiffness = -chunk[:n_modes].T
        damping = -chunk[n_modes:2 * n_modes].T
        boundary = chunk[2 * n_modes:n_state].T
        gain = chunk[n_state:].T

        if enforce_passive:
            stiffness = _project_psd(stiffness, floor_ratio)
            damping = _project_psd(damping, floor_ratio)

        a_c = np.zeros((n_state, n_state))
        a_c[:n_modes, n_modes:2 * n_modes] = np.eye(n_modes)
        a_c[n_modes:2 * n_modes, :n_modes] = -stiffness
        a_c[n_modes:2 * n_modes, n_modes:2 * n_modes] = -damping
        if n_boundary:
            a_c[n_modes:2 * n_modes, 2 * n_modes:] = boundary

        b_c = np.zeros((n_state, n_input))
        b_c[n_modes:2 * n_modes] = gain
        if n_boundary:
            b_c[2 * n_modes:] = np.eye(n_boundary, n_input)

        joint = np.zeros((n_state + n_input, n_state + n_input))
        joint[:n_state, :n_state] = a_c
        joint[:n_state, n_state:] = b_c
        discrete = expm(joint * dt)
        a_vertices.append(discrete[:n_state, :n_state])
        b_vertices.append(discrete[:n_state, n_state:])
    return a_vertices, b_vertices


def _project_psd(matrix, floor_ratio=0.0):
    """Nearest symmetric matrix whose eigenvalues are at least a small floor.

    Clipping to exactly zero is too blunt: a noisy fit that comes out slightly
    indefinite then loses its damping altogether, and the mode becomes a pure
    integrator that is barely stabilisable. A cable always has some stiffness
    and some damping, so the floor is a fraction of the largest eigenvalue.
    """
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    floor = floor_ratio * max(float(values.max()), 0.0)
    return vectors @ np.diag(np.clip(values, floor, None)) @ vectors.T


def blended_predict(a_vertices, b_vertices, premise_bounds, states, commands,
                    offsets=None):
    """One-step predictions of the full fuzzy model, from true states."""
    n_premises = len(premise_bounds)
    states = np.asarray(states, dtype=float)
    commands = np.asarray(commands, dtype=float)

    predicted = np.zeros_like(states)
    for step, (state, command) in enumerate(zip(states, commands)):
        weights = rule_memberships(state[:n_premises], premise_bounds)
        a = sum(w * m for w, m in zip(weights, a_vertices))
        b = sum(w * m for w, m in zip(weights, b_vertices))
        value = a @ state + b @ command
        if offsets is not None:
            value = value + sum(w * c for w, c in zip(weights, offsets))
        predicted[step] = value
    return predicted


def blended_rollout(a_vertices, b_vertices, premise_bounds, initial_state,
                    commands, offsets=None):
    """Free-running trajectory of the FULL fuzzy model.

    The memberships are recomputed from the predicted premise at every step,
    which is what the plant and controller actually do. Freezing one local
    model for the whole horizon measures something the model never claims.
    """
    n_premises = len(premise_bounds)
    state = np.asarray(initial_state, dtype=float).copy()
    commands = np.asarray(commands, dtype=float)

    predicted = np.zeros((len(commands), state.size))
    for step, command in enumerate(commands):
        weights = rule_memberships(state[:n_premises], premise_bounds)
        a = sum(w * m for w, m in zip(weights, a_vertices))
        b = sum(w * m for w, m in zip(weights, b_vertices))
        state = a @ state + b @ command
        if offsets is not None:
            state = state + sum(w * c for w, c in zip(weights, offsets))
        predicted[step] = state
    return predicted


def blended_rollout_rmse(a_vertices, b_vertices, premise_bounds, initial_state,
                         commands, reference_states, offsets=None,
                         position_dim=None):
    """RMS error of a full fuzzy rollout against a contiguous reference.

    ``reference_states`` must be the real successors of ``commands`` in their
    original temporal order; scattered validation samples are not a trajectory.
    """
    reference_states = np.asarray(reference_states, dtype=float)
    if len(commands) != len(reference_states):
        raise ValueError("commands and reference_states must have the same length")

    predicted = blended_rollout(a_vertices, b_vertices, premise_bounds,
                                initial_state, commands, offsets)
    residual = predicted - reference_states
    if position_dim is not None:
        residual = residual[:, :position_dim]
    return float(np.sqrt(np.mean(residual ** 2)))


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
