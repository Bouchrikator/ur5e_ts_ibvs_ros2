#!/usr/bin/env python3
"""Identify the TS vertex matrices of the reduced cable.

Step 7 of the plan. Projects the SOFA dataset onto the modal basis, splits it
by premise vertex, and fits a local ``(A_i, B_i)`` per rule and per physical
parameter vertex. Models are validated on a held-out part of the data so a fit
that only memorises its training trajectory is rejected.

    ros2 run cable_ts_control identify_ts_vertices \
        --dataset /tmp/cable_dataset.npz --basis cable_modal_basis.yaml \
        --output cable_ts_model.yaml
"""

import argparse
import sys

import numpy as np

from cable_ts_control.local_identification import (
    fit_local_model,
    one_step_rmse,
    rollout_rmse,
    split_by_premise_vertex,
)
from cable_ts_control.modal_basis import ModalBasis
from cable_ts_control.ts_model import CableTsModel


def build_state_trajectory(shapes, basis, dt):
    """Modal state ``x = [q; qdot]`` of a shape trajectory."""
    q = np.array([basis.project(shape) for shape in shapes])
    velocity = np.gradient(q, dt, axis=0)
    return np.hstack([q, velocity])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--basis", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--control-rate-hz", type=float, default=30.0,
                        help="controller rate; the model is identified at this rate")
    parser.add_argument("--premise-margin", type=float, default=1.0,
                        help="scales the premise range around the observed data")
    parser.add_argument("--n-premises", type=int, default=2,
                        help="leading modal coordinates used as TS premises; "
                             "rules = 2**n_premises, so this is what controls "
                             "the rule count, not the number of modes")
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--max-one-step-rmse", type=float, default=5e-3,
                        help="acceptance threshold on the held-out one-step error")
    parser.add_argument("--max-rollout-rmse", type=float, default=0.05,
                        help="acceptance threshold on the held-out free-running "
                             "error; one-step error alone is nearly blind to "
                             "eigenvalue error and passes divergent models")
    parser.add_argument("--rollout-horizon", type=int, default=30,
                        help="steps used for the free-running check")
    parser.add_argument("--max-spectral-radius", type=float, default=1.02,
                        help="a damped cable is dissipative, so rho(A) must not "
                             "exceed 1 by more than the fitting tolerance")
    parser.add_argument("--ridge", type=float, default=1e-8)
    args = parser.parse_args(argv)

    data = np.load(args.dataset, allow_pickle=False)
    basis = ModalBasis.load(args.basis)
    sofa_dt = float(data["timestep_s"][0])
    control_dt = 1.0 / args.control_rate_hz
    decimation = max(1, int(round(control_dt / sofa_dt)))

    states, commands, next_states, vertex_ids = [], [], [], []
    for vertex in np.unique(data["vertex_ids"]):
        selection = data["vertex_ids"] == vertex
        trajectory = build_state_trajectory(
            data["shapes"][selection], basis, sofa_dt)
        # Decimate to the controller rate: the TS model must be discrete at the
        # rate it will actually run at.
        trajectory = trajectory[::decimation]
        command = data["commands"][selection][::decimation]
        states.append(trajectory[:-1])
        next_states.append(trajectory[1:])
        commands.append(command[:-1])
        vertex_ids.append(np.full(len(trajectory) - 1, vertex))

    states = np.concatenate(states)
    commands = np.concatenate(commands)
    next_states = np.concatenate(next_states)
    vertex_ids = np.concatenate(vertex_ids)

    n_modes = basis.n_modes
    # Premises are a design choice, not the state dimension. Keeping more modes
    # in the state improves the prediction, but making every mode a premise
    # would give 2**n_modes rules and starve each cell of data. The plan uses
    # rho = (q1, q2), i.e. two premises and four rules, whatever the state is.
    n_premises = min(int(args.n_premises), n_modes)
    premises = states[:, :n_premises]
    span = args.premise_margin
    # Rules are split at the midpoint of the premise box, so that midpoint must
    # fall strictly inside EVERY parameter vertex's data: otherwise a vertex
    # sits entirely on one side and the rules on the other side have nothing to
    # fit. Split at the mean of the per-vertex medians, then grow the box only
    # as far as the most restrictive vertex allows.
    blocks = [premises[vertex_ids == v] for v in np.unique(vertex_ids)]
    split = np.mean([np.median(block, axis=0) for block in blocks], axis=0)
    half = np.min([np.minimum(split - block.min(axis=0),
                              block.max(axis=0) - split) for block in blocks],
                  axis=0)
    if np.any(half <= 0.0):
        raise SystemExit(
            "no premise split is straddled by every parameter vertex; the "
            "excitation must drive all vertices over a common shape range")
    half = half * span
    bounds = [(float(split[i] - half[i]), float(split[i] + half[i]))
              for i in range(n_premises)]
    print(f"{n_modes} modes, {n_premises} premises -> {2 ** n_premises} rules")
    print(f"premise split (straddled by all vertices): {split.tolist()}")
    print(f"premise bounds: {bounds}")
    print(f"samples: {len(states)} at {args.control_rate_hz:.0f} Hz "
          f"(decimation {decimation})")

    rng = np.random.default_rng(0)
    is_validation = rng.random(len(states)) < args.validation_fraction

    parameter_names = [str(n) for n in data["parameter_names"]]
    parameter_values = data["parameter_values"]

    a_sets, b_sets, worst_rmse = [], [], 0.0
    worst_rollout, worst_rho = 0.0, 0.0
    for vertex in np.unique(vertex_ids):
        in_vertex = vertex_ids == vertex
        rules = split_by_premise_vertex(premises[in_vertex], bounds)
        a_vertices, b_vertices = [], []
        label = dict(zip(parameter_names, parameter_values[int(vertex)]))
        print(f"\nparameter vertex {int(vertex)}: {label}")

        indices = np.flatnonzero(in_vertex)
        for rule, local in enumerate(rules):
            selected = indices[local]
            train = selected[~is_validation[selected]]
            test = selected[is_validation[selected]]
            if len(train) < 2 * n_modes + 4:
                raise SystemExit(
                    f"rule {rule} of parameter vertex {int(vertex)} has only "
                    f"{len(train)} training samples; excite the cable over a "
                    f"wider shape range")

            a, b = fit_local_model(states[train], commands[train],
                                   next_states[train], ridge=args.ridge)
            rmse = one_step_rmse(a, b, states[test], commands[test],
                                 next_states[test],
                                 position_dim=n_modes) if len(test) else float("nan")
            rho = float(max(abs(np.linalg.eigvals(a))))
            horizon = min(args.rollout_horizon, len(test))
            rollout = rollout_rmse(
                a, b, states[test][0], commands[test][:horizon],
                next_states[test][:horizon]) if horizon > 1 else float("nan")

            worst_rmse = max(worst_rmse, 0.0 if np.isnan(rmse) else rmse)
            worst_rho = max(worst_rho, rho)
            if not np.isnan(rollout):
                worst_rollout = max(worst_rollout, rollout)
            print(f"  rule {rule}: {len(train)} train / {len(test)} test, "
                  f"one-step {rmse * 1e3:.3f} mm, rollout {rollout * 1e3:.4g} mm, "
                  f"rho(A) = {rho:.4f}")
            a_vertices.append(a)
            b_vertices.append(b)

        a_sets.append(a_vertices)
        b_sets.append(b_vertices)

    if worst_rmse > args.max_one_step_rmse:
        print(f"\nERROR: worst held-out one-step rmse {worst_rmse * 1e3:.3f} mm "
              f"exceeds {args.max_one_step_rmse * 1e3:.3f} mm")
        return 1

    # A small one-step error says almost nothing about the eigenvalues: a model
    # can predict the next sample to a fraction of a millimetre and still have
    # rho(A) ~ 4, which diverges in free running and makes the LMIs infeasible.
    if worst_rho > args.max_spectral_radius:
        print(f"\nERROR: worst rho(A) {worst_rho:.4f} exceeds "
              f"{args.max_spectral_radius:.4f}: a damped cable cannot be "
              f"non-dissipative, so the fit is unphysical and no common "
              f"Lyapunov function can certify it.")
        return 1
    if worst_rollout > args.max_rollout_rmse:
        print(f"\nERROR: worst held-out rollout rmse {worst_rollout * 1e3:.4g} mm "
              f"exceeds {args.max_rollout_rmse * 1e3:.3f} mm")
        return 1

    # The controller carries a single model; the extra parameter vertices are
    # kept for the LMI robustness check and stored alongside it.
    model = CableTsModel(
        a_sets[0], b_sets[0], bounds, control_dt,
        parameter_bounds={
            "names": parameter_names,
            "values": parameter_values.tolist(),
            "A": [[a.flatten().tolist() for a in group] for group in a_sets],
            "B": [[b.flatten().tolist() for b in group] for group in b_sets],
        })
    model.save(args.output)
    print(f"\nworst held-out one-step rmse: {worst_rmse * 1e3:.4f} mm")
    print(f"worst held-out rollout rmse:  {worst_rollout * 1e3:.4f} mm")
    print(f"worst rho(A):                 {worst_rho:.4f}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
