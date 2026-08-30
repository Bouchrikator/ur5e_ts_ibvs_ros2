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
    blended_predict,
    blended_rollout_rmse,
    fit_fuzzy_model,
    fit_structured_fuzzy_model,
    split_by_premise_vertex,
)
from cable_ts_control.modal_basis import ModalBasis
from cable_ts_control.state_filter import velocity_series
from cable_ts_control.ts_model import CableTsModel, rule_memberships


def build_state_trajectory(shapes, basis, dt, alpha, gripper=None,
                           gripper_reference=None):
    """Reduced state of one contiguous trajectory.

    ``x = [q; qdot]``, or ``[q; qdot; p_g - p_ref]`` when the gripper is given.

    The gripper position belongs in the state: the input is gripper VELOCITY,
    so a displaced but stationary gripper keeps the cable deformed while
    contributing nothing to the ``[q, qdot]`` dynamics. Without it the reduced
    state is not Markov, and a free-running rollout has no way to know where
    the boundary actually is.

    The velocity uses the same causal filter as the runtime state reducer.
    Fitting on a centred ``np.gradient`` and executing on a causal low-pass
    means the model was identified for a state the controller never sees.
    """
    q = np.array([basis.project(shape) for shape in shapes])
    blocks = [q, velocity_series(q, dt, alpha)]
    if gripper is not None:
        reference = np.zeros(2) if gripper_reference is None else gripper_reference
        blocks.append(np.asarray(gripper, dtype=float) - reference)
    return np.hstack(blocks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--basis", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--premise-margin", type=float, default=1.0,
                        help="scales the premise range around the observed data")
    parser.add_argument("--n-premises", type=int, default=2,
                        help="leading modal coordinates used as TS premises; "
                             "rules = 2**n_premises, so this is what controls "
                             "the rule count, not the number of modes")
    parser.add_argument("--velocity-alpha", type=float, default=0.4,
                        help="must match the runtime state reducer")
    parser.add_argument("--no-gripper-state", action="store_true",
                        help="drop the gripper position from the state; the "
                             "reduced model is then not Markov and free "
                             "rollouts diverge")
    parser.add_argument("--n-validation-trajectories", type=int, default=2,
                        help="trajectories held out whole; a rollout over "
                             "scattered samples is not a trajectory")
    parser.add_argument("--max-one-step-rmse", type=float, default=5e-3,
                        help="acceptance threshold on the held-out one-step error")
    parser.add_argument("--max-rollout-rmse", type=float, default=0.05,
                        help="acceptance threshold on the held-out free-running "
                             "error of the FULL fuzzy model")
    parser.add_argument("--rollout-horizon", type=int, default=50,
                        help="steps used for the free-running check")
    parser.add_argument("--affine", action="store_true",
                        help="fit a per-rule offset so the static component is "
                             "not absorbed into A and B")
    parser.add_argument("--unstructured", action="store_true",
                        help="fit A and B freely instead of imposing the "
                             "second-order mechanical structure; the position "
                             "and boundary rows are then only approximately "
                             "right and the rules drift apart")
    parser.add_argument("--allow-active-damping", action="store_true",
                        help="skip the PSD projection of the fitted stiffness "
                             "and damping")
    parser.add_argument("--ridge", type=float, default=1e-8)
    args = parser.parse_args(argv)

    data = np.load(args.dataset, allow_pickle=False)
    basis = ModalBasis.load(args.basis)
    dt = float(data["timestep_s"][0])
    if "trajectory_ids" not in data.files:
        raise SystemExit(
            "dataset has no trajectory_ids: regenerate it with the current "
            "generate_sofa_dataset. Splitting scattered samples makes a "
            "contiguous rollout impossible to evaluate.")

    vertices = [int(v) for v in np.unique(data["vertex_ids"])]
    trajectories = [int(t) for t in np.unique(data["trajectory_ids"])]
    if len(trajectories) < 2:
        raise SystemExit("need at least two trajectories to hold one out")
    n_validation = min(max(1, args.n_validation_trajectories),
                       len(trajectories) - 1)
    validation_ids = trajectories[-n_validation:]
    training_ids = [t for t in trajectories if t not in validation_ids]

    # Contiguous segments, keyed by (parameter vertex, trajectory).
    use_gripper = (not args.no_gripper_state) and "gripper" in data.files
    # Centre the boundary state: with no reference the dynamics are forced
    # through the global origin and the static offset lands in A and B.
    gripper_reference = data["gripper"].mean(axis=0) if use_gripper else None
    segments = {}
    for vertex in vertices:
        for traj in trajectories:
            selection = ((data["vertex_ids"] == vertex)
                         & (data["trajectory_ids"] == traj))
            if not selection.any():
                continue
            trajectory = build_state_trajectory(
                data["shapes"][selection], basis, dt, args.velocity_alpha,
                data["gripper"][selection] if use_gripper else None,
                gripper_reference)
            command = data["commands"][selection]
            segments[(vertex, traj)] = (
                trajectory[:-1], command[:-1], trajectory[1:])

    n_modes = basis.n_modes
    # Premises are a design choice, not the state dimension. Keeping more modes
    # in the state improves the prediction, but making every mode a premise
    # would give 2**n_modes rules and starve each cell of data. The plan uses
    # rho = (q1, q2), i.e. two premises and four rules, whatever the state is.
    n_premises = min(int(args.n_premises), n_modes)
    n_rules = 2 ** n_premises

    # Premise box from TRAINING trajectories only, so the validation shapes do
    # not leak into the partition the model is defined on.
    training = {v: np.concatenate([segments[(v, t)][0] for t in training_ids])
                for v in vertices}
    blocks = [training[v][:, :n_premises] for v in vertices]
    split = np.mean([np.median(block, axis=0) for block in blocks], axis=0)
    half = np.min([np.minimum(split - block.min(axis=0),
                              block.max(axis=0) - split) for block in blocks],
                  axis=0)
    if np.any(half <= 0.0):
        raise SystemExit(
            "no premise split is straddled by every parameter vertex; the "
            "excitation must drive all vertices over a common shape range")
    half = half * args.premise_margin
    bounds = [(float(split[i] - half[i]), float(split[i] + half[i]))
              for i in range(n_premises)]

    print(f"{n_modes} modes, {n_premises} premises -> {n_rules} rules")
    layout = "[q, qdot, gripper]" if use_gripper else "[q, qdot]"
    print(f"state = {layout} ({training[vertices[0]].shape[1]} dims)")
    print(f"sample period {dt * 1e3:.3f} ms ({1.0 / dt:.2f} Hz), "
          f"velocity alpha {args.velocity_alpha}")
    print(f"trajectories: train {training_ids}, validation {validation_ids}")
    print(f"premise bounds: {bounds}")

    parameter_names = [str(n) for n in data["parameter_names"]]
    parameter_values = data["parameter_values"]

    a_sets, b_sets = [], []
    worst_one_step, worst_rollout, worst_rho = 0.0, 0.0, 0.0
    for vertex in vertices:
        label = dict(zip(parameter_names, parameter_values[vertex]))
        print(f"\nparameter vertex {vertex}: {label}")

        states = training[vertex]
        commands = np.concatenate([segments[(vertex, t)][1] for t in training_ids])
        next_states = np.concatenate(
            [segments[(vertex, t)][2] for t in training_ids])

        # Coverage is still reported per crisp cell: a rule with no data nearby
        # is unconstrained by the joint fit even though the fit itself blends.
        cells = split_by_premise_vertex(states[:, :n_premises], bounds)
        counts = [len(c) for c in cells]
        minimum = states.shape[1] + commands.shape[1] + 4
        if min(counts) < minimum:
            raise SystemExit(
                f"rule {int(np.argmin(counts))} of parameter vertex {vertex} "
                f"has only {min(counts)} training samples (need {minimum}); "
                f"excite the cable over a wider shape range")

        memberships = np.array([rule_memberships(x[:n_premises], bounds)
                                for x in states])
        if args.unstructured:
            a_vertices, b_vertices, offsets = fit_fuzzy_model(
                states, commands, next_states, memberships,
                ridge=args.ridge, affine=args.affine)
        else:
            a_vertices, b_vertices = fit_structured_fuzzy_model(
                states, commands, next_states, memberships, dt, n_modes,
                ridge=args.ridge,
                enforce_passive=not args.allow_active_damping)
            offsets = None

        rho = max(float(np.max(np.abs(np.linalg.eigvals(a))))
                  for a in a_vertices)
        worst_rho = max(worst_rho, rho)

        # Validation on whole held-out trajectories, in their original order.
        one_step, rollouts = [], []
        for traj in validation_ids:
            x, u, x_next = segments[(vertex, traj)]
            predicted = blended_predict(a_vertices, b_vertices, bounds, x, u,
                                        offsets)
            one_step.append(float(np.sqrt(np.mean(
                (predicted[:, :n_modes] - x_next[:, :n_modes]) ** 2))))
            horizon = min(args.rollout_horizon, len(u))
            rollouts.append(blended_rollout_rmse(
                a_vertices, b_vertices, bounds, x[0], u[:horizon],
                x_next[:horizon], offsets, position_dim=n_modes))

        vertex_one_step, vertex_rollout = max(one_step), max(rollouts)
        worst_one_step = max(worst_one_step, vertex_one_step)
        worst_rollout = max(worst_rollout, vertex_rollout)
        print(f"  cells {counts}, {len(states)} training samples")
        print(f"  held-out one-step {vertex_one_step * 1e3:.3f} mm, "
              f"blended rollout {vertex_rollout * 1e3:.4g} mm, "
              f"max rho(A) {rho:.4f}")

        a_sets.append(a_vertices)
        b_sets.append(b_vertices)

    # rho(A) is reported, not gated: PDC certifies the CLOSED loop A_i - B_i K_j,
    # so an open-loop unstable but stabilisable vertex is perfectly admissible.
    print(f"\nworst held-out one-step rmse: {worst_one_step * 1e3:.4f} mm")
    print(f"worst blended rollout rmse:   {worst_rollout * 1e3:.4f} mm")
    print(f"worst open-loop rho(A):       {worst_rho:.4f} (diagnostic only)")

    if worst_one_step > args.max_one_step_rmse:
        print(f"\nERROR: worst held-out one-step rmse "
              f"{worst_one_step * 1e3:.3f} mm exceeds "
              f"{args.max_one_step_rmse * 1e3:.3f} mm")
        return 1
    if worst_rollout > args.max_rollout_rmse:
        print(f"\nERROR: worst blended rollout rmse {worst_rollout * 1e3:.4g} mm "
              f"exceeds {args.max_rollout_rmse * 1e3:.3f} mm")
        return 1

    # The controller carries a single model; the extra parameter vertices are
    # kept for the LMI robustness check and stored alongside it.
    model = CableTsModel(
        a_sets[0], b_sets[0], bounds, dt,
        parameter_bounds={
            "names": parameter_names,
            "values": parameter_values.tolist(),
            "A": [[a.flatten().tolist() for a in group] for group in a_sets],
            "B": [[b.flatten().tolist() for b in group] for group in b_sets],
        })
    model.save(args.output)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
