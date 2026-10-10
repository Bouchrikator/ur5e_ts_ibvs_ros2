"""Time-step selection control flow of the FOM qualification; no SOFA runtime needed."""

import numpy as np

from cable_identification import fom_verification as fv


def check_dynamics_with(outcomes, production_step):
    steps = [4.0, 2.0, 1.0]
    selected, attempts = fv.select_timestep(steps, lambda h, ref: outcomes[(h, ref)])
    c = {"rms": 1e-5, "max": 2e-5, "length": 1e-6, "force": np.array([0.5]), "moment": np.array([0.2]),
         "F": np.zeros((1, 3)), "F_ref": np.zeros((1, 3))}
    measured = {h: (c, [-2e-4, -1e-4], [np.array([2e-4, 2e-5, 1e-4]), np.array([1e-4, 1e-5, 5e-5])],
                    [(True, "within"), (True, "within")]) for h in steps}
    original, fv.qualify_timestep = fv.qualify_timestep, lambda cfg: (selected, attempts, measured)
    try:
        results = []
        fv.check_dynamics({"timestep_s": production_step, "convective_inertia": True}, results)
    finally:
        fv.qualify_timestep = original
    return selected, attempts, results


def test_rejected_candidate_does_not_fail_a_later_selection():
    selected, attempts, results = check_dynamics_with(
        {(4.0, 2.0): (True, ["reaction force 7.44 of its bound"]), (2.0, 1.0): (True, [])}, 2.0)
    assert selected == 2.0 and [a[:2] for a in attempts] == [(4.0, 2.0), (2.0, 1.0)]
    assert results and all(ok for _, ok, _ in results)


def test_no_qualified_candidate_fails_the_selection():
    selected, attempts, results = check_dynamics_with(
        {(4.0, 2.0): (True, ["budget"]), (2.0, 1.0): (True, ["budget"])}, 2.0)
    assert selected is None and len(attempts) == 2
    assert results and not any(ok for _, ok, _ in results)
