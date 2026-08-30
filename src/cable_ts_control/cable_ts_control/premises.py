"""Premise variables of the Takagi-Sugeno cable model.

Tanaka & Wang build a TS model by *sector nonlinearity*: the premise variables
are the bounded nonlinearities of the plant, and the rule consequents are the
plant evaluated at the corners of the premise box. The blend is then an exact
representation of the model, not an approximation of it.

Which signal plays that role is a modelling decision, and it decides whether
the vertex matrices can be identified at all:

``modal``
    The leading modal coordinates, ``rho = (q1, q2)``. These are components of
    the state, so the fuzzy regressor contains the same monomial twice
    (``rho_1 * q_2`` and ``rho_2 * q_1`` are both ``q1 q2``) and is exactly
    rank deficient. The blended prediction is still unique but the individual
    vertex matrices are not: least squares returns one arbitrary point of a
    whole affine set, and their spread carries no physical meaning. Handing
    that spread to a common-Lyapunov LMI is what makes the synthesis
    infeasible.

``boundary``
    The boundary condition the gripper imposes on the rod, in polar form about
    the clamp: foreshortening ``1 - |p_g| / L`` and bearing ``atan2(p_g)``.
    For a clamped inextensible rod the transverse stiffness seen at the tip is
    governed by how far the chord falls short of the arc length, so this is
    where the geometric nonlinearity actually lives. Neither premise is a
    component of the state, so no monomial repeats and the vertex matrices are
    identifiable. The box is also bounded by construction: the excitation is
    confined to that annulus and angle range.

Pure numerics, shared by the identification scripts and the controller so the
premises are computed identically offline and online.
"""

import numpy as np


class ModalPremises:
    """Leading modal coordinates of the state."""

    kind = "modal"

    def __init__(self, n_premises):
        self.n_premises = int(n_premises)
        if self.n_premises < 1:
            raise ValueError("at least one premise is required")

    def __call__(self, state):
        state = np.asarray(state, dtype=float)
        return state[..., :self.n_premises]

    def to_dict(self):
        return {"kind": self.kind, "n_premises": self.n_premises}


class BoundaryPremises:
    """Foreshortening and bearing of the gripper about the clamped end.

    The state stores the gripper position centred on a reference, so the
    reference is added back before the polar form is taken.
    """

    kind = "boundary"
    n_premises = 2

    def __init__(self, gripper_index, cable_length_m, gripper_reference=(0.0, 0.0)):
        self.gripper_index = int(gripper_index)
        self.cable_length_m = float(cable_length_m)
        if self.cable_length_m <= 0.0:
            raise ValueError("cable_length_m must be > 0")
        self.gripper_reference = np.asarray(gripper_reference, dtype=float)
        if self.gripper_reference.shape != (2,):
            raise ValueError("gripper_reference must be a planar (x, y) pair")

    def __call__(self, state):
        state = np.asarray(state, dtype=float)
        i = self.gripper_index
        if state.shape[-1] < i + 2:
            raise ValueError(
                f"state of {state.shape[-1]} has no gripper block at index {i}")
        position = state[..., i:i + 2] + self.gripper_reference
        radius = np.linalg.norm(position, axis=-1)
        foreshortening = 1.0 - radius / self.cable_length_m
        bearing = np.arctan2(position[..., 1], position[..., 0])
        return np.stack([foreshortening, bearing], axis=-1)

    def to_dict(self):
        return {"kind": self.kind,
                "gripper_index": self.gripper_index,
                "cable_length_m": self.cable_length_m,
                "gripper_reference": self.gripper_reference.tolist()}


def premise_map_from_dict(data):
    """Rebuild a premise map from its serialised form."""
    if data is None:
        return None
    kind = data.get("kind")
    if kind == ModalPremises.kind:
        return ModalPremises(data["n_premises"])
    if kind == BoundaryPremises.kind:
        return BoundaryPremises(data["gripper_index"], data["cable_length_m"],
                                data.get("gripper_reference", (0.0, 0.0)))
    raise ValueError(f"unknown premise map {kind!r}")


def make_premise_map(kind, n_premises=2, gripper_index=None,
                     cable_length_m=None, gripper_reference=(0.0, 0.0)):
    """Premise map by name, as selected on the identification command line."""
    if kind == ModalPremises.kind:
        return ModalPremises(n_premises)
    if kind == BoundaryPremises.kind:
        if gripper_index is None or cable_length_m is None:
            raise ValueError(
                "boundary premises need the gripper index and the cable length")
        return BoundaryPremises(gripper_index, cable_length_m, gripper_reference)
    raise ValueError(f"unknown premise map {kind!r}")
