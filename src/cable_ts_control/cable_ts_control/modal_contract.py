"""Contract of the Cosserat-modal state shared by the SOFA phases and the ROS nodes.

Pure numerics and dictionary checks: no SOFA, no ROS, so every refusal is unit
tested. The state is ``x = [a, a_dot, p_g - p_g0]`` with ``a`` the coordinates
of the ONE strain POD; the legacy marker POD (``ModalBasis``) is never involved.
"""

import numpy as np

COORDINATES = "cosserat_modal"
LEGACY_COORDINATES = "marker_modal"


def build_modal_state(modal_positions, modal_velocities, gripper, gripper_reference):
    """``x = [a, a_dot, p_g - p_g0]`` for one sample or a whole trajectory."""
    a = np.asarray(modal_positions, dtype=float)
    v = np.asarray(modal_velocities, dtype=float)
    g = np.asarray(gripper, dtype=float) - np.asarray(gripper_reference, dtype=float)
    if a.shape != v.shape or g.shape[-1] != 2 or a.shape[:-1] != g.shape[:-1]:
        raise ValueError(f"inconsistent modal state blocks {a.shape}, {v.shape}, {g.shape}")
    return np.concatenate([a, v, g], axis=-1)


def check_dataset_contract(dataset, basis_metadata):
    """Refuse datasets that carry the legacy marker coordinates or another basis."""
    keys = set(dataset.keys()) if hasattr(dataset, "keys") else set(dataset.files)
    required = {"modal_positions", "modal_velocities_state", "gripper", "commands",
                "state_coordinates", "strain_basis_sha256"}
    missing = sorted(required - keys)
    if missing:
        raise ValueError(f"dataset lacks the Cosserat-modal fields {missing}; "
                         "regenerate it with the SOFA ROM (cable_sofa_modal_dataset)")
    if str(dataset["state_coordinates"]) != COORDINATES:
        raise ValueError(f"dataset coordinates {dataset['state_coordinates']!s} != {COORDINATES}")
    if str(dataset["strain_basis_sha256"]) != basis_metadata["strain_basis_sha256"]:
        raise ValueError("dataset was generated with another strain basis")
    if int(np.asarray(dataset["modal_positions"]).shape[1]) != int(basis_metadata["n_modes"]):
        raise ValueError("dataset modal dimension differs from the basis")


def check_model_contract(model, basis_metadata=None, gains=None, expected_coordinates=None):
    """Coordinates, order, ``state_dim = 2r + 2`` and basis hash must all agree.

    ``model``/``gains`` are the ``cable_ts_model`` sections of their YAML files.
    Returns the coordinate type of the model.
    """
    coordinates = model.get("state_coordinates", LEGACY_COORDINATES)
    if expected_coordinates is not None and coordinates != expected_coordinates:
        raise ValueError(f"model coordinates {coordinates!r}, expected {expected_coordinates!r}")
    if coordinates == COORDINATES:
        n_modes = int(model["n_modes"])
        if int(model["state_dim"]) != 2 * n_modes + 2:
            raise ValueError(f"state_dim {model['state_dim']} != 2*{n_modes}+2")
        if not model.get("strain_basis_sha256"):
            raise ValueError("a Cosserat-modal model must name its strain basis hash")
        if basis_metadata is not None:
            if basis_metadata.get("state_coordinates") != COORDINATES:
                raise ValueError("basis metadata is not a Cosserat strain basis")
            if int(basis_metadata["n_modes"]) != n_modes:
                raise ValueError(f"basis has {basis_metadata['n_modes']} modes, model {n_modes}")
            if basis_metadata["strain_basis_sha256"] != model["strain_basis_sha256"]:
                raise ValueError("model and strain basis hashes differ")
    if gains is not None:
        for key in ("state_coordinates", "strain_basis_sha256", "n_modes"):
            if gains.get(key, model.get(key)) != model.get(key):
                raise ValueError(f"gains {key} {gains.get(key)!r} != model {model.get(key)!r}")
        if "ts_model_sha256" in gains and "sha256" in model and gains["ts_model_sha256"] != model["sha256"]:
            raise ValueError("gains were synthesised for another TS model")
    return coordinates


def target_state(target, n_modes, state_dim):
    """``[a*, 0, g*]`` from a target given as ``a*``, ``[a*, g*]`` or the full state."""
    target = np.asarray(target, dtype=float).ravel()
    full = np.zeros(state_dim)
    if target.size == n_modes:
        full[:n_modes] = target
    elif target.size == n_modes + 2:
        full[:n_modes] = target[:n_modes]
        full[2 * n_modes:] = target[n_modes:]
    elif target.size == state_dim:
        full[:] = target
    else:
        raise ValueError(f"target of {target.size} entries fits neither a*, [a*, g*] nor the "
                         f"{state_dim}-state")
    return full
