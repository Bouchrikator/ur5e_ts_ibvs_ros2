"""Validate SOFA state/mode files before passing them to native components.

The strain POD lives in the strain-energy inner product ``<q1, q2> = q1^T W q2``,
``W = diag(Sigma) l`` of the nominal cable (``cosserat_model.strain_weights``): the
official SOFA-MOR POD is run on the scaled snapshots ``W^1/2 q`` and its Euclidean-
orthonormal modes ``Phi_tilde`` are stored unscaled, ``Phi = W^-1/2 Phi_tilde``, so the
ROM mapping reads ``q = q0 + Phi a`` directly and ``Phi^T W Phi = I``. The matching
projection is ``a = Phi^T W (q - q0)`` (``project_strain``), never ``Phi^T q``.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from cable_identification import cosserat_model as cm

INNER_PRODUCT = "strain_energy"
MAPPING_TEMPLATE = "Vec1d->Vec6d"


def installed_plugin_patches(sofa_root=None):
    """Patch-set markers of the installed Cosserat and ModelOrderReduction builds
    (``{name: "<commit> <patches sha256>"}``); a basis is only compatible with the mechanics
    and mapping it was generated with."""
    import os

    root = Path(sofa_root or os.environ.get("SOFA_ROOT", "/opt/sofa")) / "plugins"
    markers = {"Cosserat": root / "Cosserat/.cable-cosserat-build",
               "ModelOrderReduction": root / "ModelOrderReduction/.cable-mor-build"}
    return {name: (path.read_text().strip() if path.is_file() else "unknown") for name, path in markers.items()}


@dataclass(frozen=True)
class ReductionSpec:
    modes_path: str
    metadata_path: str
    n_modes: int


def file_sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def config_sha256(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_state_file(path, full_dimension):
    records = []
    reference = None
    with open(path) as stream:
        for line in stream:
            key, separator, values = line.partition("=")
            key = key.strip()
            if not separator:
                if line.strip():
                    raise ValueError(f"Invalid WriteState line in {path}: {line.strip()}")
                continue
            vector = np.asarray(values.split(), dtype=float)
            if not np.all(np.isfinite(vector)):
                raise ValueError(f"Non-finite WriteState data in {path}")
            if key == "T":
                if vector.size != 1:
                    raise ValueError("WriteState time must be scalar")
                records.append({"time": float(vector[0])})
            elif key in ("X0", "X", "V"):
                if vector.size != full_dimension:
                    raise ValueError(f"{key} has {vector.size}, expected {full_dimension} scalars")
                if key == "X0":
                    if reference is not None and not np.array_equal(vector, reference):
                        raise ValueError("WriteState changed the physical neutral reference")
                    reference = vector
                else:
                    if not records or key in records[-1]:
                        raise ValueError("WriteState has duplicate or untimed state vectors")
                    records[-1][key] = vector
            else:
                raise ValueError(f"Unexpected WriteState field: {key}")
    if reference is None or not records or any("X" not in row or "V" not in row
                                              for row in records):
        raise ValueError("WriteState requires X0 and paired T/X/V records")
    times = np.asarray([row["time"] for row in records])
    if np.any(np.diff(times) <= 0):
        raise ValueError("WriteState times must increase strictly within an episode")
    return {"times": times, "reference": reference,
            "strain": np.stack([row["X"] for row in records]),
            "strain_velocity": np.stack([row["V"] for row in records])}


def read_modes(path):
    with open(path) as stream:
        header = stream.readline().split()
        if len(header) != 2:
            raise ValueError("MOR modes require a rows/columns header")
        rows, columns = map(int, header)
        lines = [line.split() for line in stream]
    if rows < 1 or columns < 1 or len(lines) != rows or any(
            len(line) != columns for line in lines):
        raise ValueError("MOR modes dimensions do not match the exact MatrixLoader format")
    modes = np.asarray(lines, dtype=float)
    if not np.all(np.isfinite(modes)):
        raise ValueError("MOR modes contain non-finite values")
    return modes


def write_modes(path, modes):
    """Exact MatrixLoader text format (``rows cols`` header), full precision."""
    modes = np.asarray(modes, dtype=float)
    np.savetxt(path, modes, header=f"{modes.shape[0]} {modes.shape[1]}", comments="", fmt="%.17g")


def scale_state_file(source, target, factors):
    """Copy a WriteState file with every ``X0=/X=/V=`` vector multiplied by ``factors``."""
    factors = np.asarray(factors, dtype=float)
    with open(source) as src, open(target, "w") as dst:
        for line in src:
            key, separator, values = line.partition("=")
            if separator and key.strip() in ("X0", "X", "V"):
                vector = np.asarray(values.split(), dtype=float)
                if vector.size != factors.size:
                    raise ValueError(f"{key.strip()} has {vector.size}, expected {factors.size} scalars")
                dst.write(f"{key}= " + " ".join(f"{v:.17g}" for v in vector * factors) + "\n")
            else:
                dst.write(line)


def project_strain(modes, weights, strain, reference=0.0):
    """W-orthogonal projection ``a = Phi^T W (q - q0)`` (rows of ``strain`` are states)."""
    return (np.asarray(strain, dtype=float) - reference) @ (np.asarray(weights)[:, None] * modes)


def weighted_norm(weights, strain):
    return float(np.sqrt(np.sum(np.asarray(weights) * np.asarray(strain, dtype=float) ** 2)))


def validate_reduction(cfg, reduction):
    modes = read_modes(reduction.modes_path)
    with open(reduction.metadata_path) as stream:
        metadata = yaml.safe_load(stream)
    order = reduction.n_modes
    dimension = cm.strain_dimension(cfg)
    if not isinstance(order, int) or isinstance(order, bool) or not 1 <= order <= modes.shape[1]:
        raise ValueError("Requested MOR order is outside the mode file")
    if modes.shape[0] != dimension or metadata["full_dimension"] != dimension:
        raise ValueError("MOR strain dimension does not match the cable")
    if metadata["n_modes"] != modes.shape[1] or metadata["number_of_sections"] != cfg["number_of_sections"]:
        raise ValueError("MOR metadata dimensions disagree with the mode file")
    if metadata["state_coordinates"] != "cosserat_modal" or metadata["component_order"] != list(
            cm.STRAIN_COMPONENTS):
        raise ValueError("MOR coordinates must be section-major 6-component Cosserat strains")
    if metadata["strain_basis_sha256"] != file_sha256(reduction.modes_path):
        raise ValueError("MOR mode file hash mismatch")
    if metadata["config_sha256"] != config_sha256(cfg):
        raise ValueError("MOR basis was generated for another cable configuration")
    reference = np.asarray(metadata["q_0"], dtype=float)
    if reference.shape != (dimension,) or np.any(reference != 0.):
        raise ValueError("The current cable ROM requires the physical zero-strain reference")
    weights = np.asarray(metadata["pod_weights"], dtype=float)
    if metadata["inner_product"] != INNER_PRODUCT or weights.shape != (dimension,) or not np.allclose(
            weights, cm.strain_weights(cfg), rtol=1e-12, atol=0.):
        raise ValueError("MOR inner product is not the strain-energy metric of this cable")
    if metadata["mapping_template"] != MAPPING_TEMPLATE:
        raise ValueError("MOR basis was generated for another mapping template")
    installed = installed_plugin_patches()
    if any(installed[name] != "unknown" and installed[name] != metadata["plugin_patches"].get(name)
           for name in installed):
        raise ValueError("MOR basis was generated with other Cosserat/MOR plugin patches than installed")
    singular = np.asarray(metadata["singular_values"], dtype=float)
    if singular.ndim != 1 or len(singular) < order or not np.all(np.isfinite(singular)):
        raise ValueError("Missing or invalid POD singular values")
    if singular[order - 1] <= singular[0] * metadata["rank_relative_tolerance"]:
        raise ValueError("MOR order includes numerical null-space modes")
    tolerance = float(metadata["orthogonality_tolerance"])
    if not 0. < tolerance <= 0.001:
        raise ValueError("Invalid MOR text precision tolerance")
    retained = modes[:, :order]
    np.testing.assert_allclose(retained.T @ (weights[:, None] * retained), np.eye(order),
                               atol=tolerance, rtol=0.)
    active = cm.active_components(cfg)
    if metadata["active_components"] != active:
        raise ValueError("MOR active components do not match the planar configuration")
    inactive = sorted(set(range(dimension)) - set(active))
    if inactive and np.max(np.abs(retained[inactive])) > 1e-10:
        raise ValueError("Planar ROM has torsion, out-of-plane bending or out-of-plane shear modes")
    return str(Path(reduction.modes_path).resolve()), retained, metadata
