"""TS identification on the ROM's own modal coordinates, run from the SOFA scene.

The numerical fit is the repository's ``local_identification`` /
``premises`` / ``ts_model``; nothing here re-implements dynamics. The only
SOFA work is the marker-space metric: a predicted ``a`` is pushed through the
scene mapping (``Phi a -> DiscreteCosseratMapping``), never through a marker POD.
"""

import numpy as np
import Sofa.Core
from scipy.linalg import expm

from cable_identification.strain_basis import config_sha256
from cable_ts_control.local_identification import (
    _project_psd, blended_predict, blended_rollout, split_by_premise_vertex)
from cable_ts_control.modal_contract import COORDINATES, build_modal_state, check_dataset_contract
from cable_ts_control.premises import make_premise_map
from cable_ts_control.scripts.identify_ts_vertices import check_input_alignment
from cable_ts_control.ts_model import CableTsModel, rule_memberships


def mass_metric(observation, modal, mass_kg, step=1e-4):
    """Translational mass metric ``M(a) = sum_f m_f J_f' J_f`` of the ROM at ``a``.

    ``J_f = d p_f / d a`` comes from central differences of the scene mapping
    (``Phi a -> frames``), so the metric is the one the SOFA graph implies,
    not a second model. Strain coordinates are far from mass-normalised (the
    condition number is ~1e6), which is why the passive projection must be
    done in this metric and not in raw ``a``.
    """
    modal = np.asarray(modal, dtype=float)
    columns = []
    for unit in np.eye(modal.size):
        forward = observation.frames(modal + step * unit)
        backward = observation.frames(modal - step * unit)
        columns.append((forward - backward) / (2.0 * step))
    jac = np.column_stack(columns)
    metric = (mass_kg / (jac.shape[0] // 3)) * jac.T @ jac
    return 0.5 * (metric + metric.T)


def fit_modal_structured_model(states, commands, next_states, memberships, dt, n_modes, metric,
                               ridge, input_feedthrough, floor_ratio=1e-9):
    """Structured second-order TS fit in ``a`` with the passive projection in the mass metric.

    ``fit_structured_fuzzy_model`` projects the fitted stiffness/damping on the PSD
    cone in raw state coordinates. In strain-POD coordinates ``M^-1 K`` is far from
    symmetric (condition number of ``M`` ~1e6), so that projection destroys the fit
    (held-out one-step error 7-37 mm). Here the same regression is done in the
    mass-normalised coordinates ``z = L' a`` (``M = L L'``), where a passive rod
    has symmetric PSD matrices, and the exactly discretised ``(A_i, B_i)`` are
    mapped back by similarity so the model stays in ``[a, a_dot, g]``.
    """
    states, commands = np.asarray(states, float), np.asarray(commands, float)
    next_states, memberships = np.asarray(next_states, float), np.asarray(memberships, float)
    n_state, n_input, n_rules = states.shape[1], commands.shape[1], memberships.shape[1]
    n_boundary = n_state - 2 * n_modes
    lower = np.linalg.cholesky(metric)
    transform = np.eye(n_state)
    transform[:n_modes, :n_modes] = lower.T
    transform[n_modes:2 * n_modes, n_modes:2 * n_modes] = lower.T
    transform_inv = np.linalg.inv(transform)
    z, z_next = states @ transform.T, next_states @ transform.T
    acceleration = (z_next[:, n_modes:2 * n_modes] - z[:, n_modes:2 * n_modes]) / dt
    columns = [z, commands] if input_feedthrough else [z]
    block = sum(c.shape[1] for c in columns)
    regressor = (memberships[:, :, None] * np.hstack(columns)[:, None, :]).reshape(len(z), n_rules * block)
    width = regressor.shape[1]
    regressor = np.vstack([regressor, np.sqrt(ridge) * np.eye(width)])
    target = np.vstack([acceleration, np.zeros((width, n_modes))])
    theta, *_ = np.linalg.lstsq(regressor, target, rcond=None)

    a_vertices, b_vertices = [], []
    for rule in range(n_rules):
        chunk = theta[rule * block:(rule + 1) * block]
        stiffness = _project_psd(-chunk[:n_modes].T, floor_ratio)
        damping = _project_psd(-chunk[n_modes:2 * n_modes].T, floor_ratio)
        boundary = chunk[2 * n_modes:n_state].T
        gain = chunk[n_state:].T if input_feedthrough else np.zeros((n_modes, n_input))
        a_c = np.zeros((n_state, n_state))
        a_c[:n_modes, n_modes:2 * n_modes] = np.eye(n_modes)
        a_c[n_modes:2 * n_modes, :n_modes] = -stiffness
        a_c[n_modes:2 * n_modes, n_modes:2 * n_modes] = -damping
        a_c[n_modes:2 * n_modes, 2 * n_modes:] = boundary
        b_c = np.zeros((n_state, n_input))
        b_c[n_modes:2 * n_modes] = gain
        b_c[2 * n_modes:] = np.eye(n_boundary, n_input)
        joint = np.zeros((n_state + n_input, n_state + n_input))
        joint[:n_state, :n_state], joint[:n_state, n_state:] = a_c, b_c
        discrete = expm(joint * dt)
        a_vertices.append(transform_inv @ discrete[:n_state, :n_state] @ transform)
        b_vertices.append(transform_inv @ discrete[:n_state, n_state:])
    return a_vertices, b_vertices


class CableModalTsIdentificationController(Sofa.Core.Controller):
    def __init__(self, observation, dataset, basis_metadata, settings, dataset_sha256, **kwargs):
        super().__init__(**kwargs)
        check_dataset_contract(dataset, basis_metadata)
        self.observation = observation
        self.data = {key: np.asarray(dataset[key]) for key in dataset.files}
        self.metadata = basis_metadata
        self.settings = settings
        self.dataset_sha256 = dataset_sha256
        self.dt = float(self.data["timestep_s"][0])
        self.n_modes = int(basis_metadata["n_modes"])
        check_input_alignment(self.data, self.dt)

    def marker_rmse(self, modal_states, shapes):
        predicted = np.stack([self.observation.markers(a) for a in modal_states])
        return float(np.sqrt(np.mean((predicted - np.asarray(shapes)) ** 2)))

    def windowed_rollout(self, a_rules, b_rules, bounds, premise_map, xk, uk, shapes, horizon):
        """Free rollouts of ``horizon`` steps restarted from the truth every ``horizon`` steps.

        One rollout from the first sample would sit entirely in the initial hold.
        ``a_rules=None`` gives the trivial constant-velocity extrapolation baseline.
        """
        errors = []
        for start in range(0, len(uk) - horizon, horizon):
            if a_rules is None:
                steps = np.arange(1, horizon + 1)[:, None]
                modal = xk[start, :self.n_modes] + steps * self.dt * xk[start, self.n_modes:2 * self.n_modes]
            else:
                modal = blended_rollout(a_rules, b_rules, bounds, xk[start], uk[start:start + horizon],
                                        None, premise_map)[:, :self.n_modes]
            errors.append(np.stack([self.observation.markers(a) for a in modal]) - shapes[start:start + horizon])
        return float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))

    def segments(self, gripper_reference):
        out = {}
        keys = np.stack([self.data["vertex_ids"], self.data["trajectory_ids"]], axis=1)
        for vertex, trajectory in np.unique(keys, axis=0):
            sel = (keys == (vertex, trajectory)).all(axis=1)
            x = build_modal_state(self.data["modal_positions"][sel],
                                  self.data["modal_velocities_state"][sel],
                                  self.data["gripper"][sel], gripper_reference)
            out[(int(vertex), int(trajectory))] = (
                x[:-1], self.data["commands"][sel][:-1], x[1:], self.data["shapes"][sel][1:])
        return out

    def identify(self):
        ts = self.settings["ts"]
        train_ids = list(self.settings["train_trajectories"])
        held_out = list(self.settings["validation_trajectories"]) + list(self.settings["test_trajectories"])
        vertices = sorted(int(v) for v in np.unique(self.data["vertex_ids"]))
        train_mask = np.isin(self.data["trajectory_ids"], train_ids)
        gripper_reference = self.data["gripper"][train_mask].mean(axis=0)
        segments = self.segments(gripper_reference)
        premise_map = make_premise_map("boundary", gripper_index=2 * self.n_modes,
                                       cable_length_m=float(self.data["cable_length_m"][0]),
                                       gripper_reference=gripper_reference)
        training = {v: np.concatenate([segments[(v, t)][0] for t in train_ids]) for v in vertices}
        blocks = [premise_map(training[v]) for v in vertices]
        split = np.mean([np.median(b, axis=0) for b in blocks], axis=0)
        half = np.min([np.minimum(split - b.min(axis=0), b.max(axis=0) - split) for b in blocks], axis=0)
        if np.any(half <= 0.0):
            raise ValueError("no premise split is straddled by every physical vertex")
        bounds = [(float(split[i] - half[i]), float(split[i] + half[i])) for i in range(2)]
        horizon = round(ts["rollout_seconds"] / self.dt)
        a_mean = self.data["modal_positions"][train_mask].mean(axis=0)
        metric = mass_metric(self.observation, a_mean, float(self.observation.cable.cfg["mass_kg"]))

        a_sets, b_sets, report = [], [], {"vertices": [], "mass_metric_condition": float(np.linalg.cond(metric))}
        for vertex in vertices:
            x = training[vertex]
            u = np.concatenate([segments[(vertex, t)][1] for t in train_ids])
            x_next = np.concatenate([segments[(vertex, t)][2] for t in train_ids])
            memberships = np.array([rule_memberships(p, bounds) for p in premise_map(x)])
            counts = [len(c) for c in split_by_premise_vertex(premise_map(x), bounds)]
            if min(counts) < x.shape[1] + u.shape[1] + 4:
                raise ValueError(f"vertex {vertex}: rule cell with only {min(counts)} samples")
            a_rules, b_rules = fit_modal_structured_model(
                x, u, x_next, memberships, self.dt, self.n_modes, metric, ts["ridge"], ts["input_feedthrough"])
            entry = {"vertex_id": vertex, "cells": counts, "open_loop_rho": float(max(
                np.max(np.abs(np.linalg.eigvals(a))) for a in a_rules)), "held_out": []}
            for trajectory in held_out:
                xk, uk, xk1, shapes = segments[(vertex, trajectory)]
                predicted = blended_predict(a_rules, b_rules, bounds, xk, uk, None, premise_map)
                entry["held_out"].append({
                    "trajectory_id": trajectory,
                    "one_step_modal_rmse": float(np.sqrt(np.mean(
                        (predicted[:, :self.n_modes] - xk1[:, :self.n_modes]) ** 2))),
                    "one_step_marker_rmse_m": self.marker_rmse(predicted[:, :self.n_modes], shapes),
                    "rollout_marker_rmse_m": self.windowed_rollout(
                        a_rules, b_rules, bounds, premise_map, xk, uk, shapes, horizon),
                    "trivial_constant_velocity_rollout_m": self.windowed_rollout(
                        None, None, bounds, premise_map, xk, uk, shapes, horizon),
                })
            report["vertices"].append(entry)
            a_sets.append(a_rules)
            b_sets.append(b_rules)
            print(f"TS vertex={vertex} cells={counts} rho={entry['open_loop_rho']:.4f} held-out "
                  + " ".join(f"t{h['trajectory_id']}: one-step {h['one_step_marker_rmse_m'] * 1e3:.3f} mm, "
                             f"rollout {h['rollout_marker_rmse_m'] * 1e3:.3f} mm "
                             f"(const-velocity {h['trivial_constant_velocity_rollout_m'] * 1e3:.1f} mm)"
                             for h in entry["held_out"]),
                  flush=True)
        worst_one_step = max(h["one_step_marker_rmse_m"] for e in report["vertices"] for h in e["held_out"])
        worst_rollout = max(h["rollout_marker_rmse_m"] for e in report["vertices"] for h in e["held_out"])
        report.update(worst_one_step_marker_rmse_m=worst_one_step, worst_rollout_marker_rmse_m=worst_rollout,
                      premise_bounds=bounds, gripper_reference=gripper_reference.tolist(),
                      passed=bool(worst_one_step <= ts["max_one_step_marker_rmse_m"]
                                  and worst_rollout <= ts["max_rollout_marker_rmse_m"]))
        names = [str(n) for n in self.data["parameter_names"]]
        model = CableTsModel(
            a_sets[0], b_sets[0], bounds, self.dt, n_modes=self.n_modes, premise_map=premise_map,
            parameter_bounds={"names": names, "values": self.data["parameter_values"].tolist(),
                              "A": [[a.flatten().tolist() for a in g] for g in a_sets],
                              "B": [[b.flatten().tolist() for b in g] for g in b_sets]})
        payload = model.to_dict()
        payload["cable_ts_model"].update({
            "state_coordinates": COORDINATES, "state_layout": "[a, a_dot, p_g - p_g0]",
            "strain_basis_sha256": self.metadata["strain_basis_sha256"],
            "dataset_sha256": self.dataset_sha256,
            "velocity_source": "causal_filter", "velocity_alpha": float(self.data["velocity_alpha"][0]),
            "gripper_reference": gripper_reference.tolist(),
            "input_feedthrough": bool(ts["input_feedthrough"]),
            "validation": report, "settings_sha256": config_sha256(self.settings)})
        return payload, report
