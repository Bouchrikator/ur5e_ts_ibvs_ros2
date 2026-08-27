#!/usr/bin/env python3
"""qmm_mpc_solver_node.py — Online Quasi-Min-Max MPC solver for IBVS (ROS 2).

Paper-faithful implementation of:
  Wang et al. (2014) "Quasi-Min-Max MPC for IBVS with TP Model Transformation"
  Asian Journal of Control, Vol.17, No.2, pp.402-416

Equations (17)-(23) from this paper.

Decision variables (all in PHYSICAL units, no internal scaling):
  v_c(k|k) in R^6  — current camera velocity screw
  Q in S^2_{++}    — Lyapunov shape matrix (common to all 4 points)
  Y in R^{6x2}     — future gain variable (F = Y Q^{-1})
  gamma > 0        — worst-case cost bound

  min  gamma
  s.t. (a) Current step cost LMI     — Eq.(18) for j=1..4
       (b) Vertex Lyapunov LMI       — Eq.(19) for r=1..R
       (e) Current output constraint — Eq.(22) for j=1..4 [RECTANGULAR]
       (f) Future output LMI         — Eq.(23) for r=1..R [RECTANGULAR]

NOTE: The input-saturation constraints (Eq.20/21) are intentionally removed:
camera velocity is shaped solely by the Rw input-cost weight.

PERFORMANCE: cp.Parameter one-time compilation; each service call only
updates parameter data -> fast re-solve without recompilation.
"""

import os
import time

import cvxpy as cp
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from ibvs_msgs.srv import SolveMPC


class QMMSolverNode(Node):
    """ROS 2 service node wrapping the CVXPY SDP solver (Wang 2014)."""

    def __init__(self):
        super().__init__("qmm_mpc_solver_node")

        # ------------------------------------------------------------------
        # Load TP vertex matrices
        # ------------------------------------------------------------------
        vertex_file = self.declare_parameter("vertex_file", "").value
        if not vertex_file:
            vertex_file = os.path.join(
                get_package_share_directory("ibvs_control"), "config", "tp_vertices.yaml")

        self.get_logger().info(f"Loading vertices from: {vertex_file}")
        self.vertex_matrices = []
        self.vertex_matrices_by_point = []
        self.per_point_vertices = False

        with open(vertex_file, "r") as f:
            data = yaml.safe_load(f)

        # Support both TP (Wang 2014) and UD (Wang 2020) vertex formats.
        if "tp_model" in data:
            tp = data["tp_model"]
            vertices_key = "vertices"
        elif "tp_ud_model" in data:
            tp = data["tp_ud_model"]
            model_variant = self.declare_parameter("model_variant", "1param").value
            self.get_logger().info(f"Model variant requested: {model_variant}")
            if model_variant == "1param" and "model_1param" in tp:
                tp = tp["model_1param"]
                vertices_key = "vertices_1param"
            elif model_variant == "3param" and "model_3param" in tp:
                tp = tp["model_3param"]
                vertices_key = "vertices_3param"
            elif "vertices_5param" in tp:
                vertices_key = "vertices_5param"
            elif "model_3param" in tp:
                tp = tp["model_3param"]
                vertices_key = "vertices_3param"
            else:
                vertices_key = "vertices_5param"
        else:
            raise ValueError("YAML has neither 'tp_model' nor 'tp_ud_model' key")

        R = tp["num_vertices"]
        if vertices_key == "vertices_1param" and "vertices_1param_per_point" in tp:
            # Depth-only comparison: each feature point has its own two depth
            # vertices because u_i and v_i are fixed per point.
            self.per_point_vertices = True
            entries = sorted(tp["vertices_1param_per_point"],
                             key=lambda item: item["point_index"])
            if len(entries) != 4:
                raise ValueError("1-param per-point vertex model must contain 4 points")
            for point_entry in entries:
                vertices = sorted(point_entry["vertices"], key=lambda item: item["index"])
                if len(vertices) != R:
                    raise ValueError("Each point must contain num_vertices matrices")
                self.vertex_matrices_by_point.append(
                    [np.array(v["matrix_2x6"]).reshape(2, 6) for v in vertices])
            self.vertex_matrices = list(self.vertex_matrices_by_point[0])
            self.get_logger().info(
                f"Loaded {R} per-point vertex matrices for 4 points (1/Z only)")
        else:
            for v in tp[vertices_key]:
                self.vertex_matrices.append(np.array(v["matrix_2x6"]).reshape(2, 6))
            self.vertex_matrices_by_point = [list(self.vertex_matrices) for _ in range(4)]
            self.get_logger().info(f"Loaded {R} shared vertex matrices (2x6)")

        self.R = R

        # ------------------------------------------------------------------
        # Cost weights (Wang 2014 form: Qw = q*I2, Rw diagonal PD). Exposing
        # the scales avoids bang-bang saturation while preserving the exact
        # LMI structure (Eqs.17-23).
        # ------------------------------------------------------------------
        self.qw_scale = max(float(self.declare_parameter("Qw_scale", 1.0).value), 1e-12)
        self.rw_scale = max(float(self.declare_parameter("Rw_scale", 1.0).value), 1e-12)
        self.rw_linear_scale = float(
            self.declare_parameter("Rw_linear_scale", self.rw_scale).value)
        self.rw_angular_scale = float(
            self.declare_parameter("Rw_angular_scale", self.rw_scale).value)
        # wz decoupled from wx/wy so the solver can prefer wrist-spin over
        # camera tilt for in-plane alignment.
        self.rw_wz_scale = float(
            self.declare_parameter("Rw_wz_scale", self.rw_angular_scale).value)
        if self.rw_linear_scale <= 0.0:
            self.rw_linear_scale = self.rw_scale
        if self.rw_angular_scale <= 0.0:
            self.rw_angular_scale = self.rw_scale
        if self.rw_wz_scale <= 0.0:
            self.rw_wz_scale = self.rw_angular_scale

        self.Qw = self.qw_scale * np.eye(2)
        self.Rw_diag = np.array([
            self.rw_linear_scale, self.rw_linear_scale, self.rw_linear_scale,
            self.rw_angular_scale, self.rw_angular_scale, self.rw_wz_scale
        ], dtype=float)
        self.Rw = np.diag(self.Rw_diag)
        self.sqrtQw = np.sqrt(self.qw_scale) * np.eye(2)
        self.sqrtRw = np.diag(np.sqrt(self.Rw_diag))

        self.solver = self.declare_parameter("solver", "CLARABEL").value
        self.solver_max_iters = int(self.declare_parameter("solver_max_iters", 5000).value)
        self.solver_eps = float(self.declare_parameter("solver_eps", 1e-6).value)

        self._build_parametrized_problem()

        # Warm-start cache (physical variables).
        self._prev_vc = None
        self._prev_Q = None
        self._prev_Y = None
        self._prev_gamma = None

        self.srv = self.create_service(SolveMPC, "~/solve_mpc", self.handle_solve)
        self.get_logger().info("Service ready: ~/solve_mpc (Wang 2014, physical variables)")

    def _build_parametrized_problem(self):
        """Build the CVXPY problem once with cp.Parameter placeholders."""
        R = self.R
        t0 = time.time()

        self.p_errors = [cp.Parameter(2, name=f"e_{j}") for j in range(4)]
        self.p_J_current = [cp.Parameter((2, 6), name=f"Je_{j}") for j in range(4)]
        self.p_J_vertex = [[cp.Parameter((2, 6), name=f"Jv_{j}_{r}") for r in range(R)]
                           for j in range(4)]
        self.p_current_lb = [cp.Parameter(2, name=f"lb_{j}") for j in range(4)]
        self.p_current_ub = [cp.Parameter(2, name=f"ub_{j}") for j in range(4)]
        # Pre-squared symmetric future bounds (DPP-compliant).
        self.p_future_bound_sq = [cp.Parameter(2, nonneg=True, name=f"fut_sq_{j}")
                                  for j in range(4)]

        self.var_vc = cp.Variable(6, name="vc")
        self.var_Q = cp.Variable((2, 2), symmetric=True, name="Q")
        self.var_Y = cp.Variable((6, 2), name="Y")
        self.var_gamma = cp.Variable(name="gamma")

        vc, Q, Y, gamma = self.var_vc, self.var_Q, self.var_Y, self.var_gamma
        constraints = [Q >> 1e-4 * np.eye(2), gamma >= 1e-6]

        # (a) Current step cost LMI — Eq.(18)
        for j in range(4):
            ej = self.p_errors[j]
            zj = ej + self.p_J_current[j] @ vc
            sqQw_ej = self.sqrtQw @ ej
            sqRw_vc = self.sqrtRw @ vc
            M_a = cp.bmat([
                [np.ones((1, 1)), cp.reshape(zj, (1, 2), order="F"),
                 cp.reshape(sqQw_ej, (1, 2), order="F"),
                 cp.reshape(sqRw_vc, (1, 6), order="F")],
                [cp.reshape(zj, (2, 1), order="F"), Q, np.zeros((2, 2)), np.zeros((2, 6))],
                [cp.reshape(sqQw_ej, (2, 1), order="F"), np.zeros((2, 2)), gamma * np.eye(2),
                 np.zeros((2, 6))],
                [cp.reshape(sqRw_vc, (6, 1), order="F"), np.zeros((6, 2)), np.zeros((6, 2)),
                 gamma * np.eye(6)],
            ])
            constraints.append(M_a >> 0)

        # (b) Vertex Lyapunov LMI — Eq.(19)
        for j in range(4):
            for r in range(R):
                G_r = Q + self.p_J_vertex[j][r] @ Y
                sqQw_Q = self.sqrtQw @ Q
                sqRw_Y = self.sqrtRw @ Y
                M_b = cp.bmat([
                    [Q, G_r.T, sqQw_Q.T, sqRw_Y.T],
                    [G_r, Q, np.zeros((2, 2)), np.zeros((2, 6))],
                    [sqQw_Q, np.zeros((2, 2)), gamma * np.eye(2), np.zeros((2, 6))],
                    [sqRw_Y, np.zeros((6, 2)), np.zeros((6, 2)), gamma * np.eye(6)],
                ])
                constraints.append(M_b >> 0)

        # (e) Current output constraint — Eq.(22), rectangular per point/dim.
        for j in range(4):
            zj = self.p_errors[j] + self.p_J_current[j] @ vc
            for d in range(2):
                constraints.append(zj[d] >= self.p_current_lb[j][d])
                constraints.append(zj[d] <= self.p_current_ub[j][d])

        # (f) Future output LMI — Eq.(23): the invariant ellipsoid
        # {x: x'Q^-1 x <= 1} stays within the per-point rectangular bounds
        # after any vertex dynamics (I + J_r F).
        for j in range(4):
            for r in range(R):
                G_r = Q + self.p_J_vertex[j][r] @ Y
                for d in range(2):
                    g_row = cp.reshape(G_r[d, :], (1, 2), order="F")
                    g_col = cp.reshape(G_r[d, :], (2, 1), order="F")
                    M_f = cp.bmat([
                        [cp.reshape(self.p_future_bound_sq[j][d], (1, 1), order="F"), g_row],
                        [g_col, Q],
                    ])
                    constraints.append(M_f >> 0)

        self.prob = cp.Problem(cp.Minimize(gamma), constraints)

        # Force compilation with dummy data.
        self._set_dummy_parameters()
        try:
            self.prob.solve(solver=self.solver, verbose=False, warm_start=False)
            self.get_logger().info(f"Problem compiled OK (status={self.prob.status})")
        except Exception as e:  # noqa: BLE001 — first solve may legitimately fail
            self.get_logger().warn(f"Dummy solve warning (expected): {e}")

        self.get_logger().info(
            f"Parametrized SDP built in {(time.time() - t0) * 1000:.0f}ms "
            f"({len(constraints)} constraints, R={R})")

    def _set_dummy_parameters(self):
        for j in range(4):
            self.p_errors[j].value = np.array([10.0, 10.0])
            self.p_J_current[j].value = np.eye(2, 6) * 0.01
            self.p_current_lb[j].value = np.array([-200.0, -200.0])
            self.p_current_ub[j].value = np.array([200.0, 200.0])
            self.p_future_bound_sq[j].value = np.array([150.0**2, 150.0**2])
            for r in range(self.R):
                self.p_J_vertex[j][r].value = np.eye(2, 6) * 0.01

    def handle_solve(self, req, resp):
        t_start = time.time()
        try:
            result = self._solve_sdp(req)
        except Exception as e:  # noqa: BLE001 — solver must never kill the service
            self.get_logger().warn(f"Solver exception: {e}")
            resp.vc = [0.0] * 6
            resp.gamma = -1.0
            resp.feasible = False
            resp.solver_time_ms = (time.time() - t_start) * 1000.0
            resp.lyapunov_v = 0.0
            resp.q_flat = [0.0] * 4
            resp.y_flat = [0.0] * 12
            resp.solver_status = f"exception: {e}"
            return resp

        resp.vc = result["vc"].tolist()
        resp.gamma = float(result["gamma"])
        resp.feasible = result["feasible"]
        resp.solver_time_ms = (time.time() - t_start) * 1000.0
        resp.lyapunov_v = float(result["lyapunov_V"])
        resp.q_flat = result["Q"].flatten().tolist()
        resp.y_flat = result["Y"].flatten().tolist()
        resp.solver_status = result["status"]
        return resp

    def _solve_sdp(self, req):
        """Update parameter values and re-solve the pre-compiled SDP."""
        Ts = req.ts
        errors = np.array(req.errors_flat).reshape(4, 2)
        jacobians = np.array(req.jacobians_flat).reshape(4, 2, 6)
        desired = np.array(req.desired_positions_flat).reshape(4, 2)
        R = self.R

        # Discrete Jacobians: J_e,j = Ts * J_m,j (Eq.6).
        J_current = [Ts * jacobians[j] for j in range(4)]
        J_vertices = [[Ts * self.vertex_matrices_by_point[j][r] for r in range(R)]
                      for j in range(4)]

        # Per-point rectangular error bounds from the visibility window.
        current_lb = np.zeros((4, 2))
        current_ub = np.zeros((4, 2))
        for j in range(4):
            current_lb[j, 0] = req.u_min - desired[j, 0]
            current_ub[j, 0] = req.u_max - desired[j, 0]
            current_lb[j, 1] = req.v_min - desired[j, 1]
            current_ub[j, 1] = req.v_max - desired[j, 1]

        # Largest symmetric region centred at s_j* inside the rectangle.
        future_bound = np.zeros((4, 2))
        for j in range(4):
            for d in range(2):
                future_bound[j, d] = max(min(current_ub[j, d], -current_lb[j, d]), 1.0)

        for j in range(4):
            self.p_errors[j].value = errors[j]
            self.p_J_current[j].value = J_current[j]
            self.p_current_lb[j].value = current_lb[j]
            self.p_current_ub[j].value = current_ub[j]
            self.p_future_bound_sq[j].value = future_bound[j] ** 2
            for r in range(R):
                self.p_J_vertex[j][r].value = J_vertices[j][r]

        use_warm = False
        if self._prev_vc is not None:
            try:
                self.var_vc.value = self._prev_vc.copy()
                self.var_Q.value = self._prev_Q.copy()
                self.var_Y.value = self._prev_Y.copy()
                self.var_gamma.value = float(self._prev_gamma)
                use_warm = True
            except Exception:  # noqa: BLE001
                pass

        solver_kwargs = {"solver": self.solver, "verbose": False, "warm_start": use_warm}
        if self.solver == "CLARABEL":
            solver_kwargs.update(tol_gap_abs=self.solver_eps, tol_gap_rel=self.solver_eps,
                                 tol_feas=self.solver_eps, max_iter=self.solver_max_iters)
        elif self.solver == "SCS":
            solver_kwargs.update(max_iters=self.solver_max_iters, eps=self.solver_eps)

        t_pre = time.time()
        try:
            self.prob.solve(**solver_kwargs)
        except cp.SolverError as e:
            self.get_logger().warn(f"SolverError with {self.solver}: {e}")
            if self.solver == "CLARABEL":
                self.get_logger().warn("Falling back to SCS...")
                try:
                    self.prob.solve(solver="SCS", max_iters=self.solver_max_iters,
                                    eps=self.solver_eps, verbose=False, warm_start=False)
                except cp.SolverError as e2:
                    return self._infeasible_result(f"both_failed: {e2}")
            else:
                return self._infeasible_result(str(e))

        t_solve = (time.time() - t_pre) * 1000.0
        self.get_logger().info(
            f"solve={t_solve:.0f}ms status={self.prob.status} warm={use_warm}",
            throttle_duration_sec=1.0)

        if self.prob.status in ("optimal", "optimal_inaccurate"):
            vc_val = self.var_vc.value.copy()
            Q_val = self.var_Q.value.copy()
            Y_val = self.var_Y.value.copy()
            gamma_val = float(self.var_gamma.value)

            self._prev_vc = vc_val.copy()
            self._prev_Q = Q_val.copy()
            self._prev_Y = Y_val.copy()
            self._prev_gamma = gamma_val

            # V(k) = sum_j e_j'Qw e_j + vc'Rw vc + z_j'S z_j, S = gamma Q^-1.
            S_val = gamma_val * np.linalg.inv(Q_val)
            V = 0.0
            for j in range(4):
                ej = errors[j]
                zj = ej + J_current[j] @ vc_val
                V += ej @ self.Qw @ ej + vc_val @ self.Rw @ vc_val + zj @ S_val @ zj

            return {"vc": vc_val, "gamma": gamma_val, "Q": Q_val, "Y": Y_val,
                    "feasible": True, "lyapunov_V": V, "status": self.prob.status}

        self.get_logger().warn(f"Problem status: {self.prob.status}")
        self._prev_vc = None
        return self._infeasible_result(self.prob.status)

    @staticmethod
    def _infeasible_result(status_str):
        return {"vc": np.zeros(6), "gamma": -1.0, "Q": np.eye(2),
                "Y": np.zeros((6, 2)), "feasible": False, "lyapunov_V": 0.0,
                "status": status_str}


def main(args=None):
    rclpy.init(args=args)
    node = QMMSolverNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
