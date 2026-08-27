#!/usr/bin/env python3
"""
DISCRETE-TIME TS-PDC IBVS LMI Gain Solver

Solves the convex LMI synthesis problem for Takagi-Sugeno PDC visual servoing
with 4 point features (k=8) using Chaumette's reduced error formulation.

DISCRETE-TIME formulation (Tanaka & Wang, 2001):
  x(k+1) = x(k) + Ts * B(Z) * u(k)

Closed-loop with PDC: u(k) = -(Σ h_j F_j) x(k)
  x(k+1) = (I - Ts * B_i * F_j) x(k) = G_ij x(k)

Discrete LMI synthesis (Schur complement form):
  For variables X ≻ 0, M_i:
    G_ij = I - Ts * B_i * F_j,  F_j = M_j @ inv(X), P = inv(X)
  
  Stability conditions (via Schur complement of G^T P G - P < 0):
    [X,        (X - Ts*B_i*M_i)]
    [(X - Ts*B_i*M_i)^T,    X  ] > 0   for each i
  
  And cross-term for i≠j:
    [X,        (X - Ts*(B_i*M_j + B_j*M_i)/2)]
    [(...)^T,                               X ] > 0

Author: Generated for UR5e IBVS Project (Discrete-Time)
"""

import numpy as np
import cvxpy as cp
from typing import Tuple, Optional
import yaml
import os


def build_interaction_matrix_point(x: float, y: float, Z: float) -> np.ndarray:
    """
    Build the 2x6 interaction matrix for a single point feature.
    
    Following Chaumette 2006:
    L = [[-1/Z,    0, x/Z,  xy, -(1+x²),  y],
         [   0, -1/Z, y/Z, 1+y²,   -xy, -x]]
    """
    L = np.array([
        [-1.0/Z,     0.0, x/Z,      x*y, -(1+x*x),  y],
        [    0.0, -1.0/Z, y/Z,  1+y*y,     -x*y, -x]
    ])
    return L


def build_stacked_interaction_matrix(points: np.ndarray, Z: float) -> np.ndarray:
    """
    Build the 8x6 stacked interaction matrix for 4 point features.
    """
    L_e = np.zeros((8, 6))
    for i in range(4):
        x, y = points[i]
        L_e[2*i:2*i+2, :] = build_interaction_matrix_point(x, y, Z)
    return L_e


def compute_vertex_matrices(
    s_star: np.ndarray,
    Z0: float,
    Z_min: float,
    Z_max: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the constant matrices for TS-PDC design.
    """
    L_hat = build_stacked_interaction_matrix(s_star, Z0)
    L_hat_pinv = np.linalg.pinv(L_hat)
    
    L_far = build_stacked_interaction_matrix(s_star, Z_max)
    L_close = build_stacked_interaction_matrix(s_star, Z_min)
    
    B1 = L_hat_pinv @ L_far
    B2 = L_hat_pinv @ L_close
    
    return L_hat, L_hat_pinv, B1, B2


def solve_ts_pdc_lmi_discrete(
    B1: np.ndarray,
    B2: np.ndarray,
    Ts: float,
    eps: float = 1e-6,
    verbose: bool = True
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool]:
    """
    Solve the DISCRETE-TIME convex LMI synthesis problem for TS-PDC gains.
    
    Discrete closed-loop: G_ij = I - Ts * B_i * F_j
    Stability: G_ij^T P G_ij - P < 0  (eigenvalues inside unit circle)
    
    Using Schur complement, G^T P G - P < 0  with P = X^{-1} becomes:
        [X,       G_ij @ X]
        [X @ G_ij^T,    X ] > 0
    
    With F_j = M_j @ X^{-1}:
        G_ij @ X = (I - Ts * B_i * F_j) @ X = X - Ts * B_i * M_j
    
    So the LMI conditions become:
        [X,                    X - Ts * B_i * M_j  ]
        [(X - Ts * B_i * M_j)^T,              X    ] > 0
    
    Args:
        B1: 6x6 projected input matrix at far vertex
        B2: 6x6 projected input matrix at close vertex
        Ts: Sampling period
        eps: Small positive margin for strict inequality
        verbose: Print solver output
    
    Returns:
        F1: 6x6 gain matrix for far rule (or None if infeasible)
        F2: 6x6 gain matrix for close rule (or None if infeasible)
        P: 6x6 Lyapunov matrix (or None if infeasible)
        feasible: True if LMIs are feasible
    """
    n = 6  # State dimension
    
    # Decision variables
    X = cp.Variable((n, n), symmetric=True)
    M1 = cp.Variable((n, n))
    M2 = cp.Variable((n, n))
    
    # Constraints
    constraints = []
    
    # X ≻ eps*I (positive definite)
    constraints.append(X >> eps * np.eye(n))
    
    # Build G_ij @ X = X - Ts * B_i * M_j
    GX_11 = X - Ts * B1 @ M1   # G_11 @ X
    GX_22 = X - Ts * B2 @ M2   # G_22 @ X
    GX_12 = X - Ts * B1 @ M2   # G_12 @ X
    GX_21 = X - Ts * B2 @ M1   # G_21 @ X
    
    # Condition 1 (Schur): G_11^T P G_11 - P < 0
    #   [X,     GX_11]
    #   [GX_11^T,  X ] > eps*I
    S1 = cp.bmat([[X, GX_11],
                   [GX_11.T, X]])
    constraints.append(S1 >> eps * np.eye(2*n))
    
    # Condition 2 (Schur): G_22^T P G_22 - P < 0
    S2 = cp.bmat([[X, GX_22],
                   [GX_22.T, X]])
    constraints.append(S2 >> eps * np.eye(2*n))
    
    # Condition 3 (Schur): cross-term ((G_12+G_21)/2)^T P ((G_12+G_21)/2) - P < 0
    #   Use average: GX_cross = (GX_12 + GX_21) / 2
    GX_cross = 0.5 * (GX_12 + GX_21)
    S3 = cp.bmat([[X, GX_cross],
                   [GX_cross.T, X]])
    constraints.append(S3 >> eps * np.eye(2*n))
    
    # Objective: Minimize trace(X) + regularization on gains
    objective = cp.Minimize(cp.trace(X) + 0.01 * (cp.norm(M1, 'fro') + cp.norm(M2, 'fro')))
    
    # Solve
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.MOSEK, verbose=verbose)
    except (cp.error.SolverError, Exception):
        try:
            prob.solve(solver=cp.SCS, verbose=verbose, max_iters=10000)
        except (cp.error.SolverError, Exception):
            prob.solve(solver=cp.CVXOPT, verbose=verbose)
    
    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        X_val = X.value
        M1_val = M1.value
        M2_val = M2.value
        
        # Recover gains: F_i = M_i @ inv(X)
        X_inv = np.linalg.inv(X_val)
        F1 = M1_val @ X_inv
        F2 = M2_val @ X_inv
        
        # Lyapunov matrix: P = inv(X)
        P = X_inv
        
        return F1, F2, P, True
    else:
        print(f"LMI problem status: {prob.status}")
        return None, None, None, False


def verify_stability_discrete(
    B1: np.ndarray, B2: np.ndarray,
    F1: np.ndarray, F2: np.ndarray,
    P: np.ndarray,
    Ts: float,
    Z_min: float, Z_max: float,
    n_samples: int = 50
) -> Tuple[float, np.ndarray]:
    """
    Verify discrete closed-loop stability: all eigenvalues of G_cl inside unit circle.
    """
    alpha_min = 1.0 / Z_max
    alpha_max = 1.0 / Z_min
    
    max_eig_mag = 0.0
    all_eigs = []
    
    Z_samples = np.linspace(Z_min, Z_max, n_samples)
    
    print("\n" + "="*70)
    print("DISCRETE STABILITY VERIFICATION")
    print("="*70)
    
    for Z in Z_samples:
        alpha = 1.0 / Z
        
        h_far = np.clip((alpha_max - alpha) / (alpha_max - alpha_min), 0, 1)
        h_close = np.clip((alpha - alpha_min) / (alpha_max - alpha_min), 0, 1)
        
        B_z = h_far * B1 + h_close * B2
        F_z = h_far * F1 + h_close * F2
        
        # Discrete closed-loop: x(k+1) = (I - Ts*B*F) x(k)
        G_cl = np.eye(6) - Ts * B_z @ F_z
        
        eigs = np.linalg.eigvals(G_cl)
        max_mag = np.max(np.abs(eigs))
        
        all_eigs.append(eigs)
        
        if max_mag > max_eig_mag:
            max_eig_mag = max_mag
    
    print(f"Depth range: [{Z_min}, {Z_max}] m")
    print(f"Sampling period: Ts = {Ts} s")
    print(f"Maximum |eigenvalue|: {max_eig_mag:.6e}")
    
    if max_eig_mag < 1.0:
        print("✓ All closed-loop eigenvalues INSIDE UNIT CIRCLE (|λ| < 1)")
        print("✓ Discrete system is STABLE across entire depth range.")
    else:
        print("✗ WARNING: Some eigenvalues OUTSIDE unit circle!")
        print("✗ Discrete stability NOT guaranteed!")
    
    return max_eig_mag, np.array(all_eigs)


def verify_lyapunov_conditions_discrete(
    B1: np.ndarray, B2: np.ndarray,
    F1: np.ndarray, F2: np.ndarray,
    P: np.ndarray,
    Ts: float
) -> bool:
    """
    Verify the discrete Lyapunov LMI conditions:
      G_ii^T P G_ii - P < 0
      G_cross^T P G_cross - P < 0
    where G_ij = I - Ts*B_i*F_j, G_cross = (G_12 + G_21)/2
    """
    print("\n" + "="*70)
    print("DISCRETE LYAPUNOV LMI VERIFICATION")
    print("="*70)
    
    I6 = np.eye(6)
    
    G11 = I6 - Ts * B1 @ F1
    G22 = I6 - Ts * B2 @ F2
    G12 = I6 - Ts * B1 @ F2
    G21 = I6 - Ts * B2 @ F1
    
    # Condition 1: G_11^T P G_11 - P < 0
    Q1 = G11.T @ P @ G11 - P
    eigs1 = np.linalg.eigvalsh(Q1)
    max_eig1 = np.max(eigs1)
    print(f"Condition 1: G11^T P G11 - P")
    print(f"  Eigenvalues: {eigs1}")
    print(f"  Max eigenvalue: {max_eig1:.6e} {'< 0 ✓' if max_eig1 < 0 else '>= 0 ✗'}")
    
    # Condition 2: G_22^T P G_22 - P < 0
    Q2 = G22.T @ P @ G22 - P
    eigs2 = np.linalg.eigvalsh(Q2)
    max_eig2 = np.max(eigs2)
    print(f"\nCondition 2: G22^T P G22 - P")
    print(f"  Eigenvalues: {eigs2}")
    print(f"  Max eigenvalue: {max_eig2:.6e} {'< 0 ✓' if max_eig2 < 0 else '>= 0 ✗'}")
    
    # Condition 3: G_cross^T P G_cross - P < 0
    G_cross = 0.5 * (G12 + G21)
    Q3 = G_cross.T @ P @ G_cross - P
    eigs3 = np.linalg.eigvalsh(Q3)
    max_eig3 = np.max(eigs3)
    print(f"\nCondition 3: Gcross^T P Gcross - P")
    print(f"  Eigenvalues: {eigs3}")
    print(f"  Max eigenvalue: {max_eig3:.6e} {'< 0 ✓' if max_eig3 < 0 else '>= 0 ✗'}")
    
    all_satisfied = (max_eig1 < 0) and (max_eig2 < 0) and (max_eig3 < 0)
    
    print("\n" + "-"*70)
    if all_satisfied:
        print("✓ ALL THREE DISCRETE LYAPUNOV CONDITIONS SATISFIED")
    else:
        print("✗ SOME CONDITIONS NOT SATISFIED")
    
    return all_satisfied


def save_gains_to_yaml(
    F1: np.ndarray, F2: np.ndarray, P: np.ndarray,
    L_hat_pinv: np.ndarray,
    B1: np.ndarray, B2: np.ndarray,
    s_star: np.ndarray,
    Z0: float, Z_min: float, Z_max: float,
    Ts: float,
    filepath: str
):
    """Save discrete LMI-solved gains to YAML file for ROS node loading."""
    
    data = {
        'ts_pdc_lmi_discrete_gains': {
            # Depth parameters
            'Z0': float(Z0),
            'Z_min': float(Z_min),
            'Z_max': float(Z_max),
            
            # Sampling period
            'Ts': float(Ts),
            
            # Desired features (flattened)
            's_star': s_star.flatten().tolist(),
            
            # PDC gains (row-major flattened)
            'F1': F1.flatten().tolist(),
            'F2': F2.flatten().tolist(),
            
            # Lyapunov matrix
            'P': P.flatten().tolist(),
            
            # Constant pseudo-inverse matrix
            'L_hat_pinv': L_hat_pinv.flatten().tolist(),
            
            # Vertex matrices (for verification)
            'B1': B1.flatten().tolist(),
            'B2': B2.flatten().tolist(),
            
            # Matrix dimensions
            'n_features': 4,
            'state_dim': 6,
            'error_dim': 8,
        }
    }
    
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, width=200)
    
    print(f"\nGains saved to: {filepath}")


def save_gains_to_npz(
    F1: np.ndarray, F2: np.ndarray, P: np.ndarray,
    L_hat_pinv: np.ndarray,
    B1: np.ndarray, B2: np.ndarray,
    s_star: np.ndarray,
    Z0: float, Z_min: float, Z_max: float,
    Ts: float,
    filepath: str
):
    """Save discrete LMI-solved gains to NPZ file."""
    
    np.savez(
        filepath,
        F1=F1, F2=F2, P=P,
        L_hat_pinv=L_hat_pinv,
        B1=B1, B2=B2,
        s_star=s_star,
        Z0=np.array([Z0]),
        Z_min=np.array([Z_min]),
        Z_max=np.array([Z_max]),
        Ts=np.array([Ts])
    )
    
    print(f"Gains saved to: {filepath}")


def print_matrices_for_cpp(
    F1: np.ndarray, F2: np.ndarray, P: np.ndarray,
    L_hat_pinv: np.ndarray
):
    """Print matrices in C++ array initialization format."""
    
    print("\n" + "="*70)
    print("C++ MATRIX INITIALIZATION CODE")
    print("="*70)
    
    def format_matrix_cpp(name: str, M: np.ndarray, var_type: str = "double"):
        rows, cols = M.shape
        print(f"\n// {name}: {rows}x{cols}")
        print(f"const {var_type} {name}[{rows}][{cols}] = {{")
        for i in range(rows):
            row_str = ", ".join([f"{M[i,j]:+.10e}" for j in range(cols)])
            comma = "," if i < rows - 1 else ""
            print(f"    {{{row_str}}}{comma}")
        print("};")
    
    format_matrix_cpp("F1", F1)
    format_matrix_cpp("F2", F2)
    format_matrix_cpp("P", P)
    format_matrix_cpp("L_hat_pinv", L_hat_pinv)


def main():
    """Main function to solve discrete LMIs and generate gains."""
    
    print("="*70)
    print("DISCRETE-TIME TS-PDC IBVS LMI GAIN SOLVER")
    print("4 Point Features (k=8) → Reduced State (n=6)")
    print("Model: x(k+1) = x(k) + Ts * B(Z) * u(k)")
    print("="*70)
    
    # ============================================
    # CONFIGURATION - Match your simulation setup
    # ============================================
    
    half_norm = 0.0625  # cube_size/2 / desired_Z = 0.025/0.4
    
    s_star = np.array([
        [-half_norm, -half_norm],
        [ half_norm, -half_norm],
        [ half_norm,  half_norm],
        [-half_norm,  half_norm],
    ])
    
    # Depth parameters
    Z0 = 0.5
    Z_min = 0.2
    Z_max = 1.0
    
    # Sampling period
    Ts = 0.033  # ~30Hz
    
    print(f"\nConfiguration:")
    print(f"  Desired features s*:\n{s_star}")
    print(f"  Nominal depth Z0: {Z0} m")
    print(f"  Depth range: [{Z_min}, {Z_max}] m")
    print(f"  Sampling period Ts: {Ts} s")
    
    # ============================================
    # COMPUTE VERTEX MATRICES
    # ============================================
    
    L_hat, L_hat_pinv, B1, B2 = compute_vertex_matrices(s_star, Z0, Z_min, Z_max)
    
    print(f"\nL_hat (8x6) at s*, Z0={Z0}:")
    print(L_hat)
    
    print(f"\nL_hat_pinv (6x8):")
    print(L_hat_pinv)
    
    print(f"\nB1 = L_hat_pinv @ L(s*, Z_max) [far vertex, 6x6]:")
    print(B1)
    print(f"  Eigenvalues: {np.linalg.eigvals(B1)}")
    
    print(f"\nB2 = L_hat_pinv @ L(s*, Z_min) [close vertex, 6x6]:")
    print(B2)
    print(f"  Eigenvalues: {np.linalg.eigvals(B2)}")
    
    # ============================================
    # SOLVE DISCRETE LMIs
    # ============================================
    
    print("\n" + "="*70)
    print("SOLVING DISCRETE LMI SYNTHESIS PROBLEM")
    print("="*70)
    
    F1, F2, P, feasible = solve_ts_pdc_lmi_discrete(B1, B2, Ts, eps=1e-6, verbose=True)
    
    if not feasible:
        print("\n✗ Discrete LMI problem is INFEASIBLE!")
        print("  Try adjusting Ts, depth range, or feature configuration.")
        return
    
    print("\n✓ Discrete LMI problem SOLVED!")
    
    print(f"\nF1 (6x6) - Far vertex gain:")
    print(F1)
    
    print(f"\nF2 (6x6) - Close vertex gain:")
    print(F2)
    
    print(f"\nP (6x6) - Lyapunov matrix:")
    print(P)
    print(f"  Eigenvalues: {np.linalg.eigvalsh(P)}")
    print(f"  Condition number: {np.linalg.cond(P):.2e}")
    
    # ============================================
    # VERIFY CONDITIONS
    # ============================================
    
    verify_lyapunov_conditions_discrete(B1, B2, F1, F2, P, Ts)
    
    max_eig_mag, _ = verify_stability_discrete(B1, B2, F1, F2, P, Ts, Z_min, Z_max)
    
    # ============================================
    # SAVE RESULTS
    # ============================================
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, '..', 'config')
    os.makedirs(config_dir, exist_ok=True)
    
    yaml_path = os.path.join(config_dir, 'ts_pdc_lmi_discrete_gains.yaml')
    npz_path = os.path.join(config_dir, 'ts_pdc_lmi_discrete_gains.npz')
    
    save_gains_to_yaml(F1, F2, P, L_hat_pinv, B1, B2, s_star, Z0, Z_min, Z_max, Ts, yaml_path)
    save_gains_to_npz(F1, F2, P, L_hat_pinv, B1, B2, s_star, Z0, Z_min, Z_max, Ts, npz_path)
    
    print_matrices_for_cpp(F1, F2, P, L_hat_pinv)
    
    print("\n" + "="*70)
    print("DONE - Discrete gains ready for ur5e_ts_ibvs_lmi_discrete_node")
    print("="*70)


if __name__ == "__main__":
    main()
