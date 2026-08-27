#!/usr/bin/env python3
"""
TS-PDC IBVS LMI Synthesis with Reduced Error (Chaumette's Approach)

This script solves the LMI problem for TS-PDC IBVS using the reduced error:
    x = L_hat_pinv @ e  ∈ R^6
where:
    e ∈ R^8 is the raw feature error (4 points × 2 coordinates)
    L_hat_pinv ∈ R^{6×8} is the pseudoinverse of L_e at (s*, Z0)

Key insight: 
    Using e ∈ R^8 directly with u ∈ R^6 leads to rank-deficient A_cl ∈ R^{8×8}
    with at least 2 zero eigenvalues → LMIs infeasible.
    
    The reduced error x ∈ R^6 gives B(Z) = L_hat_pinv @ L_e(Z) ∈ R^{6×6}
    which is generically full rank → LMIs become feasible!

TS Model (in reduced coordinates x):
    x_dot = B(Z) @ u
    
    where B(Z) = h_far(Z) * B1 + h_close(Z) * B2
    
    B1 = L_hat_pinv @ L_e(Z_max)   (far vertex, 6×6)
    B2 = L_hat_pinv @ L_e(Z_min)   (close vertex, 6×6)

PDC Controller:
    u = -(h_far * F1 + h_close * F2) @ x

Closed-loop:
    x_dot = -sum_i sum_j h_i h_j B_i F_j x = sum_i sum_j h_i h_j G_ij x
    where G_ij = -B_i @ F_j

LMI Conditions (Tanaka & Wang Relaxed Stability for r=2):
    sym(P @ G_ii) < 0  for i=1,2
    sym(P @ (G_12 + G_21)/2) < 0
    
    With G_ij = -B_i F_j and A_i = 0:
    sym(P B_i F_i) > 0  (diagonal)
    sym(P (B_1 F_2 + B_2 F_1)/2) > 0  (cross)

Convex LMI (change of variables X = P^{-1}, M_i = F_i X):
    X > 0
    sym(B_1 M_1) > 2γX
    sym(B_2 M_2) > 2γX  
    sym(B_1 M_2 + B_2 M_1) > 2γX

Recovery: P = X^{-1}, F_i = M_i @ P

Author: TS-IBVS Project (Corrected for k>6 features)
Date: 2026
"""

import numpy as np
import cvxpy as cp
from typing import Tuple, Optional, Dict, List
import json
import argparse
import matplotlib.pyplot as plt


def build_interaction_matrix_point(x: float, y: float, Z: float) -> np.ndarray:
    """
    Build the interaction matrix for one point feature.
    
    L(x,y,Z) = [[-1/Z,    0,  x/Z,   xy,    -(1+x²),  y  ],
                [  0,  -1/Z,  y/Z, 1+y²,     -xy,   -x  ]]
    
    Args:
        x, y: normalized image coordinates
        Z: depth value
        
    Returns:
        L: 2×6 interaction matrix
    """
    L = np.array([
        [-1/Z,    0,  x/Z,  x*y,      -(1 + x**2),  y],
        [   0, -1/Z,  y/Z,  1 + y**2,     -x*y,    -x]
    ])
    return L


def build_interaction_matrix_4points(points: np.ndarray, Z: float) -> np.ndarray:
    """
    Build the stacked interaction matrix for 4 point features.
    
    Args:
        points: 4×2 array of normalized image coordinates [[x1,y1], ..., [x4,y4]]
        Z: depth value (scalar, same for all points)
        
    Returns:
        Le: 8×6 interaction matrix
    """
    Le = np.vstack([build_interaction_matrix_point(points[i, 0], points[i, 1], Z) 
                    for i in range(4)])
    return Le


def compute_desired_features(cube_size: float, Z_desired: float) -> np.ndarray:
    """
    Compute desired normalized image coordinates for 4 corner points.
    
    Args:
        cube_size: size of the target square in meters
        Z_desired: desired servoing depth in meters
        
    Returns:
        s_star: 4×2 array of (x*, y*) coordinates
    """
    half = cube_size / 2.0
    
    # 3D points in object frame (Z=0 plane)
    points_3D = np.array([
        [-half, -half, 0],  # Top-left
        [+half, -half, 0],  # Top-right
        [+half, +half, 0],  # Bottom-right
        [-half, +half, 0],  # Bottom-left
    ])
    
    # Project to normalized image coordinates at Z_desired
    s_star = points_3D[:, :2] / Z_desired
    
    return s_star


def solve_ts_pdc_lmi_reduced(
    B1: np.ndarray, 
    B2: np.ndarray,
    gamma: float = 0.1,
    eps: float = 1e-6,
    solver: str = 'MOSEK',
    verbose: bool = True
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """
    Solve TS-PDC LMI synthesis with reduced error (6D state).
    
    Find X, M1, M2 such that:
        X > eps*I
        sym(B1 @ M1) > 2*gamma*X + eps*I
        sym(B2 @ M2) > 2*gamma*X + eps*I
        sym(B1 @ M2 + B2 @ M1) > 2*gamma*X + eps*I
    
    Then recover P = X^{-1}, F1 = M1 @ P, F2 = M2 @ P
    
    Args:
        B1: 6×6 projected input matrix at Z_max (far)
        B2: 6×6 projected input matrix at Z_min (close)
        gamma: minimum decay rate
        eps: small positive constant for strict inequalities
        solver: CVXPY solver name
        verbose: print progress
        
    Returns:
        F1: 6×6 controller gain for far rule
        F2: 6×6 controller gain for close rule
        P: 6×6 Lyapunov matrix
        info: dictionary with solver info
    """
    n = 6  # State dimension (reduced error)
    
    if B1.shape != (n, n) or B2.shape != (n, n):
        raise ValueError(f"B1 and B2 must be {n}×{n}, got {B1.shape} and {B2.shape}")
    
    if verbose:
        print("=" * 70)
        print("TS-PDC LMI Synthesis (Reduced Error, x ∈ R^6)")
        print("=" * 70)
        print(f"\nDimensions: n={n} (reduced error = velocity)")
        print(f"B1 (far)  shape: {B1.shape}")
        print(f"B2 (close) shape: {B2.shape}")
        print(f"Solver: {solver}, gamma: {gamma}, eps: {eps}")
        
        print("\nB1 (far vertex, Z_max):")
        print(B1)
        print("\nB2 (close vertex, Z_min):")
        print(B2)
        
        # Check rank
        print(f"\nrank(B1) = {np.linalg.matrix_rank(B1)}")
        print(f"rank(B2) = {np.linalg.matrix_rank(B2)}")
    
    # Decision variables
    X = cp.Variable((n, n), symmetric=True)
    M1 = cp.Variable((n, n))
    M2 = cp.Variable((n, n))
    
    # Helper: sym(Q) = Q + Q.T
    def sym(Q):
        return Q + Q.T
    
    # Identity matrix
    I = np.eye(n)
    
    # Constraints
    constraints = [
        # X > eps * I (positive definite)
        X >> eps * I,
        
        # LMI_1: sym(B1 @ M1) > 2*gamma*X (diagonal, far)
        sym(B1 @ M1) >> 2*gamma*X + eps*I,
        
        # LMI_2: sym(B2 @ M2) > 2*gamma*X (diagonal, close)
        sym(B2 @ M2) >> 2*gamma*X + eps*I,
        
        # LMI_12: sym(B1 @ M2 + B2 @ M1) > 2*gamma*X (cross-term)
        sym(B1 @ M2 + B2 @ M1) >> 2*gamma*X + eps*I,
    ]
    
    # Objective: minimize trace(X) for numerical conditioning
    # Or use feasibility (minimize 0)
    objective = cp.Minimize(cp.trace(X))
    
    problem = cp.Problem(objective, constraints)
    
    # Try to solve
    try:
        if solver == 'MOSEK':
            problem.solve(solver=cp.MOSEK, verbose=verbose)
        elif solver == 'SCS':
            problem.solve(solver=cp.SCS, verbose=verbose, eps=1e-8)
        elif solver == 'CVXOPT':
            problem.solve(solver=cp.CVXOPT, verbose=verbose)
        else:
            problem.solve(verbose=verbose)
    except Exception as e:
        print(f"Solver {solver} failed: {e}")
        # Try SCS as fallback
        try:
            problem.solve(solver=cp.SCS, verbose=verbose, eps=1e-8)
        except:
            pass
    
    info = {
        'status': problem.status,
        'optimal_value': problem.value if problem.value is not None else float('inf'),
        'solver': solver,
    }
    
    if problem.status in ['optimal', 'optimal_inaccurate']:
        X_val = X.value
        M1_val = M1.value
        M2_val = M2.value
        
        # Recover P and gains
        P = np.linalg.inv(X_val)
        F1 = M1_val @ P
        F2 = M2_val @ P
        
        if verbose:
            print("\n" + "=" * 70)
            print("LMI SOLVED SUCCESSFULLY!")
            print("=" * 70)
            
            print("\nP (Lyapunov matrix):")
            print(P)
            
            print("\nF1 (gain for far rule, Z_max):")
            print(F1)
            
            print("\nF2 (gain for close rule, Z_min):")
            print(F2)
            
            # Verify LMIs
            print("\n" + "-" * 70)
            print("LMI Verification:")
            print("-" * 70)
            
            def check_posdef(name, M):
                eigvals = np.linalg.eigvalsh(M)
                min_eig = np.min(eigvals)
                print(f"  {name}: min_eig = {min_eig:.6e} {'> 0 ✓' if min_eig > 0 else '≤ 0 ✗'}")
                return min_eig > 0
            
            check_posdef("P", P)
            check_posdef("sym(B1 F1) - 2γI", sym(B1 @ F1) - 2*gamma*I)
            check_posdef("sym(B2 F2) - 2γI", sym(B2 @ F2) - 2*gamma*I)
            check_posdef("sym(B1 F2 + B2 F1) - 2γI", sym(B1 @ F2 + B2 @ F1) - 2*gamma*I)
        
        info['P'] = P.tolist()
        info['F1'] = F1.tolist()
        info['F2'] = F2.tolist()
        
        return F1, F2, P, info
    else:
        if verbose:
            print(f"\nLMI INFEASIBLE: {problem.status}")
        return None, None, None, info


def verify_stability_grid(
    F1: np.ndarray, 
    F2: np.ndarray,
    L_hat_pinv: np.ndarray,
    s_star: np.ndarray,
    Z_min: float,
    Z_max: float,
    n_points: int = 50,
    verbose: bool = True
) -> Dict:
    """
    Verify closed-loop stability across depth range.
    
    Args:
        F1, F2: Controller gains
        L_hat_pinv: Pseudoinverse of L_hat (6×8)
        s_star: Desired features (4×2)
        Z_min, Z_max: Depth range
        n_points: Number of grid points
        verbose: Print details
        
    Returns:
        results: Dictionary with stability analysis
    """
    Z_grid = np.linspace(Z_min, Z_max, n_points)
    
    alpha_min = 1.0 / Z_max
    alpha_max = 1.0 / Z_min
    
    results = {
        'Z': [],
        'h_far': [],
        'h_close': [],
        'max_real_eig': [],
        'all_stable': True,
    }
    
    if verbose:
        print("\n" + "=" * 70)
        print("Stability Verification Across Depth Range")
        print("=" * 70)
        print(f"Z range: [{Z_min:.2f}, {Z_max:.2f}] m")
        print(f"α range: [{alpha_min:.2f}, {alpha_max:.2f}]")
    
    for Z in Z_grid:
        # Compute membership weights
        alpha = 1.0 / Z
        h_close = (alpha - alpha_min) / (alpha_max - alpha_min)
        h_far = 1.0 - h_close
        h_close = np.clip(h_close, 0, 1)
        h_far = np.clip(h_far, 0, 1)
        
        # Blended gain
        F_Z = h_far * F1 + h_close * F2
        
        # Actual B(Z) at this depth
        L_e_Z = build_interaction_matrix_4points(s_star, Z)
        B_Z = L_hat_pinv @ L_e_Z
        
        # Closed-loop matrix
        A_cl = -B_Z @ F_Z
        
        # Eigenvalues
        eigvals = np.linalg.eigvals(A_cl)
        max_real = np.max(np.real(eigvals))
        
        results['Z'].append(Z)
        results['h_far'].append(h_far)
        results['h_close'].append(h_close)
        results['max_real_eig'].append(max_real)
        
        if max_real >= 0:
            results['all_stable'] = False
            if verbose:
                print(f"  WARNING: Unstable at Z={Z:.3f}m, max Re(λ)={max_real:.6f}")
    
    if verbose:
        if results['all_stable']:
            print(f"  ✓ System is stable for all Z ∈ [{Z_min:.2f}, {Z_max:.2f}]")
            print(f"  Max Re(eigenvalue) over all Z: {np.max(results['max_real_eig']):.6e}")
        else:
            print(f"  ✗ System has unstable regions!")
    
    return results


def plot_stability_analysis(results: Dict, save_path: str = None):
    """
    Plot stability analysis results.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Max eigenvalue vs depth
    ax1 = axes[0]
    ax1.plot(results['Z'], results['max_real_eig'], 'b-', linewidth=2)
    ax1.axhline(0, color='r', linestyle='--', linewidth=1, label='Stability boundary')
    ax1.set_xlabel('Depth Z (m)')
    ax1.set_ylabel('Max Re(eigenvalue)')
    ax1.set_title('Closed-Loop Stability Check')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Membership functions
    ax2 = axes[1]
    ax2.plot(results['Z'], results['h_far'], 'g-', linewidth=2, label='h_far (Z_max)')
    ax2.plot(results['Z'], results['h_close'], 'r-', linewidth=2, label='h_close (Z_min)')
    ax2.set_xlabel('Depth Z (m)')
    ax2.set_ylabel('Membership weight')
    ax2.set_title('TS Membership Functions')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def generate_cpp_code(
    F1: np.ndarray, 
    F2: np.ndarray, 
    P: np.ndarray,
    L_hat_pinv: np.ndarray,
    cube_size: float,
    Z0: float,
    Z_min: float,
    Z_max: float,
    gamma: float
) -> str:
    """
    Generate C++ code for the LMI-synthesized gains.
    """
    code = f"""
// ============================================================
// LMI-Synthesized TS-PDC Gains (Reduced Error)
// Generated by solve_ts_lmi_reduced.py
// ============================================================
// 
// Parameters:
//   cube_size = {cube_size} m
//   Z0 (nominal for L_hat) = {Z0} m
//   Z_min = {Z_min} m
//   Z_max = {Z_max} m
//   gamma (decay rate) = {gamma}
//
// Reduced error: x = L_hat_pinv @ e  ∈ R^6
// Control law: u = -(h_far*F1 + h_close*F2) @ x
// ============================================================

// L_hat_pinv: 6×8 (pseudoinverse at Z0)
static const double L_hat_pinv[6][8] = {{
"""
    for i in range(6):
        row = ", ".join([f"{L_hat_pinv[i, j]:12.8f}" for j in range(8)])
        code += f"    {{ {row} }}"
        code += ",\n" if i < 5 else "\n"
    code += "};\n\n"
    
    code += f"""// F1: 6×6 (gain for far rule, Z_max={Z_max})
static const double F1[6][6] = {{
"""
    for i in range(6):
        row = ", ".join([f"{F1[i, j]:12.8f}" for j in range(6)])
        code += f"    {{ {row} }}"
        code += ",\n" if i < 5 else "\n"
    code += "};\n\n"
    
    code += f"""// F2: 6×6 (gain for close rule, Z_min={Z_min})
static const double F2[6][6] = {{
"""
    for i in range(6):
        row = ", ".join([f"{F2[i, j]:12.8f}" for j in range(6)])
        code += f"    {{ {row} }}"
        code += ",\n" if i < 5 else "\n"
    code += "};\n\n"
    
    code += f"""// P: 6×6 (Lyapunov matrix for V = x^T P x)
static const double P_lyap[6][6] = {{
"""
    for i in range(6):
        row = ", ".join([f"{P[i, j]:12.8f}" for j in range(6)])
        code += f"    {{ {row} }}"
        code += ",\n" if i < 5 else "\n"
    code += "};\n"
    
    return code


def main():
    parser = argparse.ArgumentParser(
        description='Solve TS-PDC LMI with reduced error for IBVS')
    parser.add_argument('--cube_size', type=float, default=0.05,
                        help='Target cube size in meters (default: 0.05)')
    parser.add_argument('--Z_desired', type=float, default=0.4,
                        help='Desired servoing depth (default: 0.4)')
    parser.add_argument('--Z0', type=float, default=0.5,
                        help='Nominal depth for L_hat computation (default: 0.5)')
    parser.add_argument('--Z_min', type=float, default=0.2,
                        help='Minimum depth / close vertex (default: 0.2)')
    parser.add_argument('--Z_max', type=float, default=1.0,
                        help='Maximum depth / far vertex (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Minimum decay rate (default: 0.1)')
    parser.add_argument('--solver', type=str, default='SCS',
                        choices=['MOSEK', 'SCS', 'CVXOPT', 'ECOS'],
                        help='CVXPY solver (default: SCS)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for C++ code (default: None)')
    parser.add_argument('--plot', type=str, default=None,
                        help='Output file for stability plot (default: None)')
    parser.add_argument('--json', type=str, default=None,
                        help='Output file for JSON results (default: None)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("TS-PDC IBVS LMI Synthesis (Reduced Error Approach)")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  cube_size = {args.cube_size} m")
    print(f"  Z_desired = {args.Z_desired} m")
    print(f"  Z0 (nominal) = {args.Z0} m")
    print(f"  Z_min = {args.Z_min} m")
    print(f"  Z_max = {args.Z_max} m")
    print(f"  gamma = {args.gamma}")
    
    # Step 1: Compute desired features
    s_star = compute_desired_features(args.cube_size, args.Z_desired)
    print(f"\nDesired features s* (normalized image coords):")
    for i, (x, y) in enumerate(s_star):
        print(f"  Point {i}: ({x:.6f}, {y:.6f})")
    
    # Step 2: Compute L_hat and L_hat_pinv at Z0
    L_hat = build_interaction_matrix_4points(s_star, args.Z0)
    L_hat_pinv = np.linalg.pinv(L_hat)
    
    print(f"\nL_hat (8×6 at Z0={args.Z0}):")
    print(L_hat)
    
    print(f"\nL_hat_pinv (6×8):")
    print(L_hat_pinv)
    
    # Verify pseudoinverse
    LpL = L_hat_pinv @ L_hat
    print(f"\nL_hat_pinv @ L_hat (should be ≈ I_6):")
    print(LpL)
    print(f"  max|LpL - I| = {np.max(np.abs(LpL - np.eye(6))):.2e}")
    
    # Step 3: Compute vertex matrices B1, B2
    L_e_far = build_interaction_matrix_4points(s_star, args.Z_max)
    L_e_close = build_interaction_matrix_4points(s_star, args.Z_min)
    
    B1 = L_hat_pinv @ L_e_far    # Far vertex (6×6)
    B2 = L_hat_pinv @ L_e_close  # Close vertex (6×6)
    
    print(f"\nB1 = L_hat_pinv @ L_e(Z_max) (6×6, far vertex):")
    print(B1)
    print(f"rank(B1) = {np.linalg.matrix_rank(B1)}")
    
    print(f"\nB2 = L_hat_pinv @ L_e(Z_min) (6×6, close vertex):")
    print(B2)
    print(f"rank(B2) = {np.linalg.matrix_rank(B2)}")
    
    # Step 4: Solve LMIs
    F1, F2, P, info = solve_ts_pdc_lmi_reduced(
        B1, B2, 
        gamma=args.gamma,
        solver=args.solver,
        verbose=True
    )
    
    if F1 is None:
        print("\nFailed to solve LMIs!")
        return 1
    
    # Step 5: Verify stability across depth range
    results = verify_stability_grid(
        F1, F2, L_hat_pinv, s_star, 
        args.Z_min, args.Z_max,
        verbose=True
    )
    
    # Step 6: Generate outputs
    if args.output:
        cpp_code = generate_cpp_code(
            F1, F2, P, L_hat_pinv,
            args.cube_size, args.Z0, args.Z_min, args.Z_max, args.gamma
        )
        with open(args.output, 'w') as f:
            f.write(cpp_code)
        print(f"\nC++ code saved to {args.output}")
    
    if args.plot:
        plot_stability_analysis(results, args.plot)
    
    if args.json:
        output = {
            'parameters': {
                'cube_size': args.cube_size,
                'Z_desired': args.Z_desired,
                'Z0': args.Z0,
                'Z_min': args.Z_min,
                'Z_max': args.Z_max,
                'gamma': args.gamma,
            },
            's_star': s_star.tolist(),
            'L_hat_pinv': L_hat_pinv.tolist(),
            'B1': B1.tolist(),
            'B2': B2.tolist(),
            'F1': F1.tolist(),
            'F2': F2.tolist(),
            'P': P.tolist(),
            'stability': {
                'all_stable': results['all_stable'],
                'max_eigenvalue': float(np.max(results['max_real_eig'])),
            }
        }
        with open(args.json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nJSON results saved to {args.json}")
    
    # Print summary for copy-paste into C++ node
    print("\n" + "=" * 70)
    print("SUMMARY FOR C++ NODE")
    print("=" * 70)
    print(generate_cpp_code(
        F1, F2, P, L_hat_pinv,
        args.cube_size, args.Z0, args.Z_min, args.Z_max, args.gamma
    ))
    
    return 0


if __name__ == '__main__':
    exit(main())
