#!/usr/bin/env python3
"""
tp_model_ud_build.py — TP Model Transformation with Uniform Design (Wang 2020)

Implements the efficient TP model transformation from:
  Wang et al. (2020) "An Efficient TP Model Transformation Algorithm for Robust
  Visual Servoing in the Presence of Uncertain Data"
  Acta Polytechnica Hungarica, Vol.17, No.6, pp.155-169

Key difference from tp_model_build.py (Wang 2014, regular grid):
  - Wang 2020 uses UNIFORM DESIGN (UD) sampling instead of regular grid
  - The image Jacobian includes 5 uncertain parameters:
      p = (u_i, v_i, 1/z_i, k_x, k_y)
    instead of 3: (u_c, v_c, 1/Z)
  - UD dramatically reduces the number of sample points while maintaining
    space-filling uniformity, via the Good Lattice Point (GLP) method
    - After HOSVD on the UD-sampled tensor, Wang 2020 reports a reduction
        from 162 regular-grid vertices to 5 vertices for its chosen bounds.
        The rank produced by this script depends on the selected uncertainty
        ranges and singular-value threshold; the depth-only model used in the
        manuscript keeps 2 vertices per feature point.

Algorithm (Wang 2020 Section 3.2):
  STEP 1: Generate UD table U_M(M^s) using the GLP method (s=5 factors)
  STEP 2: Map UD levels to the transformation space Omega, sample J(p)
          at M uniform-design points -> tensor T in R^{M x 2 x 6}
  STEP 3: HOSVD on the first dimension of T, discard zero/small SVs
          -> core tensor, weight matrix U_1 in R^{M x T}, T <= M
  STEP 4: Apply SN/NN/NO normalization to get convex vertex matrices

For the UR5e eye-to-hand simulation we adapt:
  - Camera intrinsic bounds from calibration uncertainty
  - Depth range from workspace geometry
  - Feature coordinate ranges from FOV

Usage:
  python3 tp_model_ud_build.py [--output config/tp_ud_vertices.yaml]
"""

import argparse
import numpy as np
import yaml
import os
import sys


# =============================================================================
# Uniform Design Table Generation (Good Lattice Point method)
# =============================================================================

def is_coprime(a, b):
    """Check if a and b are coprime."""
    from math import gcd
    return gcd(a, b) == 1


def good_lattice_point(M, s):
    """
    Generate a Uniform Design table U_M(M^s) via the Good Lattice Point method.

    Reference: Fang & Wang (1994) "Number-theoretic Methods in Statistics"
               Cui et al. (2016) ref [27] in Wang 2020

    The GLP method finds a generator vector h = (h_1, ..., h_s) where each h_j
    is coprime with M. The UD table entries are:
        u_{ij} = (i * h_j) mod M,  i = 1..M, j = 1..s

    The generator that minimizes the centered L2-discrepancy (CD2) is selected.

    Args:
        M: number of runs (levels per factor)
        s: number of factors (dimensions)

    Returns:
        table: M x s array with values in {1, 2, ..., M} (1-indexed levels)
        CD2: centered L2-discrepancy of the selected design
    """
    # Find all integers coprime with M in [1, M)
    coprimes = [h for h in range(1, M) if is_coprime(h, M)]

    best_cd2 = np.inf
    best_table = None

    if s == 1:
        table = np.arange(1, M + 1, dtype=int).reshape(M, 1)
        return table, centered_l2_discrepancy(table, M)

    # Search Korobov-type GLP generators h=(1,a,a^2,...,a^{s-1}) mod M.
    # Reject degenerate generators such as a=1, which make all columns equal.
    for h_base in coprimes:
        # Generator vector: h = (h_base^0, h_base^1, ..., h_base^(s-1)) mod M
        h = np.zeros(s, dtype=int)
        for j in range(s):
            h[j] = pow(h_base, j, M)

        if np.any(h == 0) or len(set(h.tolist())) < s:
            continue

        # Build the design table
        table = np.zeros((M, s), dtype=int)
        for i in range(M):
            for j in range(s):
                table[i, j] = ((i + 1) * h[j]) % M
                if table[i, j] == 0:
                    table[i, j] = M

        # Compute centered L2-discrepancy (CD2)
        cd2 = centered_l2_discrepancy(table, M)

        if cd2 < best_cd2:
            best_cd2 = cd2
            best_table = table.copy()

    if best_table is None:
        raise RuntimeError(f"Could not construct a non-degenerate GLP table for M={M}, s={s}")

    return best_table, best_cd2


def centered_l2_discrepancy(table, M):
    """
    Compute the centered L2-discrepancy (CD2) of a design table.

    Lower CD2 = more uniform space-filling.
    Vectorized formula from Hickernell (1998).
    """
    n, s = table.shape
    # Normalize to [0, 1]
    X = (table - 0.5) / M

    # Term 1
    term1 = (13.0 / 12.0) ** s

    # Term 2: vectorized
    abs_centered = np.abs(X - 0.5)
    sq_centered = (X - 0.5) ** 2
    prod_per_point = np.prod(1.0 + 0.5 * abs_centered - 0.5 * sq_centered, axis=1)
    term2 = -(2.0 / n) * np.sum(prod_per_point)

    # Term 3: vectorized with broadcasting
    # For each pair (i,k): prod_j(1 + 0.5|x_ij-0.5| + 0.5|x_kj-0.5| - 0.5|x_ij-x_kj|)
    term3 = 0.0
    for i in range(n):
        diff = np.abs(X[i:i+1, :] - X)  # (n, s)
        factor = 1.0 + 0.5 * abs_centered[i:i+1, :] + 0.5 * abs_centered - 0.5 * diff
        term3 += np.sum(np.prod(factor, axis=1))
    term3 = term3 / (n * n)

    val = term1 + term2 + term3
    cd2 = np.sqrt(max(val, 0.0))
    return cd2


# =============================================================================
# 5-Parameter Uncertain Image Jacobian (Wang 2020 Eq. 13)
# =============================================================================

def build_uncertain_jacobian(ui, vi, inv_zi, kx, ky):
    """
    Build image Jacobian with uncertain parameters (Wang 2020 Eq. 13).

    Parameters include u0_hat absorbed into u_i, v0_hat into v_i:
      p = (u_i, v_i, 1/z_i, k_x, k_y)

    Wang 2020 Eq.(13):
      J_si = [[-kx/zi,     0,   (ui-u0)/zi,  (ui-u0)(vi-v0)/ky,  -(kx+(ui-u0)^2/kx),  kx(vi-v0)/ky],
              [    0,  -ky/zi,  (vi-v0)/zi,   ky+(vi-v0)^2/ky,   -(ui-u0)(vi-v0)/kx,  -ky(ui-u0)/kx]]

    Here ui, vi are already centered (u_i - u_0 absorbed), so we use them directly.
    This is the pixel-coordinate Jacobian with uncertain k_x, k_y.
    """
    J = np.zeros((2, 6))
    # Row 1: du/dt
    J[0, 0] = -kx * inv_zi                           # -kx/z
    J[0, 1] = 0.0
    J[0, 2] = ui * inv_zi                             # u/z
    J[0, 3] = ui * vi / ky                            # uv/ky
    J[0, 4] = -(kx + ui * ui / kx)                    # -(kx + u^2/kx)
    J[0, 5] = kx * vi / ky                            # kx*v/ky
    # Row 2: dv/dt
    J[1, 0] = 0.0
    J[1, 1] = -ky * inv_zi                            # -ky/z
    J[1, 2] = vi * inv_zi                             # v/z
    J[1, 3] = ky + vi * vi / ky                       # ky + v^2/ky
    J[1, 4] = -(ui * vi / kx)                         # -uv/kx
    J[1, 5] = -ky * ui / kx                           # -ky*u/kx
    return J


# =============================================================================
# HOSVD on 1D-sampled tensor (Wang 2020 Step 3)
# =============================================================================

def hosvd_1d(tensor_2d, sv_threshold=1e-10):
    """
    HOSVD on the first dimension of tensor T in R^{M x D}.

    Wang 2020 STEP 3: Apply HOSVD to the first dimension of J_si^D.
    Discard zero or small singular values sigma_k.

    T_(1) = U_1 * Sigma * V^T
    Keep T columns where sigma > threshold * sigma_max.

    Returns:
        core: R^{T x D}  (T <= M)
        U1: R^{M x T}    (weight coefficient matrix)
        sigmas: retained singular values
        rank_T: number of retained components
    """
    M, D = tensor_2d.shape

    U, S, Vt = np.linalg.svd(tensor_2d, full_matrices=False)

    # Determine rank: keep SVs above threshold
    rank_T = int(np.sum(S > sv_threshold * S[0]))
    rank_T = max(rank_T, 1)

    print(f"  HOSVD: M={M} samples, D={D} features")
    print(f"  Singular values: {S[:min(10, len(S))]}")
    print(f"  Rank T = {rank_T} (threshold={sv_threshold})")
    if rank_T < len(S):
        print(f"  Gap: sigma_{rank_T-1}={S[rank_T-1]:.4e} -> "
              f"sigma_{rank_T}={S[min(rank_T, len(S)-1)]:.4e}")

    U1 = U[:, :rank_T]
    sigmas = S[:rank_T]

    # Core: S = U1^T @ tensor_2d  (T x D)
    core = U1.T @ tensor_2d

    return core, U1, sigmas, rank_T


def apply_no_1d(U1, tensor_2d):
    """
    Apply NO + SN transformation to the 1D weight matrix from HOSVD.

    Wang 2020: "Further transformations like SN, NN, NO or INO-RNO could be
    executed to get better application effect."

    Wang 2014 Eq.(9) convexity conditions:
      (a) w_{n,i_n}(p_n) >= 0  for all n, i_n, p_n  (NN)
      (b) sum_{i_n} w_{n,i_n}(p_n) = 1  for all n, p_n  (SN)

    Step 1 (NO): shift each column of U1 to [0, 1]
    Step 2 (SN): normalize each row to sum to 1
    Then recompute core via pseudoinverse.

    Returns:
        core_no: R^{T x D}
        W1: R^{M x T} (weight coefficients, values in [0,1], rows sum to 1)
    """
    M, T = U1.shape
    W1 = np.zeros_like(U1)

    # Step 1: NO (Non-Negative + shift to [0, 1])
    for r in range(T):
        col = U1[:, r]
        d_r = col.max() - col.min()
        if d_r > 1e-15:
            W1[:, r] = (col - col.min()) / d_r
        else:
            W1[:, r] = np.ones(M)

    # Step 2: SN (Sum Normalization) — enforce sum(w_r) = 1 per row
    # This is required by Wang 2014 Eq.(9) for convexity
    row_sums = W1.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-15)  # avoid division by zero
    W1 = W1 / row_sums

    cond_w = np.linalg.cond(W1)
    print(f"  NO+SN transform: W1 in [{W1.min():.6f}, {W1.max():.6f}], "
          f"cond={cond_w:.2e}")
    print(f"  Row sums: min={W1.sum(axis=1).min():.6f}, max={W1.sum(axis=1).max():.6f}")

    # Recompute core
    W1_pinv = np.linalg.pinv(W1)
    core_no = W1_pinv @ tensor_2d

    return core_no, W1


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TP Model Transformation with Uniform Design (Wang 2020)")
    parser.add_argument("--output", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "config", "tp_ud_vertices.yaml"))
    # Camera intrinsic parameters (nominal + uncertainty bounds)
    parser.add_argument("--kx_nom", type=float, default=554.25,
                        help="Nominal kx = fx (pixel/m) from camera calibration")
    parser.add_argument("--ky_nom", type=float, default=554.25,
                        help="Nominal ky = fy (pixel/m)")
    parser.add_argument("--kx_min", type=float, default=498.83,
                        help="kx lower bound (10% uncertainty)")
    parser.add_argument("--kx_max", type=float, default=609.68,
                        help="kx upper bound (10% uncertainty)")
    parser.add_argument("--ky_min", type=float, default=498.83,
                        help="ky lower bound (10% uncertainty)")
    parser.add_argument("--ky_max", type=float, default=609.68,
                        help="ky upper bound (10% uncertainty)")
    # Feature coordinate ranges (centered pixels)
    # Must cover full FOV: for 640x480 image with u0=320.5, v0=240.5
    # uc ranges from -320.5 to 319.5, vc from -240.5 to 239.5
    parser.add_argument("--uc_min", type=float, default=-320.0)
    parser.add_argument("--uc_max", type=float, default=320.0)
    parser.add_argument("--vc_min", type=float, default=-240.0)
    parser.add_argument("--vc_max", type=float, default=240.0)
    # Depth bounds
    parser.add_argument("--z_min", type=float, default=0.15,
                        help="Minimum depth (m)")
    parser.add_argument("--z_max", type=float, default=1.2,
                        help="Maximum depth (m)")
    parser.add_argument("--cube_size", type=float, default=0.05,
                        help="Target square/cube side length used to place desired feature points")
    parser.add_argument("--desired_Z", type=float, default=0.4,
                        help="Desired depth used to place desired feature points")
    # UD parameters
    parser.add_argument("--M_ud", type=int, default=200,
                        help="UD runs (Wang 2020: 200 for U_200(200^5))")
    parser.add_argument("--sv_threshold", type=float, default=1e-10,
                        help="Singular value threshold for HOSVD")
    # Also build the 3-param version for comparison
    parser.add_argument("--also_3param", action="store_true", default=True,
                        help="Also build 3-param UD model (u, v, 1/Z only)")

    args = parser.parse_args()

    print("=" * 70)
    print("TP Model Transformation with Uniform Design (Wang 2020)")
    print("  Paper: Wang et al., Acta Polytechnica Hungarica, Vol.17, No.6")
    print("=" * 70)

    M = args.M_ud
    inv_z_min = 1.0 / args.z_max
    inv_z_max = 1.0 / args.z_min

    print(f"\nTransformation space Omega (5 parameters):")
    print(f"  u_i in [{args.uc_min}, {args.uc_max}] px (centered)")
    print(f"  v_i in [{args.vc_min}, {args.vc_max}] px (centered)")
    print(f"  1/z in [{inv_z_min:.4f}, {inv_z_max:.4f}] "
          f"(Z in [{args.z_min}, {args.z_max}] m)")
    print(f"  k_x in [{args.kx_min}, {args.kx_max}] px/m")
    print(f"  k_y in [{args.ky_min}, {args.ky_max}] px/m")
    print(f"  UD: U_{M}({M}^5), {M} sample points")

    # =========================================================================
    # STEP 1: Generate Uniform Design table U_M(M^5)
    # =========================================================================
    print(f"\nSTEP 1: Generating UD table U_{M}({M}^5) via GLP method...")
    ud_table, cd2 = good_lattice_point(M, 5)
    print(f"  UD table: {ud_table.shape} (M x s)")
    print(f"  CD2 = {cd2:.6f}")
    if cd2 <= 0.05:
        print(f"  CD2 <= 0.05 — good uniformity (Wang 2020: CD2 <= 0.0207)")

    # =========================================================================
    # STEP 2: Map UD levels to Omega, sample J(p) at UD points
    # =========================================================================
    print(f"\nSTEP 2: Sampling J(p) at {M} UD points...")

    # Define parameter ranges
    ranges = [
        (args.uc_min, args.uc_max),    # u_i
        (args.vc_min, args.vc_max),    # v_i
        (inv_z_min, inv_z_max),        # 1/z_i
        (args.kx_min, args.kx_max),    # k_x
        (args.ky_min, args.ky_max),    # k_y
    ]

    # Map UD table levels to actual parameter values
    # level l in {1, ..., M} maps to: p_min + (p_max - p_min) * (l - 0.5) / M
    params = np.zeros((M, 5))
    for j in range(5):
        p_min, p_max = ranges[j]
        params[:, j] = p_min + (p_max - p_min) * (ud_table[:, j] - 0.5) / M

    # Sample Jacobian at each UD point
    D = 12  # flattened 2x6
    tensor_ud = np.zeros((M, D))
    for i in range(M):
        ui, vi, inv_zi, kx_i, ky_i = params[i]
        tensor_ud[i, :] = build_uncertain_jacobian(
            ui, vi, inv_zi, kx_i, ky_i).flatten()

    Jnorms = np.linalg.norm(tensor_ud, axis=1)
    print(f"  Tensor shape: {tensor_ud.shape} (M x D)")
    print(f"  Jacobian norm range: [{Jnorms.min():.2f}, {Jnorms.max():.2f}]")

    # =========================================================================
    # STEP 3: HOSVD on first dimension
    # =========================================================================
    print(f"\nSTEP 3: HOSVD on first dimension...")
    core, U1, sigmas, rank_T = hosvd_1d(tensor_ud, args.sv_threshold)

    # Verify HOSVD reconstruction
    T_recon = U1 @ core
    err = np.linalg.norm(tensor_ud - T_recon)
    rel_err_hosvd = err / np.linalg.norm(tensor_ud)
    print(f"  HOSVD reconstruction: abs={err:.4e}, rel={rel_err_hosvd:.4e}")

    # =========================================================================
    # STEP 4: NO transformation
    # =========================================================================
    print(f"\nSTEP 4: NO transformation...")
    core_no, W1 = apply_no_1d(U1, tensor_ud)

    # Verify NO reconstruction
    T_recon_no = W1 @ core_no
    err_no = np.linalg.norm(tensor_ud - T_recon_no)
    rel_err_no = err_no / np.linalg.norm(tensor_ud)
    print(f"  NO reconstruction: abs={err_no:.4e}, rel={rel_err_no:.4e}")

    # =========================================================================
    # Extract vertex Jacobians
    # =========================================================================
    R = rank_T
    print(f"\n  Result: {R} vertex Jacobians (Wang 2020 reports 5 for its full 5-parameter setup)")
    vertices = []
    for r in range(R):
        J_v = core_no[r, :].reshape(2, 6)
        vertices.append(J_v)
        print(f"    J_s{r+1}: norm={np.linalg.norm(J_v):.4f}")

    # Weight function check
    row_sums = W1.sum(axis=1)
    print(f"\n  Weight properties (after NO+SN):")
    print(f"    W1 in [{W1.min():.6f}, {W1.max():.6f}]")
    print(f"    Row sums in [{row_sums.min():.4f}, {row_sums.max():.4f}]")
    print(f"    Convexity: NN={'OK' if W1.min() >= -1e-10 else 'FAIL'}, "
          f"SN={'OK' if abs(row_sums.min()-1.0)<1e-6 and abs(row_sums.max()-1.0)<1e-6 else 'FAIL'}")

    # =========================================================================
    # Also build 3-param version for comparison (if requested)
    # =========================================================================
    vertices_3p = None
    rank_3p = None
    W1_3p = None
    core_no_3p = None

    if args.also_3param:
        print(f"\n{'='*70}")
        print(f"BONUS: 3-parameter UD model (u, v, 1/Z) for comparison")
        print(f"{'='*70}")

        ud_table_3p, cd2_3p = good_lattice_point(M, 3)
        print(f"  UD table: {ud_table_3p.shape}, CD2={cd2_3p:.6f}")

        ranges_3p = [
            (args.uc_min, args.uc_max),
            (args.vc_min, args.vc_max),
            (inv_z_min, inv_z_max),
        ]
        params_3p = np.zeros((M, 3))
        for j in range(3):
            p_min, p_max = ranges_3p[j]
            params_3p[:, j] = p_min + (p_max - p_min) * (ud_table_3p[:, j] - 0.5) / M

        kx_nom = args.kx_nom
        ky_nom = args.ky_nom
        tensor_3p = np.zeros((M, D))
        for i in range(M):
            ui, vi, inv_zi = params_3p[i]
            tensor_3p[i, :] = build_uncertain_jacobian(
                ui, vi, inv_zi, kx_nom, ky_nom).flatten()

        print(f"  HOSVD on 3-param tensor...")
        core_3p, U1_3p, sigmas_3p, rank_3p = hosvd_1d(tensor_3p, args.sv_threshold)

        core_no_3p, W1_3p = apply_no_1d(U1_3p, tensor_3p)
        T_recon_3p = W1_3p @ core_no_3p
        rel_err_3p = np.linalg.norm(tensor_3p - T_recon_3p) / np.linalg.norm(tensor_3p)
        print(f"  NO reconstruction: rel={rel_err_3p:.4e}")
        print(f"  Result: {rank_3p} vertices (3-param)")

        vertices_3p = []
        for r in range(rank_3p):
            J_v = core_no_3p[r, :].reshape(2, 6)
            vertices_3p.append(J_v)

    # =========================================================================
    # Also build 1-param version (1/Z only) for comparison
    # =========================================================================
    # Only depth (1/z) is treated as scheduling parameter.
    # Camera intrinsics are fixed at nominal, and image coordinates are fixed
    # per feature point at their desired centered-pixel values. This preserves
    # Wang's per-point 2x6 Jacobian structure while isolating depth uncertainty.
    vertices_1p_per_point = None
    rank_1p = None
    ranks_1p = []
    sigmas_1p_all = []
    rel_err_1p_all = []

    print(f"\n{'='*70}")
    print(f"BONUS: 1-parameter UD model (1/Z only, per feature point)")
    print(f"{'='*70}")

    # For 1D, UD is just uniform sampling
    M_1p = M
    inv_z_samples = np.linspace(inv_z_min, inv_z_max, M_1p)
    print(f"  Uniform sampling: {M_1p} points in 1/Z=[{inv_z_min:.4f}, {inv_z_max:.4f}]")

    kx_nom = args.kx_nom
    ky_nom = args.ky_nom
    half_norm = (args.cube_size / 2.0) / args.desired_Z
    desired_centered_points = [
        (-half_norm * kx_nom, -half_norm * ky_nom),
        ( half_norm * kx_nom, -half_norm * ky_nom),
        ( half_norm * kx_nom,  half_norm * ky_nom),
        (-half_norm * kx_nom,  half_norm * ky_nom),
    ]
    print("  Fixed centered feature coordinates (px):")
    for j, (ui, vi) in enumerate(desired_centered_points):
        print(f"    point {j}: u={ui:.4f}, v={vi:.4f}")

    vertices_1p_per_point = []
    for j, (ui, vi) in enumerate(desired_centered_points):
        tensor_1p = np.zeros((M_1p, D))
        for i in range(M_1p):
            tensor_1p[i, :] = build_uncertain_jacobian(
                ui, vi, inv_z_samples[i], kx_nom, ky_nom).flatten()

        print(f"  HOSVD on 1-param tensor for point {j}...")
        core_1p, U1_1p, sigmas_1p, rank_j = hosvd_1d(tensor_1p, args.sv_threshold)

        core_no_1p, W1_1p = apply_no_1d(U1_1p, tensor_1p)
        T_recon_1p = W1_1p @ core_no_1p
        rel_err_1p = np.linalg.norm(tensor_1p - T_recon_1p) / np.linalg.norm(tensor_1p)
        print(f"    NO+SN reconstruction: rel={rel_err_1p:.4e}")
        print(f"    Result: {rank_j} vertices (point {j}, 1/Z only)")

        ranks_1p.append(int(rank_j))
        sigmas_1p_all.append([float(s) for s in sigmas_1p])
        rel_err_1p_all.append(float(rel_err_1p))
        point_vertices = []
        for r in range(rank_j):
            J_v = core_no_1p[r, :].reshape(2, 6)
            point_vertices.append(J_v)
            print(f"      J_s{r+1}: norm={np.linalg.norm(J_v):.4f}")
        vertices_1p_per_point.append(point_vertices)

    rank_1p = max(ranks_1p)
    if len(set(ranks_1p)) != 1:
        raise RuntimeError(f"1-param point ranks differ: {ranks_1p}")
    print(f"  Result: R={rank_1p} shared depth vertices per feature point")

    # =========================================================================
    # Save to YAML
    # =========================================================================
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = {
        "tp_ud_model": {
            "method": "UD_HOSVD_NO",
            "paper": "Wang2020_ActaPolytechnicaHungarica",
            "num_scheduling_params": 5,
            "scheduling_params": ["u_i", "v_i", "1/z_i", "k_x", "k_y"],
            "num_vertices": int(R),
            "ud_runs": int(M),
            "cd2_discrepancy": float(cd2),
            "camera_nominal": {
                "kx": float(args.kx_nom),
                "ky": float(args.ky_nom),
            },
            "uncertainty_bounds": {
                "kx_min": float(args.kx_min),
                "kx_max": float(args.kx_max),
                "ky_min": float(args.ky_min),
                "ky_max": float(args.ky_max),
            },
            "transformation_space": {
                "uc_min": float(args.uc_min),
                "uc_max": float(args.uc_max),
                "vc_min": float(args.vc_min),
                "vc_max": float(args.vc_max),
                "inv_z_min": float(inv_z_min),
                "inv_z_max": float(inv_z_max),
                "z_min": float(args.z_min),
                "z_max": float(args.z_max),
            },
            "singular_values": [float(s) for s in sigmas],
            "reconstruction_relative_error": float(rel_err_no),
            "vertices_5param": []
        }
    }

    for idx, v in enumerate(vertices):
        data["tp_ud_model"]["vertices_5param"].append({
            "index": int(idx),
            "matrix_2x6": [float(x) for x in v.flatten()]
        })

    if vertices_3p is not None:
        data["tp_ud_model"]["model_3param"] = {
            "num_vertices": int(rank_3p),
            "vertices_3param": []
        }
        for idx, v in enumerate(vertices_3p):
            data["tp_ud_model"]["model_3param"]["vertices_3param"].append({
                "index": int(idx),
                "matrix_2x6": [float(x) for x in v.flatten()]
            })

    if vertices_1p_per_point is not None:
        data["tp_ud_model"]["model_1param"] = {
            "num_vertices": int(rank_1p),
            "num_points": 4,
            "scheduling_params": ["1/z_i"],
            "description": "Only depth (1/z) as scheduling parameter. Feature coordinates fixed per point at desired centered-pixel values; kx=ky=nominal.",
            "fixed_feature_points_centered_px": [
                [float(ui), float(vi)] for ui, vi in desired_centered_points
            ],
            "singular_values_by_point": sigmas_1p_all,
            "reconstruction_relative_errors_by_point": rel_err_1p_all,
            "vertices_1param_per_point": []
        }
        for point_idx, point_vertices in enumerate(vertices_1p_per_point):
            point_entry = {
                "point_index": int(point_idx),
                "vertices": []
            }
            for idx, v in enumerate(point_vertices):
                point_entry["vertices"].append({
                    "index": int(idx),
                    "matrix_2x6": [float(x) for x in v.flatten()]
                })
            data["tp_ud_model"]["model_1param"]["vertices_1param_per_point"].append(point_entry)
        # Backward-compatible shared list for older readers. The updated solver
        # always prefers vertices_1param_per_point when present.
        data["tp_ud_model"]["model_1param"]["vertices_1param"] = []
        for idx, v in enumerate(vertices_1p_per_point[0]):
            data["tp_ud_model"]["model_1param"]["vertices_1param"].append({
                "index": int(idx),
                "matrix_2x6": [float(x) for x in v.flatten()]
            })

    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*70}")
    print(f"SAVED: {output_path}")
    print(f"  5-param model: {R} vertices (u, v, 1/Z, kx, ky)")
    if rank_3p is not None:
        print(f"  3-param model: {rank_3p} vertices (u, v, 1/Z)")
    if rank_1p is not None:
        print(f"  1-param model: {rank_1p} vertices (1/Z only)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
