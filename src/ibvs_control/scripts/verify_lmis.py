#!/usr/bin/env python3
"""
verify_lmis.py - LMI Verification Toolkit for TS/Polytopic Systems

This module provides comprehensive verification of LMI solutions for
Takagi-Sugeno (TS) or polytopic systems, supporting both discrete-time
and continuous-time formulations.

Author: LMI Verification Toolkit
Date: 2026

Usage:
    1. As a module: import verify_lmis and call functions
    2. Standalone: python3 verify_lmis.py (runs demo)
    3. With your data: modify the USER DATA SECTION in main()

Supports:
    - Discrete-time: e(k+1) = A_cl e(k), stability via |λ| < 1
    - Continuous-time: ė = A_cl e, stability via Re(λ) < 0
    - Sign conventions: v = K e  or  v = -F e
    - Multiple LMI solution formats (X/Y, X/M, direct P/K/F)
"""

import numpy as np
from numpy import linalg as LA
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
import warnings

# ==============================================================================
# CONFIGURATION DATACLASS
# ==============================================================================

@dataclass
class LMIVerificationConfig:
    """
    Configuration for LMI verification.
    
    Attributes
    ----------
    mode : str
        "discrete" or "continuous"
    sign_convention : str
        "v=Ke" (A_cl = A + B @ K) or "v=-Fe" (A_cl = A - B @ F)
    eps_pos : float
        Tolerance for positive definiteness (min eigenvalue >= eps_pos)
    eps_neg : float
        Tolerance for negative definiteness (max eigenvalue <= -eps_neg)
    eps_stability : float
        Margin for stability (spectral radius < 1 - eps for discrete)
    n_samples : int
        Number of random convex samples for polytope interior check
    n_trajectories : int
        Number of simulation trajectories for optional sanity check
    sim_horizon : int
        Simulation horizon (number of steps)
    random_seed : Optional[int]
        Random seed for reproducibility (None for random)
    verbose : bool
        Print detailed output
    """
    mode: str = "discrete"
    sign_convention: str = "v=Ke"
    eps_pos: float = 1e-8
    eps_neg: float = 1e-6
    eps_stability: float = 1e-6
    n_samples: int = 500
    n_trajectories: int = 50
    sim_horizon: int = 200
    random_seed: Optional[int] = 42
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.mode not in ["discrete", "continuous"]:
            raise ValueError(f"mode must be 'discrete' or 'continuous', got '{self.mode}'")
        
        # Normalize sign convention
        sign_norm = self.sign_convention.lower().replace(" ", "").replace("*", "")
        if sign_norm in ["v=ke", "v=k*e", "v=k e"]:
            self.sign_convention = "v=Ke"
        elif sign_norm in ["v=-fe", "v=-f*e", "v=-f e"]:
            self.sign_convention = "v=-Fe"
        else:
            raise ValueError(
                f"sign_convention must be 'v=Ke' or 'v=-Fe', got '{self.sign_convention}'"
            )


@dataclass
class VertexData:
    """
    Data for a single vertex of the polytopic system.
    
    Attributes
    ----------
    A : np.ndarray
        State matrix (n x n)
    B : np.ndarray
        Input matrix (n x m)
    index : int
        Vertex index (for reporting)
    """
    A: np.ndarray
    B: np.ndarray
    index: int = 0
    
    def __post_init__(self):
        """Validate dimensions."""
        self.A = np.asarray(self.A, dtype=np.float64)
        self.B = np.asarray(self.B, dtype=np.float64)
        
        if self.A.ndim != 2:
            raise ValueError(f"A must be 2D, got shape {self.A.shape}")
        if self.B.ndim != 2:
            raise ValueError(f"B must be 2D, got shape {self.B.shape}")
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError(f"A must be square, got shape {self.A.shape}")
        if self.A.shape[0] != self.B.shape[0]:
            raise ValueError(
                f"A and B must have same number of rows: A={self.A.shape}, B={self.B.shape}"
            )
    
    @property
    def n(self) -> int:
        """State dimension."""
        return self.A.shape[0]
    
    @property
    def m(self) -> int:
        """Input dimension."""
        return self.B.shape[1]


@dataclass
class LMISolution:
    """
    LMI solution data structure.
    
    Supports multiple input formats:
    - Case 1: X and Y_i provided -> K_i = Y_i @ inv(X)
    - Case 2: X and M_i provided -> F_i = M_i @ inv(X)
    - Case 3: P and K_i/F_i provided directly
    
    Attributes
    ----------
    X : Optional[np.ndarray]
        Lyapunov variable X (n x n), where P = inv(X) or X = P
    P : Optional[np.ndarray]
        Lyapunov matrix P (n x n), P > 0
    Y : Optional[List[np.ndarray]]
        List of Y_i matrices (m x n) for K reconstruction
    M : Optional[List[np.ndarray]]
        List of M_i matrices (m x n) for F reconstruction
    K : Optional[List[np.ndarray]]
        List of gain matrices K_i (m x n) - direct input
    F : Optional[List[np.ndarray]]
        List of gain matrices F_i (m x n) - direct input
    input_format : str
        Detected or specified format: "X_Y", "X_M", "P_K", "P_F"
    """
    X: Optional[np.ndarray] = None
    P: Optional[np.ndarray] = None
    Y: Optional[List[np.ndarray]] = None
    M: Optional[List[np.ndarray]] = None
    K: Optional[List[np.ndarray]] = None
    F: Optional[List[np.ndarray]] = None
    input_format: str = "auto"
    
    def __post_init__(self):
        """Convert arrays and detect format."""
        # Convert to numpy arrays
        if self.X is not None:
            self.X = np.asarray(self.X, dtype=np.float64)
        if self.P is not None:
            self.P = np.asarray(self.P, dtype=np.float64)
        if self.Y is not None:
            self.Y = [np.asarray(y, dtype=np.float64) for y in self.Y]
        if self.M is not None:
            self.M = [np.asarray(m, dtype=np.float64) for m in self.M]
        if self.K is not None:
            self.K = [np.asarray(k, dtype=np.float64) for k in self.K]
        if self.F is not None:
            self.F = [np.asarray(f, dtype=np.float64) for f in self.F]
        
        # Auto-detect format if needed
        if self.input_format == "auto":
            self.input_format = self._detect_format()
    
    def _detect_format(self) -> str:
        """Detect input format from provided data."""
        if self.X is not None and self.Y is not None:
            return "X_Y"
        elif self.X is not None and self.M is not None:
            return "X_M"
        elif self.P is not None and self.K is not None:
            return "P_K"
        elif self.P is not None and self.F is not None:
            return "P_F"
        elif self.X is not None and self.K is not None:
            # X provided with K directly (P = inv(X))
            return "X_K"
        elif self.X is not None and self.F is not None:
            return "X_F"
        else:
            raise ValueError(
                "Cannot detect input format. Provide one of:\n"
                "  - X and Y (for K_i = Y_i @ inv(X))\n"
                "  - X and M (for F_i = M_i @ inv(X))\n"
                "  - P and K (direct)\n"
                "  - P and F (direct)"
            )


@dataclass
class VerificationResult:
    """
    Results from LMI verification.
    
    Attributes
    ----------
    success : bool
        Overall verification passed
    P_valid : bool
        Lyapunov matrix P is positive definite
    P_min_eig : float
        Minimum eigenvalue of P
    P_cond : float
        Condition number of P
    gains : List[np.ndarray]
        Reconstructed gains (K_i or F_i)
    lyapunov_results : List[Dict]
        Per-vertex Lyapunov inequality results
    stability_results : List[Dict]
        Per-vertex eigenvalue stability results
    sampling_results : Dict
        Random convex sampling results
    simulation_results : Optional[Dict]
        Optional closed-loop simulation results
    warnings : List[str]
        Warning messages
    errors : List[str]
        Error messages
    """
    success: bool = False
    P_valid: bool = False
    P_min_eig: float = 0.0
    P_cond: float = float('inf')
    gains: List[np.ndarray] = field(default_factory=list)
    lyapunov_results: List[Dict] = field(default_factory=list)
    stability_results: List[Dict] = field(default_factory=list)
    sampling_results: Dict = field(default_factory=dict)
    simulation_results: Optional[Dict] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ==============================================================================
# A) BASIC MATRIX VALIDATION UTILITIES
# ==============================================================================

def symmetrize(M: np.ndarray) -> np.ndarray:
    """
    Symmetrize a matrix: S = 0.5 * (M + M.T)
    
    Parameters
    ----------
    M : np.ndarray
        Input matrix (n x n)
    
    Returns
    -------
    np.ndarray
        Symmetrized matrix
    """
    M = np.asarray(M, dtype=np.float64)
    return 0.5 * (M + M.T)


def eigenvalues_sym(M: np.ndarray) -> np.ndarray:
    """
    Compute eigenvalues of a symmetric matrix.
    
    Parameters
    ----------
    M : np.ndarray
        Input matrix (will be symmetrized)
    
    Returns
    -------
    np.ndarray
        Real eigenvalues (sorted ascending)
    """
    M_sym = symmetrize(M)
    eigvals = LA.eigvalsh(M_sym)  # eigvalsh for symmetric matrices (real eigenvalues)
    return np.sort(eigvals)


def min_eig_sym(M: np.ndarray) -> float:
    """
    Minimum eigenvalue of symmetrized matrix.
    
    Parameters
    ----------
    M : np.ndarray
        Input matrix (n x n)
    
    Returns
    -------
    float
        Minimum eigenvalue
    """
    return float(eigenvalues_sym(M)[0])


def max_eig_sym(M: np.ndarray) -> float:
    """
    Maximum eigenvalue of symmetrized matrix.
    
    Parameters
    ----------
    M : np.ndarray
        Input matrix (n x n)
    
    Returns
    -------
    float
        Maximum eigenvalue
    """
    return float(eigenvalues_sym(M)[-1])


def is_psd(M: np.ndarray, eps: float = 1e-8) -> Tuple[bool, float]:
    """
    Check if matrix is positive semi-definite (PSD).
    
    Parameters
    ----------
    M : np.ndarray
        Input matrix (n x n)
    eps : float
        Tolerance: min eigenvalue >= eps for strict PD
    
    Returns
    -------
    Tuple[bool, float]
        (is_psd, min_eigenvalue)
    """
    min_eig = min_eig_sym(M)
    return min_eig >= eps, min_eig


def is_pd(M: np.ndarray, eps: float = 1e-8) -> Tuple[bool, float]:
    """
    Check if matrix is positive definite (PD).
    Alias for is_psd with strict inequality interpretation.
    
    Parameters
    ----------
    M : np.ndarray
        Input matrix (n x n)
    eps : float
        Tolerance: min eigenvalue > eps
    
    Returns
    -------
    Tuple[bool, float]
        (is_pd, min_eigenvalue)
    """
    return is_psd(M, eps)


def is_nsd(M: np.ndarray, eps: float = 1e-6) -> Tuple[bool, float]:
    """
    Check if matrix is negative semi-definite (NSD).
    
    Parameters
    ----------
    M : np.ndarray
        Input matrix (n x n)
    eps : float
        Tolerance: max eigenvalue <= -eps for strict ND
    
    Returns
    -------
    Tuple[bool, float]
        (is_nsd, max_eigenvalue)
    """
    max_eig = max_eig_sym(M)
    return max_eig <= -eps, max_eig


def is_nd(M: np.ndarray, eps: float = 1e-6) -> Tuple[bool, float]:
    """
    Check if matrix is negative definite (ND).
    
    Parameters
    ----------
    M : np.ndarray
        Input matrix (n x n)
    eps : float
        Tolerance: max eigenvalue < -eps
    
    Returns
    -------
    Tuple[bool, float]
        (is_nd, max_eigenvalue)
    """
    return is_nsd(M, eps)


def condition_number(M: np.ndarray) -> float:
    """
    Compute condition number of a symmetric PSD matrix via eigenvalues.
    
    Parameters
    ----------
    M : np.ndarray
        Input matrix (assumed symmetric PSD)
    
    Returns
    -------
    float
        Condition number (max_eig / min_eig), or inf if singular
    """
    eigvals = eigenvalues_sym(M)
    min_eig = eigvals[0]
    max_eig = eigvals[-1]
    
    if min_eig <= 0:
        return float('inf')
    
    return max_eig / min_eig


def spectral_radius(M: np.ndarray) -> float:
    """
    Compute spectral radius (maximum absolute eigenvalue).
    
    Parameters
    ----------
    M : np.ndarray
        Input matrix (n x n)
    
    Returns
    -------
    float
        Spectral radius max|λ_i|
    """
    eigvals = LA.eigvals(M)
    return float(np.max(np.abs(eigvals)))


def max_real_part(M: np.ndarray) -> float:
    """
    Compute maximum real part of eigenvalues.
    
    Parameters
    ----------
    M : np.ndarray
        Input matrix (n x n)
    
    Returns
    -------
    float
        max Re(λ_i)
    """
    eigvals = LA.eigvals(M)
    return float(np.max(np.real(eigvals)))


def safe_inverse(M: np.ndarray, name: str = "matrix") -> np.ndarray:
    """
    Compute matrix inverse with error checking.
    
    Parameters
    ----------
    M : np.ndarray
        Input matrix (n x n)
    name : str
        Matrix name for error messages
    
    Returns
    -------
    np.ndarray
        Inverse of M
    
    Raises
    ------
    ValueError
        If matrix is singular or ill-conditioned
    """
    M = np.asarray(M, dtype=np.float64)
    
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be square, got shape {M.shape}")
    
    cond = LA.cond(M)
    if cond > 1e12:
        warnings.warn(f"{name} is ill-conditioned (cond={cond:.2e})")
    
    try:
        return LA.inv(M)
    except LA.LinAlgError:
        raise ValueError(f"{name} is singular, cannot invert")


# ==============================================================================
# B) VALIDATE X OR P POSITIVITY
# ==============================================================================

def validate_lyapunov_matrix(
    solution: LMISolution,
    config: LMIVerificationConfig
) -> Tuple[np.ndarray, bool, float, float, List[str]]:
    """
    Validate and extract the Lyapunov matrix P.
    
    Handles different input formats:
    - If P is provided directly, validate P > 0
    - If X is provided, compute P = inv(X) and validate
    
    Parameters
    ----------
    solution : LMISolution
        LMI solution data
    config : LMIVerificationConfig
        Verification configuration
    
    Returns
    -------
    Tuple[np.ndarray, bool, float, float, List[str]]
        (P, is_valid, min_eigenvalue, condition_number, warnings)
    """
    warns = []
    
    # Determine which variable to use
    if solution.P is not None:
        P = symmetrize(solution.P)
        source = "P (direct)"
    elif solution.X is not None:
        X = symmetrize(solution.X)
        
        # Check X > 0 first
        x_valid, x_min_eig = is_pd(X, config.eps_pos)
        if not x_valid:
            warns.append(f"X is not positive definite (min_eig={x_min_eig:.2e})")
        
        # Compute P = inv(X)
        try:
            P = safe_inverse(X, "X")
            P = symmetrize(P)
            source = "P = inv(X)"
        except ValueError as e:
            warns.append(str(e))
            # Return identity as fallback
            n = X.shape[0]
            P = np.eye(n)
            source = "P = I (fallback due to singular X)"
    else:
        raise ValueError("Neither P nor X provided in LMI solution")
    
    # Validate P > 0
    is_valid, min_eig = is_pd(P, config.eps_pos)
    cond = condition_number(P)
    
    if not is_valid:
        warns.append(f"P is not positive definite (min_eig={min_eig:.2e})")
    
    if min_eig < 1e-6 and min_eig >= config.eps_pos:
        warns.append(f"P has small minimum eigenvalue ({min_eig:.2e}), may be numerically marginal")
    
    if cond > 1e10:
        warns.append(f"P is ill-conditioned (cond={cond:.2e})")
    
    if config.verbose:
        print(f"\n{'='*60}")
        print("LYAPUNOV MATRIX VALIDATION")
        print(f"{'='*60}")
        print(f"Source: {source}")
        print(f"Dimension: {P.shape[0]} x {P.shape[1]}")
        print(f"Minimum eigenvalue: {min_eig:.6e}")
        print(f"Maximum eigenvalue: {max_eig_sym(P):.6e}")
        print(f"Condition number: {cond:.6e}")
        print(f"P > 0 check: {'PASS' if is_valid else 'FAIL'} (threshold: {config.eps_pos:.0e})")
        for w in warns:
            print(f"  WARNING: {w}")
    
    return P, is_valid, min_eig, cond, warns


# ==============================================================================
# C) GAIN RECONSTRUCTION
# ==============================================================================

def reconstruct_gains(
    solution: LMISolution,
    vertices: List[VertexData],
    config: LMIVerificationConfig
) -> Tuple[List[np.ndarray], str, List[str]]:
    """
    Reconstruct control gains from LMI solution.
    
    Parameters
    ----------
    solution : LMISolution
        LMI solution data
    vertices : List[VertexData]
        Vertex data (for dimension validation)
    config : LMIVerificationConfig
        Verification configuration
    
    Returns
    -------
    Tuple[List[np.ndarray], str, List[str]]
        (gains, gain_type, warnings)
        gain_type is "K" or "F"
    """
    warns = []
    n_vertices = len(vertices)
    n = vertices[0].n
    m = vertices[0].m
    
    fmt = solution.input_format
    
    if config.verbose:
        print(f"\n{'='*60}")
        print("GAIN RECONSTRUCTION")
        print(f"{'='*60}")
        print(f"Input format: {fmt}")
        print(f"Sign convention: {config.sign_convention}")
        print(f"Number of vertices: {n_vertices}")
        print(f"State dimension n={n}, Input dimension m={m}")
    
    gains = []
    gain_type = ""
    
    if fmt == "X_Y":
        # K_i = Y_i @ inv(X)
        X_inv = safe_inverse(symmetrize(solution.X), "X")
        if len(solution.Y) != n_vertices:
            raise ValueError(f"Number of Y matrices ({len(solution.Y)}) != vertices ({n_vertices})")
        
        for i, Y_i in enumerate(solution.Y):
            if Y_i.shape != (m, n):
                raise ValueError(f"Y[{i}] has shape {Y_i.shape}, expected ({m}, {n})")
            K_i = Y_i @ X_inv
            gains.append(K_i)
        gain_type = "K"
        
    elif fmt == "X_M":
        # F_i = M_i @ inv(X)
        X_inv = safe_inverse(symmetrize(solution.X), "X")
        if len(solution.M) != n_vertices:
            raise ValueError(f"Number of M matrices ({len(solution.M)}) != vertices ({n_vertices})")
        
        for i, M_i in enumerate(solution.M):
            if M_i.shape != (m, n):
                raise ValueError(f"M[{i}] has shape {M_i.shape}, expected ({m}, {n})")
            F_i = M_i @ X_inv
            gains.append(F_i)
        gain_type = "F"
        
    elif fmt == "P_K" or fmt == "X_K":
        # K_i provided directly
        if len(solution.K) != n_vertices:
            raise ValueError(f"Number of K matrices ({len(solution.K)}) != vertices ({n_vertices})")
        
        for i, K_i in enumerate(solution.K):
            if K_i.shape != (m, n):
                raise ValueError(f"K[{i}] has shape {K_i.shape}, expected ({m}, {n})")
            gains.append(K_i.copy())
        gain_type = "K"
        
    elif fmt == "P_F" or fmt == "X_F":
        # F_i provided directly
        if len(solution.F) != n_vertices:
            raise ValueError(f"Number of F matrices ({len(solution.F)}) != vertices ({n_vertices})")
        
        for i, F_i in enumerate(solution.F):
            if F_i.shape != (m, n):
                raise ValueError(f"F[{i}] has shape {F_i.shape}, expected ({m}, {n})")
            gains.append(F_i.copy())
        gain_type = "F"
        
    else:
        raise ValueError(f"Unknown input format: {fmt}")
    
    # Check sign convention consistency
    if gain_type == "K" and config.sign_convention == "v=-Fe":
        warns.append(
            "Reconstructed gains are K_i but sign convention is 'v=-Fe'. "
            "Will use A_cl = A - B @ K (treating K as F)."
        )
    elif gain_type == "F" and config.sign_convention == "v=Ke":
        warns.append(
            "Reconstructed gains are F_i but sign convention is 'v=Ke'. "
            "Will use A_cl = A + B @ F (treating F as K)."
        )
    
    if config.verbose:
        print(f"Gain type: {gain_type}")
        print(f"Gains reconstructed: {len(gains)}")
        for i, G in enumerate(gains):
            print(f"  {gain_type}[{i}] shape: {G.shape}, norm: {LA.norm(G):.4e}")
        for w in warns:
            print(f"  WARNING: {w}")
    
    return gains, gain_type, warns


def build_closed_loop_matrix(
    A: np.ndarray,
    B: np.ndarray,
    gain: np.ndarray,
    gain_type: str,
    sign_convention: str
) -> np.ndarray:
    """
    Build closed-loop matrix A_cl = A + B @ K or A_cl = A - B @ F.
    
    Parameters
    ----------
    A : np.ndarray
        State matrix (n x n)
    B : np.ndarray
        Input matrix (n x m)
    gain : np.ndarray
        Gain matrix K or F (m x n)
    gain_type : str
        "K" or "F"
    sign_convention : str
        "v=Ke" or "v=-Fe"
    
    Returns
    -------
    np.ndarray
        Closed-loop matrix A_cl
    """
    # Determine effective sign
    # If gain_type matches sign_convention, use standard formula
    # Otherwise, adapt
    
    if sign_convention == "v=Ke":
        # v = K @ e, so ė = A @ e + B @ v = A @ e + B @ K @ e = (A + B @ K) @ e
        # For discrete: e(k+1) = A @ e(k) + B @ K @ e(k) = (A + B @ K) @ e(k)
        A_cl = A + B @ gain
    else:  # "v=-Fe"
        # v = -F @ e, so ė = A @ e + B @ v = A @ e - B @ F @ e = (A - B @ F) @ e
        A_cl = A - B @ gain
    
    return A_cl


# ==============================================================================
# D) CORE LYAPUNOV INEQUALITY CHECKS
# ==============================================================================

def check_lyapunov_inequality(
    A_cl: np.ndarray,
    P: np.ndarray,
    mode: str,
    eps_neg: float = 1e-6
) -> Tuple[bool, float, np.ndarray]:
    """
    Check Lyapunov inequality for stability.
    
    For discrete-time:
        Delta = A_cl.T @ P @ A_cl - P ≺ 0
        
    For continuous-time:
        Delta = A_cl.T @ P + P @ A_cl ≺ 0
    
    Parameters
    ----------
    A_cl : np.ndarray
        Closed-loop matrix (n x n)
    P : np.ndarray
        Lyapunov matrix (n x n), P > 0
    mode : str
        "discrete" or "continuous"
    eps_neg : float
        Tolerance for negative definiteness
    
    Returns
    -------
    Tuple[bool, float, np.ndarray]
        (is_satisfied, max_eigenvalue, Delta)
    """
    if mode == "discrete":
        # Discrete Lyapunov: A_cl.T @ P @ A_cl - P < 0
        Delta = A_cl.T @ P @ A_cl - P
    else:  # continuous
        # Continuous Lyapunov: A_cl.T @ P + P @ A_cl < 0
        Delta = A_cl.T @ P + P @ A_cl
    
    Delta = symmetrize(Delta)
    is_satisfied, max_eig = is_nd(Delta, eps_neg)
    
    return is_satisfied, max_eig, Delta


def verify_lyapunov_vertices(
    vertices: List[VertexData],
    gains: List[np.ndarray],
    P: np.ndarray,
    config: LMIVerificationConfig
) -> List[Dict]:
    """
    Verify Lyapunov inequality at all vertices.
    
    Parameters
    ----------
    vertices : List[VertexData]
        Vertex data
    gains : List[np.ndarray]
        Control gains (K_i or F_i)
    P : np.ndarray
        Lyapunov matrix
    config : LMIVerificationConfig
        Configuration
    
    Returns
    -------
    List[Dict]
        Per-vertex results with keys:
        - vertex_index
        - passed
        - max_eigenvalue
        - margin (how negative)
    """
    results = []
    
    if config.verbose:
        print(f"\n{'='*60}")
        print("LYAPUNOV INEQUALITY VERIFICATION (VERTICES)")
        print(f"{'='*60}")
        print(f"Mode: {config.mode}")
        if config.mode == "discrete":
            print("Checking: A_cl.T @ P @ A_cl - P ≺ 0")
        else:
            print("Checking: A_cl.T @ P + P @ A_cl ≺ 0")
    
    worst_max_eig = -float('inf')
    worst_idx = 0
    
    for i, (vertex, gain) in enumerate(zip(vertices, gains)):
        A_cl = build_closed_loop_matrix(
            vertex.A, vertex.B, gain, "K", config.sign_convention
        )
        
        passed, max_eig, Delta = check_lyapunov_inequality(
            A_cl, P, config.mode, config.eps_neg
        )
        
        margin = -max_eig  # How "negative" the max eigenvalue is
        
        results.append({
            'vertex_index': i,
            'passed': passed,
            'max_eigenvalue': max_eig,
            'margin': margin
        })
        
        if max_eig > worst_max_eig:
            worst_max_eig = max_eig
            worst_idx = i
        
        if config.verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  Vertex {i}: max_eig(Δ)={max_eig:+.6e} [{status}] margin={margin:.6e}")
    
    if config.verbose:
        print(f"\nWorst case: Vertex {worst_idx} with max_eig={worst_max_eig:+.6e}")
        all_passed = all(r['passed'] for r in results)
        print(f"Overall: {'ALL PASS' if all_passed else 'SOME FAILED'}")
    
    return results


# ==============================================================================
# E) EIGENVALUE STABILITY CHECKS
# ==============================================================================

def check_eigenvalue_stability(
    A_cl: np.ndarray,
    mode: str,
    eps_stability: float = 1e-6
) -> Tuple[bool, float, np.ndarray]:
    """
    Check eigenvalue stability of closed-loop matrix.
    
    For discrete-time:
        All |λ_i| < 1 (Schur stable)
        
    For continuous-time:
        All Re(λ_i) < 0 (Hurwitz stable)
    
    Parameters
    ----------
    A_cl : np.ndarray
        Closed-loop matrix (n x n)
    mode : str
        "discrete" or "continuous"
    eps_stability : float
        Stability margin
    
    Returns
    -------
    Tuple[bool, float, np.ndarray]
        (is_stable, worst_metric, eigenvalues)
        For discrete: worst_metric = max |λ_i| (should be < 1)
        For continuous: worst_metric = max Re(λ_i) (should be < 0)
    """
    eigvals = LA.eigvals(A_cl)
    
    if mode == "discrete":
        # Spectral radius should be < 1
        abs_eigvals = np.abs(eigvals)
        worst_metric = float(np.max(abs_eigvals))
        is_stable = worst_metric < (1.0 - eps_stability)
    else:  # continuous
        # Max real part should be < 0
        real_parts = np.real(eigvals)
        worst_metric = float(np.max(real_parts))
        is_stable = worst_metric < -eps_stability
    
    return is_stable, worst_metric, eigvals


def verify_eigenvalue_stability_vertices(
    vertices: List[VertexData],
    gains: List[np.ndarray],
    config: LMIVerificationConfig
) -> List[Dict]:
    """
    Verify eigenvalue stability at all vertices.
    
    Parameters
    ----------
    vertices : List[VertexData]
        Vertex data
    gains : List[np.ndarray]
        Control gains
    config : LMIVerificationConfig
        Configuration
    
    Returns
    -------
    List[Dict]
        Per-vertex results
    """
    results = []
    
    if config.verbose:
        print(f"\n{'='*60}")
        print("EIGENVALUE STABILITY VERIFICATION (VERTICES)")
        print(f"{'='*60}")
        if config.mode == "discrete":
            print("Checking: spectral radius ρ(A_cl) < 1")
        else:
            print("Checking: max Re(λ) < 0 (Hurwitz)")
    
    worst_metric = -float('inf') if config.mode == "continuous" else 0.0
    worst_idx = 0
    
    for i, (vertex, gain) in enumerate(zip(vertices, gains)):
        A_cl = build_closed_loop_matrix(
            vertex.A, vertex.B, gain, "K", config.sign_convention
        )
        
        is_stable, metric, eigvals = check_eigenvalue_stability(
            A_cl, config.mode, config.eps_stability
        )
        
        results.append({
            'vertex_index': i,
            'stable': is_stable,
            'worst_metric': metric,
            'eigenvalues': eigvals
        })
        
        if config.mode == "discrete":
            if metric > worst_metric:
                worst_metric = metric
                worst_idx = i
        else:
            if metric > worst_metric:
                worst_metric = metric
                worst_idx = i
        
        if config.verbose:
            status = "STABLE" if is_stable else "UNSTABLE"
            if config.mode == "discrete":
                print(f"  Vertex {i}: ρ(A_cl)={metric:.6f} [{status}]")
            else:
                print(f"  Vertex {i}: max_Re(λ)={metric:+.6e} [{status}]")
    
    if config.verbose:
        print(f"\nWorst case: Vertex {worst_idx}")
        if config.mode == "discrete":
            print(f"  Spectral radius: {worst_metric:.6f}")
            if worst_metric < 1:
                print(f"  Margin to instability: {1.0 - worst_metric:.6f}")
        else:
            print(f"  Max real part: {worst_metric:+.6e}")
            if worst_metric < 0:
                print(f"  Margin to instability: {-worst_metric:.6e}")
        
        all_stable = all(r['stable'] for r in results)
        print(f"Overall: {'ALL STABLE' if all_stable else 'SOME UNSTABLE'}")
    
    return results


# ==============================================================================
# F) RANDOM CONVEX SAMPLING (POLYTOPE INTERIOR)
# ==============================================================================

def generate_convex_weights(n_vertices: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate random convex weights using Dirichlet distribution.
    
    Parameters
    ----------
    n_vertices : int
        Number of vertices
    n_samples : int
        Number of samples
    rng : np.random.Generator
        Random number generator
    
    Returns
    -------
    np.ndarray
        Weights array (n_samples x n_vertices), each row sums to 1
    """
    # Dirichlet with alpha=1 gives uniform distribution over simplex
    weights = rng.dirichlet(np.ones(n_vertices), size=n_samples)
    return weights


def verify_convex_sampling(
    vertices: List[VertexData],
    gains: List[np.ndarray],
    P: np.ndarray,
    config: LMIVerificationConfig
) -> Dict:
    """
    Verify stability via random convex sampling of polytope interior.
    
    For each random convex combination h:
        A_cl(h) = Σ_i h_i * A_cl_i
    Check Lyapunov inequality and eigenvalue stability.
    
    Parameters
    ----------
    vertices : List[VertexData]
        Vertex data
    gains : List[np.ndarray]
        Control gains
    P : np.ndarray
        Lyapunov matrix
    config : LMIVerificationConfig
        Configuration
    
    Returns
    -------
    Dict
        Sampling results with keys:
        - n_samples
        - n_lyap_failures
        - n_eig_failures
        - worst_lyap_eig
        - worst_lyap_weights
        - worst_eig_metric
        - worst_eig_weights
    """
    n_vertices = len(vertices)
    n_samples = config.n_samples
    
    # Initialize RNG
    rng = np.random.default_rng(config.random_seed)
    
    # Build closed-loop matrices for all vertices
    A_cl_vertices = []
    for vertex, gain in zip(vertices, gains):
        A_cl = build_closed_loop_matrix(
            vertex.A, vertex.B, gain, "K", config.sign_convention
        )
        A_cl_vertices.append(A_cl)
    
    # Generate random convex weights
    weights = generate_convex_weights(n_vertices, n_samples, rng)
    
    if config.verbose:
        print(f"\n{'='*60}")
        print("RANDOM CONVEX SAMPLING VERIFICATION")
        print(f"{'='*60}")
        print(f"Number of samples: {n_samples}")
        print(f"Random seed: {config.random_seed}")
    
    # Track results
    lyap_failures = 0
    eig_failures = 0
    worst_lyap_eig = -float('inf')
    worst_lyap_weights = None
    worst_eig_metric = -float('inf') if config.mode == "continuous" else 0.0
    worst_eig_weights = None
    
    for s in range(n_samples):
        h = weights[s]
        
        # Build convex combination of closed-loop matrices
        A_cl_h = np.zeros_like(A_cl_vertices[0])
        for i in range(n_vertices):
            A_cl_h += h[i] * A_cl_vertices[i]
        
        # Check Lyapunov inequality
        lyap_passed, lyap_max_eig, _ = check_lyapunov_inequality(
            A_cl_h, P, config.mode, config.eps_neg
        )
        if not lyap_passed:
            lyap_failures += 1
        if lyap_max_eig > worst_lyap_eig:
            worst_lyap_eig = lyap_max_eig
            worst_lyap_weights = h.copy()
        
        # Check eigenvalue stability
        eig_stable, eig_metric, _ = check_eigenvalue_stability(
            A_cl_h, config.mode, config.eps_stability
        )
        if not eig_stable:
            eig_failures += 1
        
        if config.mode == "discrete":
            if eig_metric > worst_eig_metric:
                worst_eig_metric = eig_metric
                worst_eig_weights = h.copy()
        else:
            if eig_metric > worst_eig_metric:
                worst_eig_metric = eig_metric
                worst_eig_weights = h.copy()
    
    results = {
        'n_samples': n_samples,
        'n_lyap_failures': lyap_failures,
        'n_eig_failures': eig_failures,
        'worst_lyap_eig': worst_lyap_eig,
        'worst_lyap_weights': worst_lyap_weights,
        'worst_eig_metric': worst_eig_metric,
        'worst_eig_weights': worst_eig_weights,
        'lyap_passed': lyap_failures == 0,
        'eig_passed': eig_failures == 0
    }
    
    if config.verbose:
        print(f"\nLyapunov inequality check:")
        print(f"  Failures: {lyap_failures}/{n_samples} ({100*lyap_failures/n_samples:.1f}%)")
        print(f"  Worst max_eig(Δ): {worst_lyap_eig:+.6e}")
        if worst_lyap_weights is not None:
            print(f"  Worst weights: {np.array2string(worst_lyap_weights, precision=3)}")
        
        print(f"\nEigenvalue stability check:")
        print(f"  Failures: {eig_failures}/{n_samples} ({100*eig_failures/n_samples:.1f}%)")
        if config.mode == "discrete":
            print(f"  Worst ρ(A_cl): {worst_eig_metric:.6f}")
        else:
            print(f"  Worst max_Re(λ): {worst_eig_metric:+.6e}")
        if worst_eig_weights is not None:
            print(f"  Worst weights: {np.array2string(worst_eig_weights, precision=3)}")
        
        overall = results['lyap_passed'] and results['eig_passed']
        print(f"\nOverall sampling check: {'PASS' if overall else 'FAIL'}")
    
    return results


# ==============================================================================
# G) OPTIONAL CLOSED-LOOP SIMULATION TEST
# ==============================================================================

def simulate_closed_loop(
    vertices: List[VertexData],
    gains: List[np.ndarray],
    config: LMIVerificationConfig,
    fixed_weights: Optional[np.ndarray] = None
) -> Dict:
    """
    Run closed-loop simulation for sanity check (discrete-time only).
    
    Parameters
    ----------
    vertices : List[VertexData]
        Vertex data
    gains : List[np.ndarray]
        Control gains
    config : LMIVerificationConfig
        Configuration
    fixed_weights : Optional[np.ndarray]
        If provided, use these fixed weights. Otherwise, random per step.
    
    Returns
    -------
    Dict
        Simulation results
    """
    if config.mode != "discrete":
        return {
            'skipped': True,
            'reason': 'Simulation only implemented for discrete-time'
        }
    
    n_vertices = len(vertices)
    n = vertices[0].n
    n_traj = config.n_trajectories
    horizon = config.sim_horizon
    
    rng = np.random.default_rng(config.random_seed)
    
    # Build closed-loop matrices
    A_cl_vertices = []
    for vertex, gain in zip(vertices, gains):
        A_cl = build_closed_loop_matrix(
            vertex.A, vertex.B, gain, "K", config.sign_convention
        )
        A_cl_vertices.append(A_cl)
    
    if config.verbose:
        print(f"\n{'='*60}")
        print("CLOSED-LOOP SIMULATION (SANITY CHECK)")
        print(f"{'='*60}")
        print(f"Trajectories: {n_traj}")
        print(f"Horizon: {horizon} steps")
        print(f"Weight mode: {'fixed' if fixed_weights is not None else 'random per step'}")
    
    # Track statistics
    final_norms = []
    diverged_count = 0
    norm_histories = []
    
    for traj in range(n_traj):
        # Random initial condition (unit norm)
        e0 = rng.standard_normal(n)
        e0 = e0 / LA.norm(e0)
        
        e = e0.copy()
        norms = [LA.norm(e)]
        
        for k in range(horizon):
            # Select weights
            if fixed_weights is not None:
                h = fixed_weights
            else:
                h = rng.dirichlet(np.ones(n_vertices))
            
            # Convex combination
            A_cl = np.zeros((n, n))
            for i in range(n_vertices):
                A_cl += h[i] * A_cl_vertices[i]
            
            # Propagate
            e = A_cl @ e
            norms.append(LA.norm(e))
            
            # Check for divergence
            if LA.norm(e) > 1e6:
                diverged_count += 1
                break
        
        final_norms.append(norms[-1])
        norm_histories.append(norms)
    
    # Statistics
    final_norms = np.array(final_norms)
    avg_final = np.mean(final_norms)
    max_final = np.max(final_norms)
    converged_count = np.sum(final_norms < 1e-3)
    
    results = {
        'skipped': False,
        'n_trajectories': n_traj,
        'horizon': horizon,
        'diverged_count': diverged_count,
        'converged_count': int(converged_count),
        'avg_final_norm': float(avg_final),
        'max_final_norm': float(max_final),
        'all_converged': diverged_count == 0 and converged_count == n_traj
    }
    
    if config.verbose:
        print(f"\nResults:")
        print(f"  Diverged: {diverged_count}/{n_traj}")
        print(f"  Converged (||e|| < 1e-3): {converged_count}/{n_traj}")
        print(f"  Avg final ||e||: {avg_final:.6e}")
        print(f"  Max final ||e||: {max_final:.6e}")
        print(f"  All converged: {'YES' if results['all_converged'] else 'NO'}")
    
    return results


# ==============================================================================
# H) MAIN VERIFICATION FUNCTION
# ==============================================================================

def verify_lmi_solution(
    vertices: List[VertexData],
    solution: LMISolution,
    config: LMIVerificationConfig,
    run_simulation: bool = True
) -> VerificationResult:
    """
    Run complete LMI verification.
    
    Parameters
    ----------
    vertices : List[VertexData]
        List of vertex data (A_i, B_i)
    solution : LMISolution
        LMI solution (X, Y, P, K, F, etc.)
    config : LMIVerificationConfig
        Verification configuration
    run_simulation : bool
        Whether to run optional simulation
    
    Returns
    -------
    VerificationResult
        Complete verification results
    """
    result = VerificationResult()
    
    if config.verbose:
        print("\n" + "="*60)
        print("LMI VERIFICATION TOOLKIT")
        print("="*60)
        print(f"Mode: {config.mode}")
        print(f"Sign convention: {config.sign_convention}")
        print(f"Number of vertices: {len(vertices)}")
        print(f"State dim: {vertices[0].n}, Input dim: {vertices[0].m}")
    
    try:
        # B) Validate Lyapunov matrix
        P, p_valid, p_min_eig, p_cond, p_warns = validate_lyapunov_matrix(
            solution, config
        )
        result.P_valid = p_valid
        result.P_min_eig = p_min_eig
        result.P_cond = p_cond
        result.warnings.extend(p_warns)
        
        if not p_valid:
            result.errors.append("Lyapunov matrix P is not positive definite")
        
        # C) Reconstruct gains
        gains, gain_type, g_warns = reconstruct_gains(solution, vertices, config)
        result.gains = gains
        result.warnings.extend(g_warns)
        
        # D) Lyapunov inequality checks at vertices
        lyap_results = verify_lyapunov_vertices(vertices, gains, P, config)
        result.lyapunov_results = lyap_results
        
        lyap_all_pass = all(r['passed'] for r in lyap_results)
        if not lyap_all_pass:
            failed_vertices = [r['vertex_index'] for r in lyap_results if not r['passed']]
            result.errors.append(f"Lyapunov inequality failed at vertices: {failed_vertices}")
        
        # E) Eigenvalue stability checks at vertices
        stab_results = verify_eigenvalue_stability_vertices(vertices, gains, config)
        result.stability_results = stab_results
        
        stab_all_pass = all(r['stable'] for r in stab_results)
        if not stab_all_pass:
            unstable_vertices = [r['vertex_index'] for r in stab_results if not r['stable']]
            result.errors.append(f"Eigenvalue stability failed at vertices: {unstable_vertices}")
        
        # F) Random convex sampling
        samp_results = verify_convex_sampling(vertices, gains, P, config)
        result.sampling_results = samp_results
        
        if not samp_results['lyap_passed']:
            result.warnings.append(
                f"Lyapunov check failed in {samp_results['n_lyap_failures']} random samples"
            )
        if not samp_results['eig_passed']:
            result.warnings.append(
                f"Eigenvalue check failed in {samp_results['n_eig_failures']} random samples"
            )
        
        # G) Optional simulation
        if run_simulation and config.mode == "discrete":
            sim_results = simulate_closed_loop(vertices, gains, config)
            result.simulation_results = sim_results
            
            if not sim_results.get('skipped', True):
                if sim_results['diverged_count'] > 0:
                    result.warnings.append(
                        f"Simulation: {sim_results['diverged_count']} trajectories diverged"
                    )
        
        # Overall success
        result.success = (
            p_valid and 
            lyap_all_pass and 
            stab_all_pass and
            samp_results['lyap_passed'] and
            samp_results['eig_passed']
        )
        
    except Exception as e:
        result.errors.append(f"Verification error: {str(e)}")
        result.success = False
        raise
    
    # Print summary
    if config.verbose:
        print_summary(result, config)
    
    return result


def print_summary(result: VerificationResult, config: LMIVerificationConfig):
    """Print verification summary."""
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    print(f"\n[Lyapunov Matrix P]")
    print(f"  Valid (P > 0): {'YES' if result.P_valid else 'NO'}")
    print(f"  Min eigenvalue: {result.P_min_eig:.6e}")
    print(f"  Condition number: {result.P_cond:.6e}")
    
    print(f"\n[Vertex Lyapunov Inequalities]")
    n_pass = sum(1 for r in result.lyapunov_results if r['passed'])
    n_total = len(result.lyapunov_results)
    print(f"  Passed: {n_pass}/{n_total}")
    if result.lyapunov_results:
        worst = max(result.lyapunov_results, key=lambda r: r['max_eigenvalue'])
        print(f"  Worst max_eig(Δ): {worst['max_eigenvalue']:+.6e} (vertex {worst['vertex_index']})")
    
    print(f"\n[Vertex Eigenvalue Stability]")
    n_stable = sum(1 for r in result.stability_results if r['stable'])
    print(f"  Stable: {n_stable}/{n_total}")
    if result.stability_results:
        if config.mode == "discrete":
            worst = max(result.stability_results, key=lambda r: r['worst_metric'])
            print(f"  Worst ρ(A_cl): {worst['worst_metric']:.6f} (vertex {worst['vertex_index']})")
        else:
            worst = max(result.stability_results, key=lambda r: r['worst_metric'])
            print(f"  Worst max_Re(λ): {worst['worst_metric']:+.6e} (vertex {worst['vertex_index']})")
    
    print(f"\n[Random Convex Sampling]")
    sr = result.sampling_results
    if sr:
        print(f"  Samples: {sr['n_samples']}")
        print(f"  Lyapunov failures: {sr['n_lyap_failures']}")
        print(f"  Eigenvalue failures: {sr['n_eig_failures']}")
        print(f"  Worst Lyap max_eig: {sr['worst_lyap_eig']:+.6e}")
        if config.mode == "discrete":
            print(f"  Worst ρ(A_cl): {sr['worst_eig_metric']:.6f}")
        else:
            print(f"  Worst max_Re(λ): {sr['worst_eig_metric']:+.6e}")
    
    if result.simulation_results and not result.simulation_results.get('skipped', True):
        print(f"\n[Simulation Check]")
        sim = result.simulation_results
        print(f"  Diverged: {sim['diverged_count']}/{sim['n_trajectories']}")
        print(f"  Converged: {sim['converged_count']}/{sim['n_trajectories']}")
        print(f"  Avg final ||e||: {sim['avg_final_norm']:.6e}")
    
    if result.warnings:
        print(f"\n[Warnings]")
        for w in result.warnings:
            print(f"  ⚠ {w}")
    
    if result.errors:
        print(f"\n[Errors]")
        for e in result.errors:
            print(f"  ✗ {e}")
    
    print("\n" + "="*60)
    if result.success:
        print("✓ OVERALL: VERIFICATION PASSED")
    else:
        print("✗ OVERALL: VERIFICATION FAILED")
    print("="*60 + "\n")


# ==============================================================================
# I) DEMO / MAIN
# ==============================================================================

def create_demo_system():
    """
    Create a small synthetic system for demonstration.
    
    Returns a stabilizable 2-state, 1-input, 2-vertex system with
    a valid LMI solution.
    """
    print("\n" + "="*60)
    print("CREATING DEMO SYSTEM")
    print("="*60)
    
    # Small discrete-time system: e(k+1) = A_i @ e(k) + B_i @ u(k)
    n = 2  # state dimension
    m = 1  # input dimension
    
    # Vertex 1: Nominal
    A1 = np.array([
        [1.0, 0.1],
        [0.0, 1.0]
    ])
    B1 = np.array([
        [0.05],
        [0.1]
    ])
    
    # Vertex 2: Perturbed
    A2 = np.array([
        [1.0, 0.15],
        [0.0, 1.0]
    ])
    B2 = np.array([
        [0.04],
        [0.12]
    ])
    
    vertices = [
        VertexData(A=A1, B=B1, index=0),
        VertexData(A=A2, B=B2, index=1)
    ]
    
    print(f"State dimension: {n}")
    print(f"Input dimension: {m}")
    print(f"Number of vertices: {len(vertices)}")
    
    # Create a stabilizing controller via pole placement heuristic
    # For this simple system, we can find K by trial or simple LQR-like approach
    
    # Use common P and K for simplicity (not from actual LMI solver, but valid)
    # P = I works if K stabilizes all vertices
    
    # Design K to place closed-loop eigenvalues inside unit circle
    # A_cl = A + B @ K
    # For vertex 1: eigenvalues of A1 + B1 @ K should be inside unit circle
    
    # Simple stabilizing gain (found by inspection/iteration)
    K = np.array([[-2.0, -1.5]])  # (1 x 2)
    
    # Verify this stabilizes both vertices
    A_cl1 = A1 + B1 @ K
    A_cl2 = A2 + B2 @ K
    
    rho1 = spectral_radius(A_cl1)
    rho2 = spectral_radius(A_cl2)
    
    print(f"\nDemo gains: K = {K}")
    print(f"Vertex 1 spectral radius: {rho1:.4f}")
    print(f"Vertex 2 spectral radius: {rho2:.4f}")
    
    if rho1 >= 1 or rho2 >= 1:
        print("WARNING: Demo gains don't stabilize! Adjusting...")
        # More aggressive gain
        K = np.array([[-3.0, -2.0]])
        A_cl1 = A1 + B1 @ K
        A_cl2 = A2 + B2 @ K
        rho1 = spectral_radius(A_cl1)
        rho2 = spectral_radius(A_cl2)
        print(f"Adjusted K = {K}")
        print(f"Vertex 1 spectral radius: {rho1:.4f}")
        print(f"Vertex 2 spectral radius: {rho2:.4f}")
    
    # Find common Lyapunov P by solving discrete Lyapunov equation
    # P - A_cl.T @ P @ A_cl = Q, with Q = I
    # This is a simplification; real LMI would solve for P simultaneously
    
    # Use average closed-loop for P computation
    A_cl_avg = 0.5 * (A_cl1 + A_cl2)
    
    # Solve discrete Lyapunov: P = A_cl_avg.T @ P @ A_cl_avg + Q
    # P - A.T @ P @ A = Q  =>  P = dlyap(A.T, Q) in control theory notation
    # Manual iterative solution:
    Q = np.eye(n)
    P = Q.copy()
    for _ in range(100):
        P_new = A_cl_avg.T @ P @ A_cl_avg + Q
        if LA.norm(P_new - P) < 1e-10:
            break
        P = P_new
    
    # Ensure P is symmetric and positive definite
    P = symmetrize(P)
    
    print(f"\nComputed Lyapunov P (condition number: {condition_number(P):.2f})")
    
    # Create LMI solution structure
    solution = LMISolution(
        P=P,
        K=[K, K],  # Same K for both vertices in this simple example
        input_format="P_K"
    )
    
    return vertices, solution


def run_demo():
    """Run demonstration with synthetic system."""
    print("\n" + "#"*60)
    print("# LMI VERIFICATION TOOLKIT - DEMONSTRATION")
    print("#"*60)
    
    # Create demo system
    vertices, solution = create_demo_system()
    
    # Configure verification
    config = LMIVerificationConfig(
        mode="discrete",
        sign_convention="v=Ke",
        eps_pos=1e-8,
        eps_neg=1e-6,
        eps_stability=1e-6,
        n_samples=200,
        n_trajectories=20,
        sim_horizon=100,
        random_seed=42,
        verbose=True
    )
    
    # Run verification
    result = verify_lmi_solution(vertices, solution, config, run_simulation=True)
    
    return result


# ==============================================================================
# USER DATA SECTION - PASTE YOUR MATRICES HERE
# ==============================================================================

def load_user_data():
    """
    Load user's actual IBVS data.
    
    INSTRUCTIONS:
    1. Replace the placeholder matrices below with your actual data
    2. Adjust dimensions as needed (n=8 for 4-point IBVS, m=6 for camera velocities)
    3. Choose the appropriate input format and sign convention
    
    Your discrete-time IBVS model:
        e(k+1) = e(k) + T_s * L(k) * v(k)
    So for each vertex i:
        A_i = I (8x8 identity)
        B_i = T_s * L_i (8x6)
    
    Common sign convention for IBVS: v = -λ * L^+ * e, which is "v = -F e" form
    where F = λ * L^+
    """
    
    # =========================================================================
    # EXAMPLE: 8-state, 6-input IBVS system with 2 vertices (Z_min, Z_max)
    # =========================================================================
    
    n = 8   # State dimension (4 points × 2 coordinates)
    m = 6   # Input dimension (6-DOF camera velocity)
    T_s = 0.05  # Sampling period
    
    # Identity for A (discrete IBVS with forward Euler)
    A_common = np.eye(n)
    
    # Example interaction matrices at Z_min and Z_max
    # Replace these with your actual L_i matrices from the node!
    
    # Placeholder: Random interaction matrices (REPLACE WITH YOUR DATA)
    np.random.seed(123)
    L1 = np.random.randn(n, m) * 0.1  # L at Z_min
    L2 = np.random.randn(n, m) * 0.08  # L at Z_max
    
    B1 = T_s * L1
    B2 = T_s * L2
    
    vertices = [
        VertexData(A=A_common, B=B1, index=0),
        VertexData(A=A_common, B=B2, index=1),
    ]
    
    # =========================================================================
    # LMI SOLUTION - Replace with your solver output
    # =========================================================================
    
    # Option 1: If you have X (Lyapunov variable) and Y_i (for K reconstruction)
    # X = np.array([...])  # (n x n)
    # Y = [np.array([...]), np.array([...])]  # List of (m x n) for each vertex
    # solution = LMISolution(X=X, Y=Y)
    
    # Option 2: If you have P directly and K_i directly
    # P = np.array([...])  # (n x n) positive definite
    # K = [np.array([...]), np.array([...])]  # List of (m x n)
    # solution = LMISolution(P=P, K=K)
    
    # Option 3: For v = -F e convention
    # P = np.array([...])
    # F = [np.array([...]), np.array([...])]
    # solution = LMISolution(P=P, F=F)
    
    # Placeholder: Create dummy stabilizing solution
    P = np.eye(n) * 10  # Placeholder P
    K = [np.zeros((m, n)), np.zeros((m, n))]  # Placeholder gains (won't stabilize!)
    
    solution = LMISolution(P=P, K=K, input_format="P_K")
    
    return vertices, solution


def run_user_verification():
    """
    Run verification with user's data.
    Uncomment and modify this function to use your actual data.
    """
    print("\n" + "#"*60)
    print("# USER DATA VERIFICATION")
    print("#"*60)
    
    # Load user data
    vertices, solution = load_user_data()
    
    # Configure for IBVS
    config = LMIVerificationConfig(
        mode="discrete",           # IBVS uses discrete-time
        sign_convention="v=Ke",    # Or "v=-Fe" depending on your controller
        eps_pos=1e-8,
        eps_neg=1e-6,
        eps_stability=1e-6,
        n_samples=500,
        n_trajectories=50,
        sim_horizon=200,
        random_seed=42,
        verbose=True
    )
    
    # Run verification
    result = verify_lmi_solution(vertices, solution, config, run_simulation=True)
    
    return result


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("LMI VERIFICATION TOOLKIT")
    print("For TS/Polytopic Systems (Discrete & Continuous)")
    print("="*60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--user":
        # Run with user data
        print("\nRunning with user data...")
        result = run_user_verification()
    else:
        # Run demo
        print("\nRunning demonstration...")
        print("(Use --user flag to run with your own data)")
        result = run_demo()
    
    # Exit with appropriate code
    sys.exit(0 if result.success else 1)
