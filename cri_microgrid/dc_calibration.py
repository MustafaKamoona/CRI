"""Data-Calibrated CRI (DC-CRI): risk-aware, data-driven calibration of the
CRI indicator weights omega and supervisory thresholds eta_A/eta_M/eta_D,
implementing exactly the procedure of CRI_Main_Paper_V2.tex Sec. "Risk-Aware
Calibration of CRI Parameters" (Eqs. weight_calibration, risk_constrained_threshold,
relative_thresholds) and the finite-horizon activation-risk label (Eq.
finite_horizon_hazard).

Nothing here changes the online CRI architecture: CRISupervisor, cri.py,
sensitivity.py, and probing.py are used completely unmodified. DC-CRI is
*only* a different way of choosing the CRIParams passed into the exact same
CRISupervisor -- calibration is an offline, one-time procedure; the online
per-sample computation is identical between EC-CRI and DC-CRI.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import numpy as np
from scipy.optimize import minimize

from .config import CRIParams, ConstraintParams

INDICATOR_ORDER = ["RI", "CI", "SMI", "AI", "NCI"]


# =============================================================================
# Finite-horizon activation-risk label (Eq. finite_horizon_hazard)
# =============================================================================
def hard_limit_violation(log: Dict[str, np.ndarray], cons: ConstraintParams) -> np.ndarray:
    """Pointwise indicator of q_k outside the hard operational safe set Q_s
    (Eq. operational_safe_set): true electrical limits only -- bus voltages,
    feeder currents, and DC-link voltage (Eqs. voltage/current/dc_voltage_constraints).
    Actuator/setpoint-box limits are excluded: staying inside the P/Q box is
    an actuator-headroom margin (already captured by SMI's psi_u term), not
    a physical plant-safety violation in the same sense as V/I/Vdc.
    Evaluated on the *_true (hidden-truth) fields, exactly like the original
    z_k^unsafe hard_violation term in validation.py.
    """
    V1, V2 = log["V1_true"], log["V2_true"]
    I1, I2 = log["I1_true"], log["I2_true"]
    Vdc = log["Vdc_true"]
    return (
        (V1 < cons.Vac_min) | (V1 > cons.Vac_max) |
        (V2 < cons.Vac_min) | (V2 > cons.Vac_max) |
        (I1 > cons.I_max) | (I2 > cons.I_max) |
        (Vdc < cons.Vdc_min) | (Vdc > cons.Vdc_max)
    )


def finite_horizon_label(log: Dict[str, np.ndarray], cons: ConstraintParams, H: int) -> np.ndarray:
    """z_k^(H), Eq. finite_horizon_hazard: 1 iff the hard-limit violation
    indicator is true at any sample j in {0,...,H} after k (inclusive of k
    itself). Implemented as a reverse (right-to-left) rolling-OR over the
    pointwise violation indicator, so it is exact (not an approximation) and
    O(n) regardless of H.
    """
    viol = hard_limit_violation(log, cons).astype(bool)
    n = len(viol)
    z = np.zeros(n, dtype=bool)
    # z[k] = OR(viol[k .. min(k+H, n-1)]); build by scanning from the end.
    running = False
    # Use a simple ring/count approach: since OR over a *shrinking-from-the-right*
    # window as k increases is not directly separable, do it directly with a
    # sliding window via cumulative "distance to next True" trick.
    next_true = np.full(n, np.inf)
    last = np.inf
    for k in range(n - 1, -1, -1):
        if viol[k]:
            last = k
        next_true[k] = last
    idx = np.arange(n)
    z = (next_true - idx) <= H
    return z.astype(int)


def samples_to_horizon(T_H: float, Ts: float) -> int:
    """H = round(T_H / Ts), the verification horizon in samples for a given
    physical horizon T_H (s) at sampling interval Ts (s)."""
    return int(round(T_H / Ts))


# =============================================================================
# Calibration dataset assembly
# =============================================================================
@dataclass
class TrajectoryData:
    """One (scenario, seed) calibration/evaluation trajectory's readiness
    vectors, finite-horizon labels (computed once at the *stored* horizon),
    and everything needed to reconstruct beta_k/CRI_k offline for any omega."""
    scenario: str
    seed: int
    Gamma: np.ndarray          # (n, 5): [RI, CI, SMI, AI, NCI] per sample
    sigma_hat_sq: np.ndarray   # (n,)
    viol: np.ndarray           # (n,) pointwise hard-limit violation (bool), H-independent
    Ts: float


def collect_trajectory(log: Dict[str, np.ndarray], cons: ConstraintParams, scenario: str, seed: int, Ts: float) -> TrajectoryData:
    Gamma = np.stack([log[c] for c in INDICATOR_ORDER], axis=1)
    return TrajectoryData(
        scenario=scenario, seed=seed, Gamma=Gamma,
        sigma_hat_sq=log["sigma_hat_sq"].copy(),
        viol=hard_limit_violation(log, cons),
        Ts=Ts,
    )


def label_for_horizon(traj: TrajectoryData, H: int) -> np.ndarray:
    n = len(traj.viol)
    next_true = np.full(n, np.inf)
    last = np.inf
    for k in range(n - 1, -1, -1):
        if traj.viol[k]:
            last = k
        next_true[k] = last
    idx = np.arange(n)
    return ((next_true - idx) <= H).astype(int)


# =============================================================================
# Weight calibration (Eq. weight_calibration): risk-weighted least squares
# on the probability simplex
# =============================================================================
@dataclass
class WeightCalibrationResult:
    omega: np.ndarray
    objective: float
    n_iter: int
    n_feval: int
    runtime_s: float
    success: bool
    n_safe: int
    n_hazard: int


def calibrate_weights(Gamma: np.ndarray, z: np.ndarray, lambda_H: float,
                       omega0: Optional[np.ndarray] = None) -> WeightCalibrationResult:
    """Solve Eq. weight_calibration:
        omega* = argmin_omega (1/|Is|) sum_{Is} (1 - omega.Gamma_k)^2
                              + lambda_H (1/|Ih|) sum_{Ih} (omega.Gamma_k)^2
        s.t. omega >= 0, 1^T omega = 1
    A convex QP over the 5-dimensional simplex, solved with SLSQP and an
    analytic gradient (fast and robust at this problem size).
    """
    z = np.asarray(z, dtype=bool)
    A_s = Gamma[~z]
    A_h = Gamma[z]
    n_s, n_h = len(A_s), len(A_h)
    if n_s == 0 or n_h == 0:
        raise ValueError(f"Degenerate calibration set: n_safe={n_s}, n_hazard={n_h} "
                          "(need at least one sample of each class).")

    P = (A_s.T @ A_s) / n_s + lambda_H * (A_h.T @ A_h) / n_h
    q = -2.0 * (A_s.T @ np.ones(n_s)) / n_s

    def objective(w):
        return float(w @ P @ w + q @ w + 1.0)

    def gradient(w):
        return 2.0 * P @ w + q

    w0 = np.full(5, 0.2) if omega0 is None else np.asarray(omega0, dtype=float)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * 5

    t0 = time.perf_counter()
    res = minimize(objective, w0, jac=gradient, method="SLSQP", bounds=bounds,
                    constraints=cons, options={"maxiter": 200, "ftol": 1e-12})
    runtime_s = time.perf_counter() - t0

    omega = np.clip(res.x, 0.0, None)
    omega = omega / omega.sum()  # exact re-projection onto the simplex (SLSQP tolerance cleanup)

    return WeightCalibrationResult(
        omega=omega, objective=float(res.fun), n_iter=int(res.nit), n_feval=int(res.nfev),
        runtime_s=runtime_s, success=bool(res.success), n_safe=n_s, n_hazard=n_h,
    )


# =============================================================================
# Offline reconstruction of the dynamic CRI filter for arbitrary omega
# (Eqs. instantaneous_cri, adaptive_beta, dynamic_CRI) -- pure post-processing,
# identical formulas to cri.py, reused here to avoid re-deriving beta_k.
# =============================================================================
def reconstruct_cri_trajectory(traj: TrajectoryData, omega: np.ndarray, crip: CRIParams,
                                cri0: float = 0.5) -> np.ndarray:
    """CRI_k for one trajectory under a candidate omega, using the SAME
    sigma_hat_sq (hence beta_k) already produced by the original closed-loop
    run -- beta_k does not depend on omega (Eq. adaptive_beta), so this is an
    exact, not approximate, reconstruction of what CRISupervisor.step() would
    have computed for this omega, given the same Gamma_k/sigma_hat_sq stream."""
    n = len(traj.sigma_hat_sq)
    cri = np.zeros(n)
    cri_prev = cri0
    for k in range(n):
        cri_hat = float(omega @ traj.Gamma[k])
        beta_k = crip.beta_min + (crip.beta_max - crip.beta_min) * np.exp(
            -crip.kappa_beta * traj.sigma_hat_sq[k] / max(crip.sigma_y0_sq, 1e-12))
        cri_prev = (1.0 - beta_k) * cri_prev + beta_k * cri_hat
        cri[k] = cri_prev
    return cri


# =============================================================================
# Risk-constrained activation threshold (Eq. risk_constrained_threshold)
# =============================================================================
@dataclass
class ThresholdCalibrationResult:
    eta_A: float
    feasible: bool
    achieved_risk: float
    achieved_availability: float
    grid: np.ndarray
    risk_curve: np.ndarray
    availability_curve: np.ndarray


def calibrate_activation_threshold(CRI: np.ndarray, z: np.ndarray, eps_risk: float,
                                    grid_step: float = 0.002, eps0: float = 1e-9) -> ThresholdCalibrationResult:
    """eta_A* = argmax_eta A_hat(eta) s.t. R_hat_act(eta) <= eps_risk (Eq.
    risk_constrained_threshold), found by an exhaustive grid sweep over
    eta in [0,1] (both A_hat and, empirically, R_hat_act are monotone
    non-increasing in eta, so a fine grid is exact to within grid_step and
    far more robust than a root-find on noisy empirical curves).
    Infeasible (no eta on the grid satisfies the risk constraint) is reported
    explicitly rather than silently falling back to a default."""
    z = np.asarray(z, dtype=float)
    grid = np.arange(0.0, 1.0 + grid_step, grid_step)
    risk_curve = np.zeros_like(grid)
    avail_curve = np.zeros_like(grid)
    N = len(CRI)
    for i, eta in enumerate(grid):
        active = CRI >= eta
        n_active = active.sum()
        risk_curve[i] = (active * z).sum() / (n_active + eps0)
        avail_curve[i] = n_active / N

    feasible_mask = risk_curve <= eps_risk
    if not feasible_mask.any():
        return ThresholdCalibrationResult(
            eta_A=float("nan"), feasible=False, achieved_risk=float("nan"),
            achieved_availability=float("nan"), grid=grid,
            risk_curve=risk_curve, availability_curve=avail_curve,
        )

    # Among feasible eta, take the one maximizing availability.
    best_idx = np.argmax(np.where(feasible_mask, avail_curve, -np.inf))
    return ThresholdCalibrationResult(
        eta_A=float(grid[best_idx]), feasible=True,
        achieved_risk=float(risk_curve[best_idx]), achieved_availability=float(avail_curve[best_idx]),
        grid=grid, risk_curve=risk_curve, availability_curve=avail_curve,
    )


# =============================================================================
# Monitor/Delay margins (Eq. relative_thresholds): grid search minimizing
# supervisory switching, subject to 0 < Delta_M < Delta_D < eta_A*
# =============================================================================
def _classify(cri: np.ndarray, eta_A: float, eta_M: float, eta_D: float) -> np.ndarray:
    code = np.zeros(len(cri), dtype=int)
    code[cri >= eta_A] = 3
    code[(cri >= eta_M) & (cri < eta_A)] = 2
    code[(cri >= eta_D) & (cri < eta_M)] = 1
    code[cri < eta_D] = 0
    return code


def calibrate_relative_margins(CRI: np.ndarray, eta_A: float,
                                dM_grid: Tuple[float, ...] = (0.04, 0.06, 0.08, 0.10, 0.12),
                                dD_grid: Tuple[float, ...] = (0.12, 0.16, 0.20, 0.24, 0.28, 0.32),
                                eta_D_floor: float = 1e-4) -> Tuple[float, float]:
    """Delta_M, Delta_D selected on the calibration trajectory to minimize
    the number of classification transitions (supervisory switching),
    subject to 0 < Delta_M < Delta_D < eta_A. `eta_D_floor` guarantees a
    strictly positive eta_D even in the degenerate case eta_A -> 0 (a very
    loose eps_risk can make eta_A*=0.0 the risk-feasible, availability-
    maximizing choice), so CRIParams's 0 <= eta_D < eta_M < eta_A ordering
    assertion is never violated regardless of how permissive eta_A is."""
    best = None
    for dM in dM_grid:
        for dD in dD_grid:
            if not (0 < dM < dD < eta_A) or (eta_A - dD) <= eta_D_floor:
                continue
            eta_M, eta_D = eta_A - dM, eta_A - dD
            cls = _classify(CRI, eta_A, eta_M, eta_D)
            switches = int(np.sum(np.diff(cls) != 0))
            if best is None or switches < best[0]:
                best = (switches, dM, dD)
    if best is not None:
        return best[1], best[2]

    # Fallback: proportional split of eta_A, only used when the grid is
    # incompatible with a very small eta_A. Enforce an absolute floor on
    # eta_D so the ordering constraint holds even as eta_A -> 0.
    dD = max(0.6 * eta_A, eta_A - eta_D_floor) if eta_A > eta_D_floor else eta_A / 2.0
    dD = min(dD, eta_A - eta_D_floor) if eta_A > eta_D_floor else 0.0
    dM = 0.5 * dD
    if eta_A <= eta_D_floor:
        # eta_A itself is (numerically) at the floor: the whole [0, eta_A]
        # interval is too thin to place three ordered thresholds in: fall
        # back to two coincident-in-practice but strictly ordered margins.
        dM, dD = eta_A * 0.34, eta_A * 0.67
    return dM, dD


# =============================================================================
# Full calibration orchestration for one LOSO fold (or a pooled/no-holdout set)
# =============================================================================
@dataclass
class CalibrationOutcome:
    omega: np.ndarray
    eta_A: float
    eta_M: float
    eta_D: float
    delta_M: float
    delta_D: float
    feasible: bool
    weight_result: WeightCalibrationResult
    threshold_result: ThresholdCalibrationResult
    cal_risk: float
    cal_availability: float


def calibrate_dc_cri(trajectories: List[TrajectoryData], H: int, eps_risk: float,
                      lambda_H: float, crip_template: CRIParams,
                      cri0: float = 0.5) -> CalibrationOutcome:
    """End-to-end calibration (weights, then thresholds) on a pooled set of
    calibration trajectories, exactly implementing Algorithm 1's per-fold
    calibration steps (everything except the final held-out evaluation,
    which is a genuine closed-loop simulation done by the caller)."""
    Gamma_all = np.concatenate([t.Gamma for t in trajectories], axis=0)
    z_all = np.concatenate([label_for_horizon(t, H) for t in trajectories], axis=0)

    wres = calibrate_weights(Gamma_all, z_all, lambda_H)

    cri_per_traj = [reconstruct_cri_trajectory(t, wres.omega, crip_template, cri0) for t in trajectories]
    CRI_all = np.concatenate(cri_per_traj, axis=0)

    tres = calibrate_activation_threshold(CRI_all, z_all, eps_risk)

    if tres.feasible:
        # A tiny positive floor on eta_A itself: eta_A=0.0 exactly is a valid,
        # risk-feasible calibration outcome (a very loose eps_risk can make
        # "activate always" the availability-maximizing choice), but three
        # strictly-ordered thresholds cannot be placed in a zero-width
        # interval. Clipping to 1e-3 changes online behavior negligibly
        # (CRI_k >= 0 always) while keeping CRIParams's ordering assertion
        # satisfiable in every case.
        eta_A = max(tres.eta_A, 1e-3)
        dM, dD = calibrate_relative_margins(CRI_all, eta_A)
        eta_M, eta_D = eta_A - dM, eta_A - dD
    else:
        # No feasible eta_A under this eps_risk: report infeasibility, and
        # fall back to the most conservative point on the grid (eta_A=1.0)
        # purely so downstream evaluation code has a well-defined (maximally
        # conservative) CRIParams to run -- this fallback is never reported
        # as if it satisfied the requested risk tolerance.
        dM, dD = 0.10, 0.20
        eta_A, eta_M, eta_D = 1.0, 1.0 - dM, 1.0 - dD

    return CalibrationOutcome(
        omega=wres.omega, eta_A=eta_A, eta_M=eta_M, eta_D=eta_D,
        delta_M=dM, delta_D=dD, feasible=tres.feasible,
        weight_result=wres, threshold_result=tres,
        cal_risk=tres.achieved_risk, cal_availability=tres.achieved_availability,
    )


def dc_cri_params_from_outcome(outcome: CalibrationOutcome, crip_template: CRIParams) -> CRIParams:
    """Build a full CRIParams for the calibrated DC-CRI, sharing every
    non-calibrated parameter (N, lambda_s, r0, a0, sigma_y0_sq, epsilon_0s,
    beta_min/max, kappa_beta, kappa_min, delta_max, eta_S, eta_N) with the
    EC-CRI template -- only omega and eta_A/M/D are calibrated."""
    return CRIParams(
        N=crip_template.N, Na=crip_template.Na, lambda_s=crip_template.lambda_s,
        r0=crip_template.r0, a0=crip_template.a0, sigma_y0_sq=crip_template.sigma_y0_sq,
        epsilon_0s=crip_template.epsilon_0s,
        omega_R=float(outcome.omega[0]), omega_C=float(outcome.omega[1]),
        omega_S=float(outcome.omega[2]), omega_A=float(outcome.omega[3]),
        omega_N=float(outcome.omega[4]),
        eta_A=outcome.eta_A, eta_M=outcome.eta_M, eta_D=outcome.eta_D,
        eta_S=crip_template.eta_S, eta_N=crip_template.eta_N,
        beta_min=crip_template.beta_min, beta_max=crip_template.beta_max,
        kappa_beta=crip_template.kappa_beta, kappa_min=crip_template.kappa_min,
        delta_max=crip_template.delta_max,
        epsilon_regularization=crip_template.epsilon_regularization,
    )
