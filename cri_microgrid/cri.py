"""The Control Readiness Index (CRI): five readiness indicators, their
combination into a bounded scalar, dynamic (recursive) filtering, readiness
classification, adaptation scaling, and the digital-twin escalation flag.

Every equation from the paper is implemented as its own small, pure
function (no equation is buried inside a larger opaque routine), so each
one is independently testable and directly traceable back to the
manuscript. `CRISupervisor` is the only stateful object in this module: it
owns the moving-data windows and the previous-step values the recursive
formulas need, and orchestrates the pure functions in the same order as
Algorithm 1 of the paper.

CRI blindness: CRISupervisor.step() takes only (u_applied, y_measured,
constraints) -- never the plant object, never hidden_truth. This is what
makes the "measurement-driven" claim in the paper enforceable in code
rather than just asserted in prose.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Optional, Tuple, List
import numpy as np

from .config import CRIParams, ConstraintParams, DEFAULT_D_u, DEFAULT_D_y
from .sensitivity import estimate_sensitivity, coupling_raw, SensitivityResult

CLASS_ACTIVATE, CLASS_MONITOR, CLASS_DELAY, CLASS_REJECT = "Activate", "Monitor", "Delay", "Reject"


# =============================================================================
# Pure indicator functions
# =============================================================================
def compute_RI(rho: float, r0: float) -> float:
    """Eq.: RI_k = 1 - exp(-rho_k/r0)."""
    return float(1.0 - np.exp(-rho / max(r0, 1e-12)))


def compute_CI(S_hat: np.ndarray, eps0: float = 1e-9) -> float:
    """Eq.: CI_k = 1 - c_k (column-normalized coupling measure)."""
    return float(1.0 - coupling_raw(S_hat, eps0))


def sat(z: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, z)))


def compute_SMI(psi_list: List[float]) -> float:
    """Eq.: SMI_k = min_ell sat_[0,1](psi_ell,k)."""
    return float(min(sat(p, 0.0, 1.0) for p in psi_list)) if psi_list else 1.0


def microgrid_margins(y_meas: np.ndarray, u_applied: np.ndarray, cons: ConstraintParams,
                       eps0: float = 1e-9) -> List[float]:
    """psi_V, psi_I, psi_dc, psi_u for the microgrid case study (Sec. IV-B)."""
    V1, V2, Vdc, I1, I2 = y_meas[0], y_meas[1], y_meas[2], y_meas[3], y_meas[4]
    P1, Q1, P2, Q2 = u_applied[0], u_applied[1], u_applied[2], u_applied[3]

    Vnom = 0.5 * (cons.Vac_min + cons.Vac_max) if False else 1.0
    psi_V = min(
        min((V1 - cons.Vac_min) / (Vnom - cons.Vac_min + eps0), (cons.Vac_max - V1) / (cons.Vac_max - Vnom + eps0)),
        min((V2 - cons.Vac_min) / (Vnom - cons.Vac_min + eps0), (cons.Vac_max - V2) / (cons.Vac_max - Vnom + eps0)),
    )
    psi_I = min((cons.I_max - I1) / (cons.I_max + eps0), (cons.I_max - I2) / (cons.I_max + eps0))

    Vdc_nom = 1.0
    psi_dc = min((Vdc - cons.Vdc_min) / (Vdc_nom - cons.Vdc_min + eps0),
                 (cons.Vdc_max - Vdc) / (cons.Vdc_max - Vdc_nom + eps0))

    # Factor of 2: normalized so the MIDPOINT of each admissible setpoint
    # range corresponds to unit headroom (psi_u=1), matching how psi_V,
    # psi_I, psi_dc all already reach 1 at their own nominal point -- not
    # a fixed ceiling of 0.5 at the very best achievable operating point.
    # (compute_SMI() saturates each margin to [0,1] anyway, so a value
    # >1 in the inner half of the range is intentional and harmless.)
    psi_u = min(
        2 * (P1 - cons.P_min) / (cons.P_max - cons.P_min + eps0), 2 * (cons.P_max - P1) / (cons.P_max - cons.P_min + eps0),
        2 * (Q1 - cons.Q_min) / (cons.Q_max - cons.Q_min + eps0), 2 * (cons.Q_max - Q1) / (cons.Q_max - cons.Q_min + eps0),
        2 * (P2 - cons.P_min) / (cons.P_max - cons.P_min + eps0), 2 * (cons.P_max - P2) / (cons.P_max - cons.P_min + eps0),
        2 * (Q2 - cons.Q_min) / (cons.Q_max - cons.Q_min + eps0), 2 * (cons.Q_max - Q2) / (cons.Q_max - cons.Q_min + eps0),
    )
    return [psi_V, psi_I, psi_dc, psi_u]


def compute_adaptability_raw(ybar_k: np.ndarray, ybar_k_minus_Na: np.ndarray, eps0: float = 1e-9) -> float:
    """Eq.: a_k = ||ybar_k - ybar_{k-Na}|| / (||ybar_{k-Na}|| + eps0)."""
    num = np.linalg.norm(ybar_k - ybar_k_minus_Na)
    den = np.linalg.norm(ybar_k_minus_Na) + eps0
    return float(num / den)


def compute_AI(a_k: float, a0: float) -> float:
    """Eq.: AI_k = exp(-a_k/a0)."""
    return float(np.exp(-a_k / max(a0, 1e-12)))


def estimate_noise_variance(y_tilde_window: np.ndarray) -> float:
    """Second-difference noise-variance estimator: for iid noise of
    variance sigma^2 superimposed on a smooth trend, E[(y_k-2y_{k-1}+y_{k-2})^2]
    = 6*sigma^2, and second-differencing cancels any locally-linear trend, so
    this stays a genuine noise estimate rather than reacting to the
    ramps/drift that the Adaptability Index already handles separately.
    y_tilde_window: (n_y, T) normalized measurement history.
    """
    T = y_tilde_window.shape[1]
    if T < 3:
        return 0.0
    d2 = y_tilde_window[:, 2:] - 2.0 * y_tilde_window[:, 1:-1] + y_tilde_window[:, :-2]
    return float(np.mean(d2 ** 2) / 6.0)


def compute_NCI(sigma_hat_sq: float, sigma_y0_sq: float, epsilon_s: float, epsilon_0s: float) -> float:
    """Eq.: NCI_k = exp(-sigma_hat^2/sigma_y0^2) * exp(-eps_s^2/eps_0s^2)."""
    t1 = np.exp(-sigma_hat_sq / max(sigma_y0_sq, 1e-12))
    t2 = np.exp(-(epsilon_s ** 2) / max(epsilon_0s ** 2, 1e-12))
    return float(t1 * t2)


# =============================================================================
# CRI combination, dynamic filtering, classification, scaling
# =============================================================================
def compute_instantaneous_cri(Gamma: np.ndarray, weights: np.ndarray) -> float:
    """Eq.: CRI_hat_k = 1^T Omega Gamma_k = sum_i omega_i * gamma_i,k."""
    return float(np.dot(weights, Gamma))


def compute_beta(sigma_hat_sq: float, sigma_y0_sq: float, beta_min: float, beta_max: float,
                  kappa_beta: float) -> float:
    """Eq.: beta_k = beta_min + (beta_max-beta_min)*exp(-kappa_beta*sigma_hat^2/sigma_y0^2)."""
    return float(beta_min + (beta_max - beta_min) * np.exp(-kappa_beta * sigma_hat_sq / max(sigma_y0_sq, 1e-12)))


def update_dynamic_cri(cri_prev: float, cri_hat: float, beta_k: float) -> float:
    """Eq.: CRI_k = (1-beta_k)*CRI_{k-1} + beta_k*CRI_hat_k."""
    return float((1.0 - beta_k) * cri_prev + beta_k * cri_hat)


def classify_readiness(cri_k: float, eta_A: float, eta_M: float, eta_D: float) -> str:
    """Eq.: readiness classification into Activate/Monitor/Delay/Reject."""
    if cri_k >= eta_A:
        return CLASS_ACTIVATE
    if cri_k >= eta_M:
        return CLASS_MONITOR
    if cri_k >= eta_D:
        return CLASS_DELAY
    return CLASS_REJECT


def compute_activation(class_k: str, s_prev: int) -> int:
    """Eq.: s_k -- Activate->1, Monitor->hold previous, Delay/Reject->0."""
    if class_k == CLASS_ACTIVATE:
        return 1
    if class_k == CLASS_MONITOR:
        return int(s_prev)
    return 0


def compute_kappa(cri_k: float, eta_D: float, eta_A: float, kappa_min: float) -> float:
    """Eq.: kappa_k = sat_[kappa_min,1]((CRI_k-eta_D)/(eta_A-eta_D))."""
    span = max(eta_A - eta_D, 1e-9)
    return sat((cri_k - eta_D) / span, kappa_min, 1.0)


def compute_dt_flag(cri_k: float, smi_k: float, nci_k: float, eta_D: float, eta_S: float, eta_N: float) -> int:
    """Eq.: d_k^DT -- request digital-twin/operator validation."""
    return int((cri_k < eta_D) or (smi_k < eta_S) or (nci_k < eta_N))


# =============================================================================
# Stateful orchestration
# =============================================================================
@dataclass
class CRIStepResult:
    RI: float
    CI: float
    SMI: float
    AI: float
    NCI: float
    CRI_hat: float
    CRI: float
    beta: float
    kappa: float
    s: int
    dt_flag: int
    cls: str
    rho: float
    epsilon_s: float
    condition_number: float
    excitation_energy: float
    sigma_hat_sq: float
    a_raw: float
    S_hat: np.ndarray
    delta_bar_next: float   # SMI_k, exposed for next-step probe scaling


class CRISupervisor:
    """Owns the moving-data windows and recursive state; call `.step()`
    once per sample with (u_applied, y_measured) only."""

    def __init__(self, params: CRIParams, D_u: Optional[np.ndarray] = None,
                 D_y: Optional[np.ndarray] = None, forgetting_lambda: float = 0.97,
                 cri0: float = 0.5, n_u: int = 4, n_y: int = 7,
                 margin_fn: Optional[Callable[[np.ndarray, np.ndarray, ConstraintParams], List[float]]] = None):
        """margin_fn(y_meas, u_applied, cons) -> list of normalized margins
        psi_ell,k, one per operational constraint (Eq. 22-23). Defaults to
        the microgrid-specific `microgrid_margins` (Sec. IV-B); pass your
        own to reuse the CRI engine on a different plant/system -- see
        cri_microgrid/interfaces.py and notebooks/11_bring_your_own.ipynb."""
        self.p = params
        self.D_u = np.asarray(D_u if D_u is not None else DEFAULT_D_u, dtype=float)
        self.D_y = np.asarray(D_y if D_y is not None else DEFAULT_D_y, dtype=float)
        self.forgetting_lambda = forgetting_lambda
        self.n_u, self.n_y = n_u, n_y
        self.margin_fn = margin_fn if margin_fn is not None else microgrid_margins

        self.du_win: Deque[np.ndarray] = deque(maxlen=params.N)
        self.dy_win: Deque[np.ndarray] = deque(maxlen=params.N)
        self.ytilde_hist: Deque[np.ndarray] = deque(maxlen=2 * params.Na)

        self.u_prev: Optional[np.ndarray] = None
        self.y_prev: Optional[np.ndarray] = None
        self.CRI_prev: float = cri0
        self.s_prev: int = 0
        self.smi_prev: float = 1.0

    def reset(self, cri0: float = 0.5):
        self.du_win.clear(); self.dy_win.clear(); self.ytilde_hist.clear()
        self.u_prev = None; self.y_prev = None
        self.CRI_prev = cri0; self.s_prev = 0; self.smi_prev = 1.0

    @property
    def smi_for_probing(self) -> float:
        """Last computed SMI_k, used by sim.py to size THIS step's probe
        amplitude via Eq. (13) before the command is actually applied."""
        return self.smi_prev

    def step(self, u_applied: np.ndarray, y_meas: np.ndarray, cons: ConstraintParams) -> CRIStepResult:
        p = self.p
        u_applied = np.asarray(u_applied, dtype=float)
        y_meas = np.asarray(y_meas, dtype=float)
        y_tilde = y_meas / self.D_y

        if self.u_prev is None:
            du = np.zeros_like(u_applied)
            dy = np.zeros_like(y_meas)
        else:
            du = u_applied - self.u_prev
            dy = y_meas - self.y_prev

        self.du_win.append(du / self.D_u)
        self.dy_win.append(dy / self.D_y)
        self.ytilde_hist.append(y_tilde)

        DU = np.stack(self.du_win, axis=1)   # (n_u, T)
        DY = np.stack(self.dy_win, axis=1)   # (n_y, T)
        T = DU.shape[1]
        idx = np.arange(T)
        w = self.forgetting_lambda ** (T - 1 - idx)

        sens = estimate_sensitivity(DU, DY, w, p.lambda_s)

        RI = compute_RI(sens.rho, p.r0)
        CI = compute_CI(sens.S_hat)

        psi = self.margin_fn(y_meas, u_applied, cons)
        SMI = compute_SMI(psi)

        Yt = np.stack(self.ytilde_hist, axis=1)  # (n_y, Th)
        Na = p.Na
        if Yt.shape[1] >= 2 * Na:
            ybar_k = Yt[:, -Na:].mean(axis=1)
            ybar_km = Yt[:, -2 * Na:-Na].mean(axis=1)
            a_raw = compute_adaptability_raw(ybar_k, ybar_km)
        else:
            a_raw = 0.0
        AI = compute_AI(a_raw, p.a0)

        sigma_hat_sq = estimate_noise_variance(Yt)
        NCI = compute_NCI(sigma_hat_sq, p.sigma_y0_sq, sens.epsilon_s, p.epsilon_0s)

        Gamma = np.array([RI, CI, SMI, AI, NCI])
        CRI_hat = compute_instantaneous_cri(Gamma, p.weights())
        beta = compute_beta(sigma_hat_sq, p.sigma_y0_sq, p.beta_min, p.beta_max, p.kappa_beta)
        CRI = update_dynamic_cri(self.CRI_prev, CRI_hat, beta)

        cls = classify_readiness(CRI, p.eta_A, p.eta_M, p.eta_D)
        s_k = compute_activation(cls, self.s_prev)
        kappa_k = compute_kappa(CRI, p.eta_D, p.eta_A, p.kappa_min)
        dt_flag = compute_dt_flag(CRI, SMI, NCI, p.eta_D, p.eta_S, p.eta_N)

        self.u_prev, self.y_prev = u_applied.copy(), y_meas.copy()
        self.CRI_prev, self.s_prev, self.smi_prev = CRI, s_k, SMI

        return CRIStepResult(
            RI=RI, CI=CI, SMI=SMI, AI=AI, NCI=NCI, CRI_hat=CRI_hat, CRI=CRI,
            beta=beta, kappa=kappa_k, s=s_k, dt_flag=dt_flag, cls=cls,
            rho=sens.rho, epsilon_s=sens.epsilon_s, condition_number=sens.condition_number,
            excitation_energy=sens.excitation_energy, sigma_hat_sq=sigma_hat_sq, a_raw=a_raw,
            S_hat=sens.S_hat, delta_bar_next=SMI,
        )
